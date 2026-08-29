# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine inference service backed by one TensorRT-LLM LLM instance."""

import asyncio
import functools
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import Executor
from dataclasses import replace
from typing import Any

import grpc
from openengine.v1 import (
    error_pb2,
    generation_pb2,
    kv_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

import tensorrt_llm
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.llmapi.disagg_utils import get_global_disagg_request_id
from tensorrt_llm.logger import logger
from tensorrt_llm.scheduling_params import SchedulingParams

from .converters import decode_handoff, encode_handoff, to_priority, to_sampling_params

SCHEMA_RELEASE = "768a93c7b44e40f28c692ad0b471a8f2"
SCHEMA_REVISION = 1
MINIMUM_CLIENT_REVISION = 1


def _arg(llm: object, name: str, default: Any = None) -> Any:
    args = getattr(llm, "args", None)
    value = getattr(args, name, None)
    if value is not None:
        return value
    parallel = getattr(args, "parallel_config", None)
    return getattr(parallel, name, default)


def _data_parallel_size(llm: object) -> int:
    if bool(_arg(llm, "enable_attention_dp", False)):
        size = _arg(llm, "tensor_parallel_size", None)
        if size is None:
            size = _arg(llm, "tp_size", None)
        if size is None:
            size = _arg(llm, "data_parallel_size", 1)
        return max(1, int(size))
    return max(1, int(_arg(llm, "data_parallel_size", 1)))


class OpenEngineServicer(
    openengine_pb2_grpc.InferenceServicer,
    openengine_pb2_grpc.ControlServicer,
):
    """Implement OpenEngine generation and server/model discovery."""

    def __init__(
        self,
        llm: object,
        model: str,
        role: int,
        internal_disagg_auth_key: str | None = None,
        input_processor_executor: Executor | None = None,
        response_drain_timeout: float = 30.0,
    ) -> None:
        if role not in (
            server_pb2.ENGINE_ROLE_AGGREGATED,
            server_pb2.ENGINE_ROLE_PREFILL,
            server_pb2.ENGINE_ROLE_DECODE,
        ):
            raise ValueError(f"Unsupported OpenEngine role {role}")
        if getattr(llm, "tokenizer", None) is None:
            raise ValueError("OpenEngine requires tokenizer initialization")
        if role != server_pb2.ENGINE_ROLE_AGGREGATED and not internal_disagg_auth_key:
            raise ValueError(
                "OpenEngine prefill and decode roles require internal_request_auth_key"
            )
        if response_drain_timeout <= 0:
            raise ValueError("response_drain_timeout must be greater than zero")
        self.llm = llm
        self.model = model
        self.model_id = str(_arg(llm, "model", model) or model)
        self._accepted_model_names = {model, self.model_id}
        self.role = role
        self.internal_disagg_auth_key = internal_disagg_auth_key
        self._input_processor_executor = input_processor_executor
        self._response_drain_timeout = response_drain_timeout
        self._disagg_node_id = uuid.getnode() % 256
        self.instance_id = str(uuid.uuid4())
        self._requests: dict[str, object | None] = {}

    @staticmethod
    def _request_metadata(context: grpc.aio.ServicerContext) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for item in context.invocation_metadata():
            try:
                key, value = item
            except (TypeError, ValueError):
                key, value = item.key, item.value
            key = str(key).lower()
            if key.startswith("openengine-") and key in metadata:
                raise ValueError(f"Duplicate reserved gRPC metadata key {key!r}")
            metadata[key] = value.decode("ascii") if isinstance(value, bytes) else str(value)
        return metadata

    @staticmethod
    def _metadata_int(
        metadata: dict[str, str],
        key: str,
        minimum: int,
        maximum: int,
    ) -> int | None:
        if key not in metadata:
            return None
        value = metadata[key]
        digits = value[1:] if value.startswith("-") and minimum < 0 else value
        if not digits or not digits.isdecimal():
            raise ValueError(f"gRPC metadata {key!r} must be a base-10 integer")
        parsed = int(value, 10)
        if not minimum <= parsed <= maximum:
            raise ValueError(f"gRPC metadata {key!r} must be in [{minimum}, {maximum}]")
        return parsed

    def _scheduling_params(self, target_dp_rank: int | None) -> SchedulingParams | None:
        if target_dp_rank is None:
            return None
        data_parallel_size = _data_parallel_size(self.llm)
        if target_dp_rank >= data_parallel_size:
            raise ValueError(
                f"data_parallel_rank {target_dp_rank} is outside the configured DP size "
                f"{data_parallel_size}"
            )
        attention_dp = bool(_arg(self.llm, "enable_attention_dp", False))
        if not attention_dp or getattr(self.llm, "_on_trt_backend", False):
            if target_dp_rank == 0:
                return None
            raise ValueError(
                "A nonzero data_parallel_rank requires the PyTorch backend with attention DP"
            )
        return SchedulingParams(attention_dp_rank=target_dp_rank, attention_dp_relax=False)

    def _validate_generate(
        self,
        request: generation_pb2.GenerateRequest,
        target_dp_rank: int | None,
    ) -> None:
        if not request.request_id:
            raise ValueError("request_id must not be empty")
        if request.request_id in self._requests:
            raise ValueError(f"Request {request.request_id!r} is already active")
        if request.model and request.model not in self._accepted_model_names:
            raise ValueError(f"Unknown model {request.model!r}")
        if request.WhichOneof("input") is None:
            raise ValueError("Generate requires prompt or token_ids input")
        if request.media:
            raise ValueError("Multimodal input is not implemented by the OpenEngine adapter")
        if request.lora_name:
            raise ValueError("LoRA selection is not implemented by the OpenEngine adapter")
        if request.guided.WhichOneof("guide") is not None or request.guided.backend:
            raise ValueError("Guided decoding is not implemented by the OpenEngine adapter")
        if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
            raise ValueError("Prefix-cache bypass is not supported by TensorRT-LLM")
        if target_dp_rank is not None:
            self._scheduling_params(target_dp_rank)

        has_session = request.kv.HasField("session")
        if self.role == server_pb2.ENGINE_ROLE_DECODE:
            if not has_session:
                raise ValueError("Decode requests require a prefill KV session")
            if not request.kv.session.session_id:
                raise ValueError("Decode KV session must have a session_id")
            if request.kv.session.transfer_backend != "tensorrt_llm":
                raise ValueError("Decode KV session was not produced by TensorRT-LLM")
        elif self.role == server_pb2.ENGINE_ROLE_PREFILL:
            if has_session:
                raise ValueError("Prefill requests cannot consume a KV session")
        elif has_session:
            raise ValueError("Aggregated requests cannot consume a KV session")

    def _disaggregated_params(
        self,
        request: generation_pb2.GenerateRequest,
        target_dp_rank: int | None,
    ) -> DisaggregatedParams | None:
        if self.role == server_pb2.ENGINE_ROLE_AGGREGATED:
            return None
        if self.role == server_pb2.ENGINE_ROLE_PREFILL:
            return DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=get_global_disagg_request_id(self._disagg_node_id),
                ctx_dp_rank=target_dp_rank,
                schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            )
        return decode_handoff(request.kv.session, self.internal_disagg_auth_key)

    def _context_info_endpoint(self) -> str | None:
        params = getattr(self.llm, "disaggregated_params", None)
        if not isinstance(params, dict):
            return None
        endpoint = params.get("ctx_info_endpoint")
        if isinstance(endpoint, str):
            return endpoint or None
        if isinstance(endpoint, (list, tuple)):
            return next((item for item in endpoint if isinstance(item, str) and item), None)
        return None

    def _token_text(self, token_id: int) -> str:
        tokenizer = getattr(self.llm, "tokenizer", None)
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode([token_id], skip_special_tokens=False)
        except (AttributeError, TypeError, ValueError):
            return ""

    @staticmethod
    def _candidate_items(
        values: dict[int, Any],
        candidate_count: int | None,
    ) -> list[tuple[int, Any]]:
        if candidate_count is None:
            return list(values.items())
        if candidate_count == 0:
            return []
        ranked = sorted(
            (
                (candidate_id, candidate)
                for candidate_id, candidate in values.items()
                if candidate.rank is not None and 1 <= candidate.rank <= candidate_count
            ),
            key=lambda item: item[1].rank,
        )
        if len(ranked) < candidate_count:
            ranked_ids = {candidate_id for candidate_id, _ in ranked}
            unranked = sorted(
                (
                    (candidate_id, candidate)
                    for candidate_id, candidate in values.items()
                    if candidate_id not in ranked_ids and candidate.rank is None
                ),
                key=lambda item: item[1].logprob,
                reverse=True,
            )
            ranked.extend(unranked[: candidate_count - len(ranked)])
        return ranked[:candidate_count]

    def _token_infos(
        self,
        token_ids: list[int],
        logprobs: list[Any],
        candidate_count: int | None,
    ) -> list[Any]:
        infos: list[generation_pb2.TokenInfo] = []
        for index, token_id in enumerate(token_ids):
            info = generation_pb2.TokenInfo(
                token_id=token_id,
                token=self._token_text(token_id),
            )
            if index < len(logprobs):
                value = logprobs[index]
                if isinstance(value, dict):
                    sampled = value.get(token_id)
                    if sampled is not None:
                        info.logprob = sampled.logprob
                        if sampled.rank is not None:
                            info.rank = sampled.rank
                    for candidate_id, candidate in self._candidate_items(value, candidate_count):
                        proto = info.candidates.add(
                            token_id=candidate_id,
                            logprob=candidate.logprob,
                            token=self._token_text(candidate_id),
                        )
                        if candidate.rank is not None:
                            proto.rank = candidate.rank
                elif isinstance(value, (int, float)):
                    info.logprob = float(value)
            infos.append(info)
        return infos

    def _prompt_output(self, result: object) -> generation_pb2.PromptOutput | None:
        if not result.outputs:
            return None
        prompt_logprobs = result.outputs[0].prompt_logprobs
        if not prompt_logprobs:
            return None
        prompt_token_ids = list(result.prompt_token_ids)
        if not prompt_token_ids:
            return None
        tokens = [
            generation_pb2.TokenInfo(
                token_id=prompt_token_ids[0],
                token=self._token_text(prompt_token_ids[0]),
            )
        ]
        tokens.extend(
            self._token_infos(
                prompt_token_ids[1:],
                list(prompt_logprobs)[: len(prompt_token_ids) - 1],
                getattr(result.sampling_params, "prompt_logprobs", None),
            )
        )
        return generation_pb2.PromptOutput(tokens=tokens)

    @staticmethod
    def _usage(
        result: object,
        context_usage: dict[str, Any] | None = None,
    ) -> generation_pb2.Usage:
        prompt_tokens = len(result.prompt_token_ids)
        completion_tokens = sum(len(output.token_ids or []) for output in result.outputs)
        cached_tokens = getattr(result, "cached_tokens", None)
        if context_usage is not None:
            prompt_tokens = int(context_usage["prompt_tokens"])
            details = context_usage.get("prompt_tokens_details") or {}
            cached_tokens = int(details.get("cached_tokens", 0))
        usage = generation_pb2.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        if cached_tokens is not None:
            usage.cached_prompt_tokens = cached_tokens
        return usage

    @classmethod
    def _usage_payload(cls, result: object) -> dict[str, Any]:
        usage = cls._usage(result)
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens},
        }

    @staticmethod
    def _finished(output: object, eos_token_id: int | None) -> generation_pb2.GenerationFinished:
        reason_map = {
            "stop": generation_pb2.FINISH_REASON_STOP,
            "length": generation_pb2.FINISH_REASON_LENGTH,
            "cancelled": generation_pb2.FINISH_REASON_CANCELLED,
            "timeout": generation_pb2.FINISH_REASON_CANCELLED,
        }
        finished = generation_pb2.GenerationFinished(
            output_index=output.index,
            reason=reason_map.get(output.finish_reason, generation_pb2.FINISH_REASON_STOP),
        )
        if isinstance(output.stop_reason, int):
            finished.stop_match.stop_token_id = output.stop_reason
        elif isinstance(output.stop_reason, str):
            finished.stop_match.stop_text = output.stop_reason
        elif output.finish_reason == "stop" and eos_token_id is not None:
            finished.stop_match.eos_token_id = eos_token_id
        return finished

    async def _abort_slow_consumer(
        self,
        request_id: str,
        context: grpc.aio.ServicerContext,
    ) -> None:
        result = self._requests.get(request_id)
        if result is not None and not getattr(result, "finished", False):
            try:
                result.abort()
            except (AssertionError, RuntimeError):
                logger.warning("Failed to abort stalled OpenEngine request %s", request_id)
        logger.warning(
            "Aborting OpenEngine request %s because its client did not drain a response within %.1fs",
            request_id,
            self._response_drain_timeout,
        )
        await context.abort(
            grpc.StatusCode.DEADLINE_EXCEEDED,
            "OpenEngine client did not drain the response stream in time",
        )

    async def Generate(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[generation_pb2.GenerateResponse, None]:
        """Stream responses while bounding slow-client buffering."""
        responses = self._generate_responses(request, context)
        try:
            async for response in responses:
                watchdog: asyncio.Task[None] | None = None

                def abort_slow_consumer() -> None:
                    nonlocal watchdog
                    watchdog = asyncio.create_task(
                        self._abort_slow_consumer(request.request_id, context)
                    )

                timeout = asyncio.get_running_loop().call_later(
                    self._response_drain_timeout,
                    abort_slow_consumer,
                )
                try:
                    yield response
                finally:
                    timeout.cancel()
                    if watchdog is not None:
                        await asyncio.gather(watchdog, return_exceptions=True)
        finally:
            await responses.aclose()

    async def _generate_responses(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[generation_pb2.GenerateResponse, None]:
        """Stream aggregated output or a context-first prefill handoff."""
        result = None
        request_reserved = False
        terminal_sent = False
        try:
            metadata = self._request_metadata(context)
            priority_value = self._metadata_int(
                metadata,
                "openengine-priority",
                -(1 << 31),
                (1 << 31) - 1,
            )
            target_dp_rank = self._metadata_int(
                metadata,
                "openengine-target-dp-rank",
                0,
                (1 << 32) - 1,
            )
            if self.role == server_pb2.ENGINE_ROLE_DECODE and request.kv.HasField("session"):
                session_dp_rank = request.kv.session.dp_rank
                if target_dp_rank is not None and target_dp_rank != session_dp_rank:
                    raise ValueError(
                        "openengine-target-dp-rank does not match the KV session dp_rank"
                    )
                target_dp_rank = session_dp_rank
            self._validate_generate(request, target_dp_rank)
            self._requests[request.request_id] = None
            request_reserved = True
            disaggregated = self._disaggregated_params(request, target_dp_rank)
            context_usage = (
                disaggregated.ctx_usage
                if disaggregated is not None and disaggregated.request_type == "generation_only"
                else None
            )
            input_kind = request.WhichOneof("input")
            inputs: dict[str, Any]
            if input_kind == "prompt":
                inputs = {"prompt": request.prompt}
            else:
                inputs = {"prompt_token_ids": list(request.token_ids.ids)}
            trace_headers = {
                key: value
                for key, value in metadata.items()
                if key in ("traceparent", "tracestate")
            }
            sampling_params = to_sampling_params(request)
            preprocess = getattr(self.llm, "preprocess", None)
            if callable(preprocess):
                inputs = await asyncio.get_running_loop().run_in_executor(
                    self._input_processor_executor,
                    functools.partial(
                        preprocess,
                        inputs,
                        sampling_params,
                        disaggregated,
                    ),
                )
            result = self.llm.generate_async(
                inputs=inputs,
                sampling_params=sampling_params,
                streaming=True,
                disaggregated_params=disaggregated,
                scheduling_params=self._scheduling_params(target_dp_rank),
                cache_salt=(request.kv.cache_salt if request.kv.HasField("cache_salt") else None),
                trace_headers=trace_headers or None,
                priority=to_priority(priority_value),
            )
            self._requests[request.request_id] = result
        except asyncio.CancelledError:
            if request_reserved and self._requests.get(request.request_id) is None:
                self._requests.pop(request.request_id, None)
            raise
        except (TypeError, ValueError) as error:
            if request_reserved and self._requests.get(request.request_id) is None:
                self._requests.pop(request.request_id, None)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return
        except RuntimeError as error:
            if request_reserved and self._requests.get(request.request_id) is None:
                self._requests.pop(request.request_id, None)
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return

        sent_tokens: dict[int, int] = {}
        sent_text: dict[int, int] = {}
        prompt_sent = False
        try:
            async for current in result:
                if context.cancelled():
                    return
                if current.error is not None:
                    logger.error(
                        "OpenEngine request %s failed: %s",
                        request.request_id,
                        current.error,
                    )
                    response = generation_pb2.GenerateResponse(
                        request_id=request.request_id,
                        error=error_pb2.EngineError(
                            code=error_pb2.ERROR_CODE_INTERNAL,
                            message=current.error,
                            retryable=False,
                        ),
                    )
                    response.usage.CopyFrom(self._usage(current, context_usage))
                    yield response
                    terminal_sent = True
                    return
                if not prompt_sent:
                    prompt = self._prompt_output(current)
                    if prompt is not None:
                        prompt_sent = True
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            prompt=prompt,
                        )
                if self.role == server_pb2.ENGINE_ROLE_PREFILL:
                    if current.finished:
                        handoff = current.disaggregated_params
                        if handoff is None and current.outputs:
                            handoff = current.outputs[0].disaggregated_params
                        if handoff is None:
                            raise RuntimeError(
                                "Context-only result did not return disaggregated parameters"
                            )
                        handoff = replace(
                            handoff,
                            ctx_usage=self._usage_payload(current),
                            ctx_info_endpoint=(
                                handoff.ctx_info_endpoint or self._context_info_endpoint()
                            ),
                        )
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            prefill_ready=generation_pb2.PrefillReady(
                                kv_session=encode_handoff(handoff, self.internal_disagg_auth_key)
                            ),
                            usage=self._usage(current),
                        )
                        terminal_sent = True
                    continue

                for output in current.outputs:
                    token_start = sent_tokens.get(output.index, 0)
                    text_start = sent_text.get(output.index, 0)
                    token_ids = output.token_ids or []
                    text = output.text or ""
                    delta_ids = token_ids[token_start:]
                    delta_text = text[text_start:]
                    if delta_ids or delta_text:
                        logprobs = (output.logprobs or [])[token_start:]
                        yield generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            token=generation_pb2.TokenOutput(
                                output_index=output.index,
                                tokens=self._token_infos(
                                    delta_ids,
                                    logprobs,
                                    getattr(result.sampling_params, "logprobs", None),
                                ),
                                text=delta_text,
                            ),
                        )
                    sent_tokens[output.index] = len(token_ids)
                    sent_text[output.index] = len(text)
                if current.finished:
                    if not current.outputs:
                        raise RuntimeError("TensorRT-LLM returned no generation outputs")
                    for index, output in enumerate(current.outputs):
                        response = generation_pb2.GenerateResponse(
                            request_id=request.request_id,
                            finished=self._finished(
                                output,
                                getattr(
                                    getattr(result, "sampling_params", None),
                                    "end_id",
                                    None,
                                ),
                            ),
                        )
                        if index == len(current.outputs) - 1:
                            response.usage.CopyFrom(self._usage(current, context_usage))
                        yield response
                    terminal_sent = True
            if not terminal_sent:
                raise RuntimeError("TensorRT-LLM ended the request without a terminal result")
        except asyncio.CancelledError:
            raise
        except (RuntimeError, TypeError, ValueError) as error:
            logger.error("OpenEngine request %s failed: %s", request.request_id, error)
            response = generation_pb2.GenerateResponse(
                request_id=request.request_id,
                error=error_pb2.EngineError(
                    code=error_pb2.ERROR_CODE_INTERNAL,
                    message=str(error),
                    retryable=False,
                ),
            )
            response.usage.CopyFrom(self._usage(result, context_usage))
            yield response
        finally:
            if result is not None and not getattr(result, "finished", False):
                try:
                    result.abort()
                except (AssertionError, RuntimeError):
                    logger.warning("Failed to abort OpenEngine request %s", request.request_id)
            if self._requests.get(request.request_id) is result:
                self._requests.pop(request.request_id, None)

    async def GetModelInfo(
        self,
        request: model_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> model_pb2.ModelInfo:
        """Return the canonical and served identities for the requested model."""
        if not request.model or request.model not in self._accepted_model_names:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model!r}")
        aliases = [self.model_id] if self.model_id != self.model else []
        return model_pb2.ModelInfo(
            model_id=self.model_id,
            served_model_name=self.model,
            served_model_aliases=aliases,
            supports_text_input=True,
            supports_token_ids_input=True,
        )

    async def GetServerInfo(
        self,
        request: server_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> server_pb2.ServerInfo:
        """Return the engine identity and configured serving role."""
        del request, context
        connector = kv_pb2.KvConnectorInfo()
        if self.role != server_pb2.ENGINE_ROLE_AGGREGATED:
            connector.enabled = True
            connector.transfer_backend = "tensorrt_llm"
            connector.schema_version = 1
            connector.supports_remote_prefill = True
            connector.supports_decode_pull = True
        return server_pb2.ServerInfo(
            engine_name="tensorrt_llm",
            engine_version=getattr(tensorrt_llm, "__version__", "unknown"),
            engine_role=self.role,
            instance_id=self.instance_id,
            supported_models=[self.model],
            parallelism=server_pb2.ParallelismInfo(
                tensor_parallel_size=int(_arg(self.llm, "tensor_parallel_size", 1)),
                pipeline_parallel_size=int(_arg(self.llm, "pipeline_parallel_size", 1)),
                data_parallel_size=_data_parallel_size(self.llm),
                data_parallel_rank=int(_arg(self.llm, "data_parallel_rank", 0)),
            ),
            kv_connector=connector,
            schema_revision=SCHEMA_REVISION,
            minimum_client_revision=MINIMUM_CLIENT_REVISION,
            schema_release=SCHEMA_RELEASE,
        )


__all__ = [
    "MINIMUM_CLIENT_REVISION",
    "OpenEngineServicer",
    "SCHEMA_RELEASE",
    "SCHEMA_REVISION",
]
