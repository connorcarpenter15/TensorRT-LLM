# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversions between OpenEngine messages and TensorRT-LLM API objects."""

import base64
import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict
from openengine.v1 import generation_pb2, kv_pb2

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.sampling_params import SamplingParams

HANDOFF_ATTRIBUTE = "tensorrt_llm.disaggregated_params.v1"


def _optional(message: object, field: str) -> Any | None:
    return getattr(message, field) if message.HasField(field) else None


def _candidate_count(selection: object, enabled: bool) -> int | None:
    if not enabled:
        return None
    kind = selection.WhichOneof("selection")
    if kind in (None, "top_n"):
        return selection.top_n if kind == "top_n" else 0
    raise ValueError("TensorRT-LLM OpenEngine supports top_n logprob selection only")


def to_sampling_params(request: generation_pb2.GenerateRequest) -> SamplingParams:
    """Build TensorRT-LLM sampling parameters from an OpenEngine request."""
    sampling = request.sampling
    stopping = request.stopping
    response = request.response
    kwargs: dict[str, Any] = {
        "max_tokens": 32 if _optional(stopping, "max_tokens") is None else stopping.max_tokens,
        "detokenize": True,
    }
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    ):
        value = _optional(sampling, field)
        if value is not None:
            kwargs[field] = value

    num_sequences = _optional(sampling, "num_sequences")
    if num_sequences is not None:
        kwargs["n"] = num_sequences
        kwargs["best_of"] = num_sequences
    min_tokens = _optional(stopping, "min_tokens")
    if min_tokens is not None:
        kwargs["min_tokens"] = min_tokens
    ignore_eos = _optional(stopping, "ignore_eos")
    if ignore_eos is not None:
        kwargs["ignore_eos"] = ignore_eos
    include_stop = _optional(stopping, "include_stop_in_output")
    if include_stop is not None:
        kwargs["include_stop_str_in_output"] = include_stop

    stop_text: list[str] = []
    stop_token_ids: list[int] = []
    for condition in stopping.conditions:
        kind = condition.WhichOneof("condition")
        if kind == "stop_text":
            stop_text.append(condition.stop_text)
        elif kind == "stop_token_id":
            stop_token_ids.append(condition.stop_token_id)
    if stop_text:
        kwargs["stop"] = stop_text
    if stop_token_ids:
        kwargs["stop_token_ids"] = stop_token_ids

    prompt_logprobs = _candidate_count(
        response.prompt_candidates,
        bool(_optional(response, "return_prompt_logprobs")),
    )
    output_logprobs = _candidate_count(
        response.output_candidates,
        bool(_optional(response, "return_output_logprobs")),
    )
    if prompt_logprobs is not None:
        kwargs["prompt_logprobs"] = prompt_logprobs
    if output_logprobs is not None:
        kwargs["logprobs"] = output_logprobs
    prompt_start = _optional(response, "prompt_logprob_start")
    if prompt_start not in (None, 0):
        raise ValueError("TensorRT-LLM does not support prompt_logprob_start through OpenEngine")

    return SamplingParams(**kwargs)


def to_priority(priority: int | None) -> float:
    """Map signed OpenEngine priority into TensorRT-LLM's bounded domain."""
    if priority is None:
        return 0.5
    return 0.5 + 0.5 * priority / (1 + abs(priority))


def stable_request_id(request_id: str) -> int:
    """Map arbitrary wire request IDs into TensorRT-LLM's positive int64 domain."""
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _struct_value(value: Any) -> Any:
    if is_dataclass(value):
        return _struct_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _struct_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_struct_value(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Logprob):
        return {"logprob": float(value.logprob), "rank": value.rank}
    if hasattr(value, "item"):
        return value.item()
    return value


def encode_handoff(params: DisaggregatedParams) -> kv_pb2.KvSessionRef:
    """Encode a context-first TensorRT-LLM KV handoff."""
    unsupported = {
        "first-generation logits": params.first_gen_logits,
        "multimodal embedding handles": params.multimodal_embedding_handles,
        "multimodal hashes": params.multimodal_hashes,
        "mRoPE position IDs": params.mrope_position_ids_handle,
        "mRoPE position deltas": params.mrope_position_deltas_handle,
    }
    present = [name for name, value in unsupported.items() if value is not None]
    if present:
        raise ValueError("OpenEngine text handoff does not support " + ", ".join(present))
    if params.schedule_style not in (None, DisaggScheduleStyle.CONTEXT_FIRST):
        raise ValueError("OpenEngine supports context-first handoff only")

    session_id = params.disagg_request_id or params.ctx_request_id
    if session_id is None:
        raise ValueError("Context-only result did not provide a request ID")
    payload = {
        "first_gen_tokens": _struct_value(params.first_gen_tokens),
        "first_gen_log_probs": _struct_value(params.first_gen_log_probs),
        "ctx_request_id": None if params.ctx_request_id is None else str(params.ctx_request_id),
        "disagg_request_id": (
            None if params.disagg_request_id is None else str(params.disagg_request_id)
        ),
        "ctx_dp_rank": params.ctx_dp_rank,
        "ctx_info_endpoint": params.ctx_info_endpoint,
        "draft_tokens": _struct_value(params.draft_tokens),
        "ctx_usage": _struct_value(params.ctx_usage),
        "schedule_style": "context_first",
        "opaque_state": (
            None
            if params.opaque_state is None
            else base64.b64encode(params.opaque_state).decode("ascii")
        ),
    }
    session = kv_pb2.KvSessionRef(
        session_id=str(session_id),
        transfer_backend="tensorrt_llm",
        dp_rank=params.ctx_dp_rank or 0,
    )
    session.attributes_struct[HANDOFF_ATTRIBUTE] = payload
    return session


def _handoff_payload(session: kv_pb2.KvSessionRef) -> dict[str, Any]:
    value = session.attributes_struct.fields.get(HANDOFF_ATTRIBUTE)
    if value is None or value.WhichOneof("kind") != "struct_value":
        raise ValueError(f"KV session is missing {HANDOFF_ATTRIBUTE!r}")
    payload = MessageToDict(value.struct_value, preserving_proto_field_name=True)
    if not isinstance(payload, dict):
        raise ValueError("TensorRT-LLM handoff must contain an object")
    return payload


def _decimal_id(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"{name} must be a decimal string")
    return int(value)


def _integer_list(payload: dict[str, Any], name: str) -> list[int] | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    output: list[int] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or int(item) != item:
            raise ValueError(f"{name} must contain integers")
        output.append(int(item))
    return output


def _logprobs(payload: dict[str, Any]) -> list[Any] | None:
    value = payload.get("first_gen_log_probs")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("first_gen_log_probs must be a list")
    output: list[Any] = []
    for position in value:
        if isinstance(position, (int, float)) and not isinstance(position, bool):
            output.append(float(position))
            continue
        if not isinstance(position, dict):
            raise ValueError("first_gen_log_probs entries must be objects or numbers")
        candidates: dict[int, Logprob] = {}
        for token_id, candidate in position.items():
            if not str(token_id).isdecimal() or not isinstance(candidate, dict):
                raise ValueError("first_gen_log_probs contains an invalid candidate")
            if "logprob" not in candidate:
                raise ValueError("first_gen_log_probs candidate is missing logprob")
            rank = candidate.get("rank")
            candidates[int(token_id)] = Logprob(
                logprob=float(candidate["logprob"]),
                rank=None if rank is None else int(rank),
            )
        output.append(candidates)
    return output


def _ctx_usage(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("ctx_usage")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("ctx_usage must be an object")
    output = dict(value)
    if "prompt_tokens" not in output:
        raise ValueError("ctx_usage.prompt_tokens is required")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if name not in output:
            continue
        item = output[name]
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or int(item) != item
            or item < 0
        ):
            raise ValueError(f"ctx_usage.{name} must be a non-negative integer")
        output[name] = int(item)
    details = output.get("prompt_tokens_details")
    if details is not None and not isinstance(details, dict):
        raise ValueError("ctx_usage.prompt_tokens_details must be an object")
    if isinstance(details, dict) and "cached_tokens" in details:
        cached_tokens = details["cached_tokens"]
        if (
            not isinstance(cached_tokens, (int, float))
            or isinstance(cached_tokens, bool)
            or int(cached_tokens) != cached_tokens
            or cached_tokens < 0
        ):
            raise ValueError(
                "ctx_usage.prompt_tokens_details.cached_tokens must be a non-negative integer"
            )
        output["prompt_tokens_details"] = {
            **details,
            "cached_tokens": int(cached_tokens),
        }
    return output


def decode_handoff(session: kv_pb2.KvSessionRef) -> DisaggregatedParams:
    """Decode and validate a context-first TensorRT-LLM KV handoff."""
    payload = _handoff_payload(session)
    if payload.get("schedule_style") != "context_first":
        raise ValueError("OpenEngine supports context-first handoff only")
    opaque = payload.get("opaque_state")
    try:
        opaque_state = None if opaque is None else base64.b64decode(opaque, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("opaque_state must be canonical base64") from error

    ctx_dp_rank = payload.get("ctx_dp_rank")
    if ctx_dp_rank is None:
        ctx_dp_rank = session.dp_rank
    if (
        not isinstance(ctx_dp_rank, (int, float))
        or isinstance(ctx_dp_rank, bool)
        or int(ctx_dp_rank) != ctx_dp_rank
        or ctx_dp_rank < 0
    ):
        raise ValueError("ctx_dp_rank must be a non-negative integer")

    ctx_info_endpoint = payload.get("ctx_info_endpoint")
    if ctx_info_endpoint is not None and not isinstance(ctx_info_endpoint, str):
        raise ValueError("ctx_info_endpoint must be a string")

    return DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=_integer_list(payload, "first_gen_tokens"),
        first_gen_log_probs=_logprobs(payload),
        ctx_request_id=_decimal_id(payload, "ctx_request_id"),
        opaque_state=opaque_state,
        draft_tokens=_integer_list(payload, "draft_tokens"),
        disagg_request_id=_decimal_id(payload, "disagg_request_id"),
        ctx_dp_rank=int(ctx_dp_rank),
        ctx_info_endpoint=ctx_info_endpoint,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        ctx_usage=_ctx_usage(payload),
    )


__all__ = [
    "HANDOFF_ATTRIBUTE",
    "decode_handoff",
    "encode_handoff",
    "stable_request_id",
    "to_priority",
    "to_sampling_params",
]
