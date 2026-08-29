# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused protocol-boundary tests for the optional OpenEngine adapter."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import pytest

pytest.importorskip(
    "openengine.v1",
    reason='OpenEngine bindings are not installed (pip install "tensorrt_llm[openengine]")',
)

from openengine.v1 import generation_pb2, model_pb2, server_pb2  # noqa: E402

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle  # noqa: E402
from tensorrt_llm.executor.result import Logprob  # noqa: E402
from tensorrt_llm.grpc.openengine.converters import (  # noqa: E402
    HANDOFF_ATTRIBUTE,
    HANDOFF_AUTH_ATTRIBUTE,
    decode_handoff,
    encode_handoff,
    to_sampling_params,
)
from tensorrt_llm.grpc.openengine.server import _validate_launch_config  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineServicer  # noqa: E402
from tensorrt_llm.llmapi.llm import LLM  # noqa: E402
from tensorrt_llm.sampling_params import SamplingParams  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _Context:
    def invocation_metadata(self):
        return ()

    def cancelled(self):
        return False

    async def abort(self, code, details):
        pytest.fail(f"Unexpected gRPC abort: {code}: {details}")


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(f"<{token_id}>" for token_id in token_ids)


class _Result:
    def __init__(self, disaggregated_params=None):
        self.outputs = [
            SimpleNamespace(
                index=0,
                text="hello",
                token_ids=[7, 8],
                logprobs=[],
                prompt_logprobs=[],
                finish_reason="stop",
                stop_reason=None,
                disaggregated_params=disaggregated_params,
            )
        ]
        self.prompt_token_ids = [1, 2]
        self.cached_tokens = 0
        self.finished = True
        self.error = None
        self.disaggregated_params = disaggregated_params
        self.sampling_params = SimpleNamespace(
            end_id=2,
            logprobs=None,
            prompt_logprobs=None,
        )

    def __aiter__(self):
        async def items():
            yield self

        return items()

    def abort(self):
        pytest.fail("A completed result must not be aborted")


class _Llm:
    tokenizer = _Tokenizer()
    disaggregated_params = {"ctx_info_endpoint": ["ctx:1234"]}

    def __init__(self, result_factory):
        self.args = SimpleNamespace(
            model="model",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
            enable_attention_dp=False,
            backend="pytorch",
            disable_overlap_scheduler=False,
            max_batch_size=1,
        )
        self._result_factory = result_factory
        self.calls = []

    def generate_async(self, **kwargs):
        self.calls.append(kwargs)
        result = self._result_factory()
        sampling_params = kwargs["sampling_params"]
        if sampling_params.end_id is None:
            sampling_params.end_id = 2
        result.sampling_params = sampling_params
        return result


async def _collect(servicer, request):
    return [response async for response in servicer.Generate(request, _Context())]


@pytest.mark.asyncio
async def test_model_info_distinguishes_canonical_and_served_names():
    """Regression: discovery must not use a served alias as the tokenizer source."""
    llm = _Llm(_Result)
    llm.args.model = "org/model"
    servicer = OpenEngineServicer(llm, "served-alias", server_pb2.ENGINE_ROLE_AGGREGATED)

    info = await servicer.GetModelInfo(
        model_pb2.GetModelInfoRequest(model="served-alias"),
        _Context(),
    )

    assert info.model_id == "org/model"
    assert info.served_model_name == "served-alias"
    assert list(info.served_model_aliases) == ["org/model"]


@pytest.mark.asyncio
async def test_aggregated_generate_streams_engine_output():
    """Regression: aggregated OpenEngine requests must reach the LLM and terminate."""
    llm = _Llm(_Result)
    servicer = OpenEngineServicer(llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED)

    responses = await _collect(
        servicer,
        generation_pb2.GenerateRequest(request_id="aggregate", prompt="Hi"),
    )

    assert [response.WhichOneof("event") for response in responses] == [
        "token",
        "finished",
    ]
    assert [token.token_id for token in responses[0].token.tokens] == [7, 8]
    assert responses[-1].finished.stop_match.eos_token_id == 2
    assert responses[-1].usage.prompt_tokens == 2
    assert llm.calls[0]["disaggregated_params"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [
        server_pb2.ENGINE_ROLE_AGGREGATED,
        server_pb2.ENGINE_ROLE_PREFILL,
    ],
)
async def test_engine_failure_is_the_only_terminal_response(role):
    """Regression: accepted engine failures must not become success responses."""
    result = _Result()
    result.error = "engine rejected request"
    servicer = OpenEngineServicer(
        _Llm(lambda: result),
        "model",
        role,
        internal_disagg_auth_key=("secret" if role == server_pb2.ENGINE_ROLE_PREFILL else None),
    )

    responses = await _collect(
        servicer,
        generation_pb2.GenerateRequest(request_id="engine-error", prompt="Hi"),
    )

    assert [response.WhichOneof("event") for response in responses] == ["error"]
    assert responses[0].error.message == "engine rejected request"
    assert responses[0].usage.prompt_tokens == 2


@pytest.mark.asyncio
async def test_preprocessing_runs_off_the_grpc_event_loop():
    """Regression: prompt preprocessing must not block concurrent gRPC streams."""

    class _PreprocessingLlm(_Llm):
        def preprocess(self, inputs, sampling_params, disaggregated_params):
            self.preprocess_thread = threading.get_ident()
            self.preprocess_args = (inputs, sampling_params, disaggregated_params)
            return SimpleNamespace(prompt_token_ids=[1, 2])

        def generate_async(self, **kwargs):
            self.generate_thread = threading.get_ident()
            return super().generate_async(**kwargs)

    llm = _PreprocessingLlm(_Result)
    event_loop_thread = threading.get_ident()
    with ThreadPoolExecutor(max_workers=1) as executor:
        servicer = OpenEngineServicer(
            llm,
            "model",
            server_pb2.ENGINE_ROLE_AGGREGATED,
            input_processor_executor=executor,
        )
        await _collect(
            servicer,
            generation_pb2.GenerateRequest(request_id="preprocess", prompt="Hi"),
        )

    assert llm.preprocess_thread != event_loop_thread
    assert llm.generate_thread == event_loop_thread
    assert llm.calls[0]["inputs"].prompt_token_ids == [1, 2]


@pytest.mark.asyncio
async def test_prompt_logprobs_align_with_scored_prompt_tokens():
    """Regression: TRT prompt scores are offset and exclude the first prompt token."""
    result = _Result()
    result.prompt_token_ids = [10, 11]
    result.outputs[0].prompt_logprobs = [-0.25, -0.5]
    servicer = OpenEngineServicer(
        _Llm(lambda: result),
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
    )

    responses = await _collect(
        servicer,
        generation_pb2.GenerateRequest(request_id="prompt-logprobs", prompt="Hi"),
    )

    prompt_tokens = responses[0].prompt.tokens
    assert [token.token_id for token in prompt_tokens] == [10, 11]
    assert not prompt_tokens[0].HasField("logprob")
    assert prompt_tokens[1].logprob == pytest.approx(-0.25)


@pytest.mark.asyncio
async def test_output_candidates_exclude_sampled_token_outside_requested_top_n():
    """Regression: sampled-token metadata must not expand the requested candidate set."""
    result = _Result()
    result.outputs[0].token_ids = [7]
    result.outputs[0].logprobs = [
        {
            7: Logprob(logprob=-3.0, rank=3),
            8: Logprob(logprob=-0.1, rank=1),
            9: Logprob(logprob=-0.2, rank=2),
        }
    ]
    servicer = OpenEngineServicer(
        _Llm(lambda: result),
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
    )
    request = generation_pb2.GenerateRequest(request_id="top-n", prompt="Hi")
    request.response.return_output_logprobs = True
    request.response.output_candidates.top_n = 1

    responses = await _collect(servicer, request)

    token = responses[0].token.tokens[0]
    assert token.token_id == 7
    assert token.rank == 3
    assert token.logprob == pytest.approx(-3.0)
    assert [(candidate.token_id, candidate.rank) for candidate in token.candidates] == [(8, 1)]


def test_explicit_zero_num_sequences_is_rejected():
    """Regression: OpenEngine requires an explicitly provided sequence count to be positive."""
    request = generation_pb2.GenerateRequest(request_id="zero-sequences", prompt="Hi")
    request.sampling.num_sequences = 0

    with pytest.raises(ValueError, match="num_sequences must be greater than zero"):
        to_sampling_params(request)


@pytest.mark.asyncio
async def test_prefill_handoff_round_trips_to_decode_params():
    """Regression: a prefill session must reconstruct the native decode handoff."""
    handoff = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[9],
        ctx_request_id=77,
        opaque_state=b"opaque",
        disagg_request_id=88,
        ctx_dp_rank=0,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )
    prefill = OpenEngineServicer(
        _Llm(lambda: _Result(handoff)),
        "model",
        server_pb2.ENGINE_ROLE_PREFILL,
        internal_disagg_auth_key="secret",
    )

    prefill_responses = await _collect(
        prefill,
        generation_pb2.GenerateRequest(request_id="prefill", prompt="Hi"),
    )
    session = prefill_responses[0].prefill_ready.kv_session

    decode_llm = _Llm(_Result)
    decode = OpenEngineServicer(
        decode_llm,
        "model",
        server_pb2.ENGINE_ROLE_DECODE,
        internal_disagg_auth_key="secret",
    )
    request = generation_pb2.GenerateRequest(request_id="decode", prompt="Hi")
    request.kv.session.CopyFrom(session)
    decode_responses = await _collect(decode, request)
    params = decode_llm.calls[0]["disaggregated_params"]

    assert [response.WhichOneof("event") for response in prefill_responses] == ["prefill_ready"]
    assert session.session_id == "88"
    assert session.transfer_backend == "tensorrt_llm"
    assert params.request_type == "generation_only"
    assert params.first_gen_tokens == [9]
    assert params.ctx_request_id == 77
    assert params.disagg_request_id == 88
    assert params.opaque_state == b"opaque"
    assert decode_responses[-1].WhichOneof("event") == "finished"


@pytest.mark.asyncio
async def test_prefill_retry_mints_a_fresh_transfer_id():
    """Regression: a retried wire request must not alias a live KV transfer."""

    class _PrefillLlm(_Llm):
        def generate_async(self, **kwargs):
            self._result_factory = lambda: _Result(kwargs["disaggregated_params"])
            return super().generate_async(**kwargs)

    servicer = OpenEngineServicer(
        _PrefillLlm(_Result),
        "model",
        server_pb2.ENGINE_ROLE_PREFILL,
        internal_disagg_auth_key="secret",
    )
    request = generation_pb2.GenerateRequest(request_id="retry", prompt="Hi")

    first = await _collect(servicer, request)
    second = await _collect(servicer, request)

    assert first[0].prefill_ready.kv_session.session_id
    assert first[0].prefill_ready.kv_session.session_id != (
        second[0].prefill_ready.kv_session.session_id
    )


@pytest.mark.parametrize(
    "tamper",
    ["session_id", "dp_rank", "token_state", "strip_fields", "remove_auth"],
)
def test_authenticated_handoff_rejects_tampering(tamper):
    """Regression: every routing and transfer field must be integrity protected."""
    session = encode_handoff(
        DisaggregatedParams(
            request_type="context_only",
            first_gen_tokens=[9],
            ctx_request_id=77,
            opaque_state=b"opaque",
            disagg_request_id=88,
            ctx_dp_rank=0,
            ctx_info_endpoint="ctx:1234",
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        ),
        internal_disagg_auth_key="secret",
    )
    fields = session.attributes_struct.fields
    handoff_fields = fields[HANDOFF_ATTRIBUTE].struct_value.fields
    if tamper == "session_id":
        session.session_id = "99"
    elif tamper == "dp_rank":
        session.dp_rank = 1
    elif tamper == "token_state":
        handoff_fields["first_gen_tokens"].list_value.values[0].number_value = 10
    elif tamper == "strip_fields":
        del handoff_fields["opaque_state"]
        del handoff_fields["ctx_info_endpoint"]
    else:
        del fields[HANDOFF_AUTH_ATTRIBUTE]

    with pytest.raises(ValueError, match="Invalid internal"):
        decode_handoff(session, internal_disagg_auth_key="secret")


def test_tokenizerless_launch_is_rejected_at_real_preparation_boundary():
    """Regression: advertised text inputs must be executable by the real LLM API."""
    real_preparation = SimpleNamespace(
        args=SimpleNamespace(backend="tensorrt"),
        tokenizer=None,
        _apply_generation_config_sampling_defaults=lambda sampling_params: None,
    )
    with pytest.raises(ValueError, match="tokenizer is required"):
        LLM._prepare_sampling_params(real_preparation, SamplingParams())
    with pytest.raises(ValueError, match="skip_tokenizer_init"):
        _validate_launch_config(
            {"backend": "pytorch", "skip_tokenizer_init": True},
            server_pb2.ENGINE_ROLE_AGGREGATED,
            None,
        )


def test_autodeploy_prefill_requires_overlap_disabled_at_startup():
    """Regression: reject AutoDeploy prefill configurations that fail every request."""
    with pytest.raises(ValueError, match="disable_overlap_scheduler"):
        _validate_launch_config(
            {"backend": "_autodeploy", "disable_overlap_scheduler": False},
            server_pb2.ENGINE_ROLE_PREFILL,
            "secret",
        )


@pytest.mark.asyncio
async def test_slow_consumer_aborts_the_engine_request():
    """Regression: a stalled reader must not leave an unbounded engine stream running."""

    class _StreamingResult(_Result):
        def __init__(self):
            super().__init__()
            self.finished = False
            self.aborted = False

        def __aiter__(self):
            async def items():
                yield self
                await asyncio.Event().wait()

            return items()

        def abort(self):
            self.aborted = True

    class _AbortContext(_Context):
        def __init__(self):
            self.aborted = None

        async def abort(self, code, details):
            self.aborted = (code, details)

    result = _StreamingResult()
    context = _AbortContext()
    servicer = OpenEngineServicer(
        _Llm(lambda: result),
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        response_drain_timeout=0.01,
    )
    stream = servicer.Generate(
        generation_pb2.GenerateRequest(request_id="slow-reader", prompt="Hi"),
        context,
    )

    response = await stream.__anext__()
    assert response.WhichOneof("event") == "token"
    await asyncio.sleep(0.05)

    assert result.aborted
    assert context.aborted is not None
    assert context.aborted[0] == grpc.StatusCode.DEADLINE_EXCEEDED
    await stream.aclose()
