# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused protocol-boundary tests for the optional OpenEngine adapter."""

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "openengine.v1",
    reason='OpenEngine bindings are not installed (pip install "tensorrt_llm[openengine]")',
)

from openengine.v1 import generation_pb2, model_pb2, server_pb2  # noqa: E402

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineServicer  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _Context:
    def invocation_metadata(self):
        return ()

    def cancelled(self):
        return False

    async def abort(self, code, details):
        pytest.fail(f"Unexpected gRPC abort: {code}: {details}")


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
        self.disaggregated_params = disaggregated_params

    def __aiter__(self):
        async def items():
            yield self

        return items()

    def abort(self):
        pytest.fail("A completed result must not be aborted")


class _Llm:
    tokenizer = None
    disaggregated_params = {"ctx_info_endpoint": ["ctx:1234"]}

    def __init__(self, result_factory):
        self.args = SimpleNamespace(
            model="model",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_rank=0,
            enable_attention_dp=False,
        )
        self._result_factory = result_factory
        self.calls = []

    def generate_async(self, **kwargs):
        self.calls.append(kwargs)
        return self._result_factory()


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
    assert responses[-1].usage.prompt_tokens == 2
    assert llm.calls[0]["disaggregated_params"] is None


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
    )

    prefill_responses = await _collect(
        prefill,
        generation_pb2.GenerateRequest(request_id="prefill", prompt="Hi"),
    )
    session = prefill_responses[0].prefill_ready.kv_session

    decode_llm = _Llm(_Result)
    decode = OpenEngineServicer(decode_llm, "model", server_pb2.ENGINE_ROLE_DECODE)
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
