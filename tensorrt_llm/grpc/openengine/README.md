<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT-LLM OpenEngine server

`trtllm-serve` can expose an experimental OpenEngine gRPC server instead of its normal OpenAI HTTP server. SMG remains the default gRPC protocol.

Install the optional Python bindings from the Buf Schema Registry:

```bash
python -m pip install \
  --extra-index-url https://buf.build/gen/python \
  "tensorrt_llm[openengine]"
```

Then select OpenEngine when starting an aggregated gRPC server:

```bash
trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --host 0.0.0.0 \
  --port 50051
```

Existing `--grpc` invocations continue to select SMG. OpenEngine and VisualGen cannot be enabled together.

The OpenEngine `Generate` RPC accepts text or token-ID input and streams TensorRT-LLM output. `GetServerInfo` reports the configured aggregated, prefill, or decode role, and `GetModelInfo` reports the canonical model source separately from its served name. The other control-plane RPCs remain intentionally `UNIMPLEMENTED`.

## Disaggregated serving

OpenEngine uses TensorRT-LLM's context-first KV transfer. Configure the same cache transceiver backend on both workers:

`context.yml`:

```yaml
disable_overlap_scheduler: true
cache_transceiver_config:
  backend: NIXL
```

`generation.yml`:

```yaml
cache_transceiver_config:
  backend: NIXL
```

Start one prefill worker and one decode worker on separate GPUs or hosts:

```bash
CUDA_VISIBLE_DEVICES=0 trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --server_role context \
  --config context.yml \
  --port 50051

CUDA_VISIBLE_DEVICES=1 trtllm-serve <model> \
  --grpc \
  --grpc-protocol openengine \
  --server_role generation \
  --config generation.yml \
  --port 50052
```

An external OpenEngine router sends the request to the prefill worker, receives a terminal `PrefillReady` response, and forwards its opaque `kv_session` with the original input to the decode worker. The adapter supports text-only context-first handoff. Multimodal input, LoRA selection, and guided decoding are not included in this milestone.

OpenEngine and SMG are independent protocol integrations. This integration does not make a replacement or convergence decision between them.

## Dependency provenance

The schema source is the Apache-2.0-licensed [`ai-dynamo/openengine`](https://github.com/ai-dynamo/openengine) repository at signed Git tag [`v0.1.0`](https://github.com/ai-dynamo/openengine/releases/tag/v0.1.0). That release maps to the public [`buf.build/openengine/openengine`](https://buf.build/openengine/openengine) module at immutable BSR commit `768a93c7b44e40f28c692ad0b471a8f2`.

The BSR generated the pinned wheels from that module commit:

| Package | Generator | Version | SHA-256 |
| --- | --- | --- | --- |
| `openengine-openengine-grpc-python` | [`grpc/python`](https://buf.build/grpc/python) | `1.67.1.2.20260730172104+768a93c7b44e` | `1485aed9799c4eb9367d1a261ca5cc5319f1e9b8d950ac98a26f3cb3641b8cf6` |
| `openengine-openengine-protocolbuffers-python` | [`protocolbuffers/python`](https://buf.build/protocolbuffers/python) | `31.1.0.2.20260730172104+768a93c7b44e` | `6eae12c3d8d06147fccf608da9772d6391139031fabdafdb7cf4c71a19c1f25e` |
| `openengine-openengine-protocolbuffers-pyi` | [`protocolbuffers/pyi`](https://buf.build/protocolbuffers/pyi) | `31.1.0.2.20260730172104+768a93c7b44e` | `8b0a054dbdaaa67459b3fa4786f13d8f6f4d30cf30be325f5416dbd97aba46a6` |

Buf documents the package naming and version format in its [Python-generated SDK guide](https://buf.build/docs/bsr/generated-sdks/python/). The final version segment is the BSR commit prefix. The exact requirements are pinned in `requirements-openengine.txt`.

## Maintenance boundary

The OpenEngine contributor community owns this adapter, its tests, protocol version updates, and integration bugs. TensorRT-LLM internal APIs do not provide compatibility guarantees to protocol adapters. Adapter updates must follow core runtime changes and must not block normal TensorRT-LLM development or releases.
