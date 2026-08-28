# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine gRPC server lifecycle for TensorRT-LLM."""

import asyncio
import signal
from typing import Any

import click
import grpc
import uvloop
from openengine.v1 import openengine_pb2_grpc, server_pb2

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm.logger import logger

from .servicer import OpenEngineServicer

_GRPC_MAX_MESSAGE_LENGTH_BYTES = 32 * 1024 * 1024


def _format_bind_address(host: str, port: int) -> str:
    """Format a host and port as a gRPC bind address."""
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        host = f"[{host}]"
    return f"{host}:{port}"


def openengine_role(server_role: object | None) -> int:
    """Map TensorRT-LLM serve roles to OpenEngine roles."""
    if server_role is None:
        return server_pb2.ENGINE_ROLE_AGGREGATED
    name = getattr(server_role, "name", str(server_role)).upper()
    if name == "CONTEXT":
        return server_pb2.ENGINE_ROLE_PREFILL
    if name == "GENERATION":
        return server_pb2.ENGINE_ROLE_DECODE
    raise ValueError(f"OpenEngine does not support TensorRT-LLM server role {name!r}")


def _validate_disaggregated_config(llm: object, role: int) -> None:
    if role == server_pb2.ENGINE_ROLE_AGGREGATED:
        return
    config = getattr(getattr(llm, "args", None), "cache_transceiver_config", None)
    backend = (
        config.get("backend") if isinstance(config, dict) else getattr(config, "backend", None)
    )
    if backend is None:
        raise ValueError(
            "OpenEngine prefill and decode roles require cache_transceiver_config.backend"
        )


class OpenEngineServer:
    """OpenEngine gRPC server backed by an externally owned LLM."""

    def __init__(
        self,
        llm: object,
        model: str,
        role: int,
        host: str,
        port: int,
        internal_disagg_auth_key: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.servicer = OpenEngineServicer(
            llm=llm,
            model=model,
            role=role,
            internal_disagg_auth_key=internal_disagg_auth_key,
        )
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", _GRPC_MAX_MESSAGE_LENGTH_BYTES),
                ("grpc.max_receive_message_length", _GRPC_MAX_MESSAGE_LENGTH_BYTES),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
            ]
        )
        openengine_pb2_grpc.add_InferenceServicer_to_server(self.servicer, self._server)
        openengine_pb2_grpc.add_ControlServicer_to_server(self.servicer, self._server)
        self._bind_address = _format_bind_address(host, port)
        bound_port = self._server.add_insecure_port(self._bind_address)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {self._bind_address}")
        if port == 0:
            self.port = bound_port

    async def start(self) -> None:
        """Start accepting OpenEngine requests."""
        await self._server.start()
        address = _format_bind_address(self.host, self.port)
        logger.info("OpenEngine server started on %s", address)

    async def stop(self, grace: float = 5.0) -> None:
        """Stop accepting OpenEngine requests."""
        await self._server.stop(grace=grace)
        logger.info("OpenEngine server stopped")

    async def wait_for_termination(self) -> None:
        """Wait until the OpenEngine server terminates."""
        await self._server.wait_for_termination()


def launch_server(
    host: str,
    port: int,
    llm_args: dict[str, Any],
    served_model_name: str | None = None,
    server_role: object | None = None,
    internal_disagg_auth_key: str | None = None,
) -> None:
    """Load a model and launch the dedicated OpenEngine gRPC server."""

    async def serve() -> None:
        logger.info("Initializing TensorRT-LLM OpenEngine gRPC server...")
        role = openengine_role(server_role)
        model_args = dict(llm_args)
        backend = model_args.get("backend")
        model = served_model_name or model_args.get("model", "")
        if backend == "pytorch":
            model_args.pop("build_config", None)
            llm = PyTorchLLM(**model_args)
        elif backend == "_autodeploy":
            from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

            model_args.pop("build_config", None)
            llm = AutoDeployLLM(**model_args)
        else:
            raise click.BadParameter(
                f"{backend} is not a known backend, check help for available options.",
                param_hint="backend",
            )

        server = None
        try:
            _validate_disaggregated_config(llm, role)
            server = OpenEngineServer(
                llm=llm,
                model=model,
                role=role,
                host=host,
                port=port,
                internal_disagg_auth_key=internal_disagg_auth_key,
            )
            await server.start()
            logger.info("Model loaded successfully; OpenEngine is ready to accept requests")

            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()

            def signal_handler() -> None:
                logger.info("Received shutdown signal")
                stop_event.set()

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)
            await stop_event.wait()
        finally:
            logger.info("Shutting down TensorRT-LLM OpenEngine gRPC server...")
            try:
                if server is not None:
                    await server.stop()
            finally:
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
                logger.info("LLM engine stopped")

    uvloop.run(serve())


__all__ = ["OpenEngineServer", "launch_server", "openengine_role"]
