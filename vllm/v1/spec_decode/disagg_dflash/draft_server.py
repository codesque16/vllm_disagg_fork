# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI / library entry for the Disagg-DFlash draft server."""

from __future__ import annotations

import argparse

from msgspec import msgpack

from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.connector import DisaggDFlashServerSocket
from vllm.v1.spec_decode.disagg_dflash.draft_engine import DisaggDFlashDraftEngine
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    PAYLOAD_IPC,
    DisaggDFlashSpeculateRequest,
)

logger = init_logger(__name__)


def run_draft_server(
    *,
    draft_model: str,
    target_model: str,
    num_speculative_tokens: int,
    bind: str = "tcp://0.0.0.0:50051",
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    gpu_memory_utilization: float = 0.8,
    num_gpu_blocks: int | None = None,
    attention_backend: str | None = "FLASH_ATTN",
    transport: str = "zmq",
    ipc_max_num_tokens: int = 16384,
) -> None:
    from vllm.config import set_current_vllm_config

    engine = DisaggDFlashDraftEngine(
        draft_model=draft_model,
        target_model=target_model,
        num_speculative_tokens=num_speculative_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        num_gpu_blocks=num_gpu_blocks,
        attention_backend=attention_backend,
        ipc_max_num_tokens=ipc_max_num_tokens,
    )
    server = DisaggDFlashServerSocket(bind)
    transport = (transport or "zmq").lower()

    if transport == "cuda_ipc":
        # Eagerly allocate / export IPC handles before clients connect.
        engine.ensure_ipc_staging()
    elif transport in ("nccl", "nixl"):
        raise NotImplementedError(
            f"--transport {transport} is reserved but not implemented. "
            "Use zmq or cuda_ipc."
        )
    elif transport != "zmq":
        raise ValueError(f"Unknown --transport {transport!r}")

    def speculate_decoder(frames: list[bytes]) -> DisaggDFlashSpeculateRequest:
        meta = msgpack.decode(frames[0])
        if meta.get("payload_mode") == PAYLOAD_IPC:
            staging = engine.ensure_ipc_staging()
            # Pull peer-written IPC cudaMalloc buffer into torch staging first.
            n_ctx = int(meta["num_ctx_tokens"])
            hiddens = staging.pull_hiddens_from_ipc(n_ctx)
            return DisaggDFlashSpeculateRequest.decode(
                frames, context_hiddens=hiddens
            )
        return DisaggDFlashSpeculateRequest.decode(frames)

    def handler(cmd, payload):
        return engine.handle(cmd, payload)

    logger.info(
        "Disagg-DFlash draft server serving on %s (transport=%s)", bind, transport
    )
    try:
        # Keep VllmConfig active for attention / CustomOp during serve.
        with set_current_vllm_config(engine.vllm_config):
            engine.warmup()
            server.serve(handler, speculate_decoder=speculate_decoder)
    finally:
        server.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Disagg-DFlash draft server")
    parser.add_argument("--draft-model", required=True, help="DFlash draft model path")
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model path (for embed_tokens / lm_head weights)",
    )
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--bind", default="tcp://0.0.0.0:50051")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=None,
        help=(
            "Draft KV block pool size (token pages). Default: auto from "
            "max_num_seqs/max_model_len. 2048 is too small for large "
            "max_num_seqs (exhausts under load)."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        default="FLASH_ATTN",
        help="Attention backend for non-causal DFlash (e.g. FLASH_ATTN)",
    )
    parser.add_argument(
        "--transport",
        default="zmq",
        choices=["zmq", "cuda_ipc", "nccl", "nixl"],
        help=(
            "Data-plane transport. zmq=tensors over ZMQ (default); "
            "cuda_ipc=same-node GPU staging via CUDA IPC (control stays on ZMQ). "
            "nccl/nixl are reserved stubs."
        ),
    )
    parser.add_argument(
        "--ipc-max-num-tokens",
        type=int,
        default=16384,
        help="CUDA IPC hidden staging capacity in tokens (cuda_ipc only).",
    )
    args = parser.parse_args(argv)
    run_draft_server(
        draft_model=args.draft_model,
        target_model=args.target_model,
        num_speculative_tokens=args.num_speculative_tokens,
        bind=args.bind,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks=args.num_gpu_blocks,
        attention_backend=args.attention_backend,
        transport=args.transport,
        ipc_max_num_tokens=args.ipc_max_num_tokens,
    )


if __name__ == "__main__":
    main()
