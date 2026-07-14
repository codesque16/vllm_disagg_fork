# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA IPC data-plane transport for same-node Disagg-DFlash.

Control / metadata stay on ZMQ. Large tensors (context_hiddens, draft_tokens)
move via draft-owned ``cudaMalloc`` staging exported with ``cudaIpcGetMemHandle``.

Staging is allocated with the CUDA runtime allocator (not the PyTorch caching
allocator) so IPC handles refer to the exact base pointer. Draft copies
into/out of ordinary torch tensors around each speculate.
"""

from __future__ import annotations

import base64
import ctypes
from typing import Any

import torch

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm.logger import init_logger
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    PAYLOAD_IPC,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    decode_hello_reply,
    encode_hello,
)
from vllm.v1.spec_decode.disagg_dflash.transport import (
    TRANSPORT_CUDA_IPC,
    DisaggDFlashClientTransport,
)
from vllm.v1.spec_decode.disagg_dflash.transport_zmq import ZmqClientTransport

logger = init_logger(__name__)

_ALIGN = 1 << 21  # 2 MiB — CUDA IPC allocation alignment


def _aligned_nbytes(nbytes: int) -> int:
    return ((nbytes + _ALIGN - 1) // _ALIGN) * _ALIGN


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def export_ipc_handle_ptr(ptr: int | ctypes.c_void_p) -> str:
    """Export ``cudaIpcGetMemHandle`` for a cudaMalloc pointer (base64)."""
    lib = CudaRTLibrary()
    c_ptr = ptr if isinstance(ptr, ctypes.c_void_p) else ctypes.c_void_p(int(ptr))
    handle = lib.cudaIpcGetMemHandle(c_ptr)
    return base64.b64encode(bytes(handle.internal)).decode("ascii")


def open_ipc_ptr(handle_b64: str) -> int:
    """Open a peer CUDA IPC allocation; return device pointer."""
    lib = CudaRTLibrary()
    try:
        handle = cudaIpcMemHandle_t()
        raw = base64.b64decode(handle_b64)
        # cudaIpcMemHandle_t is opaque; libcudart historically uses 64 bytes,
        # while vLLM's ctypes wrapper allocates a 128-byte ``internal`` buffer.
        # Accept either and zero-pad into the structure.
        if len(raw) not in (64, 128):
            raise ValueError(
                f"IPC handle must be 64 or 128 bytes, got {len(raw)}"
            )
        capacity = len(handle.internal)
        for i in range(capacity):
            handle.internal[i] = raw[i] if i < len(raw) else 0
        ptr = lib.cudaIpcOpenMemHandle(handle)
        if not ptr or not ptr.value:
            raise RuntimeError("cudaIpcOpenMemHandle returned NULL")
        return int(ptr.value)
    except Exception as e:
        raise RuntimeError(
            "Failed to open Disagg-DFlash CUDA IPC handle. "
            "cuda_ipc requires draft and verify on the same host with "
            "P2P/NVLink (or PCIe P2P) between TP0's GPU and the draft GPU. "
            f"Underlying error: {e}"
        ) from e


def close_ipc_ptr(ptr: int) -> None:
    if not ptr:
        return
    try:
        lib = CudaRTLibrary()
        # cudaIpcCloseMemHandle is not wrapped; call via ctypes if present.
        if "cudaIpcCloseMemHandle" in getattr(lib, "funcs", {}):
            lib.CUDART_CHECK(
                lib.funcs["cudaIpcCloseMemHandle"](ctypes.c_void_p(ptr))
            )
        else:
            # Best-effort: look up dynamically.
            f = getattr(lib.lib, "cudaIpcCloseMemHandle", None)
            if f is not None:
                f.restype = ctypes.c_int
                f.argtypes = [ctypes.c_void_p]
                err = f(ctypes.c_void_p(ptr))
                if err != 0:
                    raise RuntimeError(f"cudaIpcCloseMemHandle err={err}")
    except Exception as e:
        logger.warning("cudaIpcCloseMemHandle failed: %s", e)


def memcpy_d2d(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
    if nbytes <= 0:
        return
    lib = CudaRTLibrary()
    lib.cudaMemcpy(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr), nbytes)
    lib.cudaDeviceSynchronize()


class CudaIpcStaging:
    """Draft-owned cudaMalloc IPC buffers + torch mirrors for compute."""

    def __init__(
        self,
        *,
        max_tokens: int,
        max_num_seqs: int,
        hidden_size: int,
        num_speculative_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if max_tokens < 1 or max_num_seqs < 1 or hidden_size < 1:
            raise ValueError("Invalid CUDA IPC staging sizes")
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self.hidden_size = hidden_size
        self.num_speculative_tokens = num_speculative_tokens
        self.dtype = dtype
        self.device = device
        self._lib = CudaRTLibrary()

        # Torch mirrors (caching allocator) — draft model reads/writes these.
        self.hidden_staging = torch.zeros(
            max_tokens, hidden_size, dtype=dtype, device=device
        )
        self.token_staging = torch.zeros(
            max_num_seqs,
            num_speculative_tokens,
            dtype=torch.int64,
            device=device,
        )

        # IPC-exported buffers (cudaMalloc base == handle base).
        self._hidden_nbytes = int(
            max_tokens * hidden_size * _dtype_nbytes(dtype)
        )
        self._token_nbytes = int(
            max_num_seqs * num_speculative_tokens * _dtype_nbytes(torch.int64)
        )
        self._hidden_ipc_ptr = self._lib.cudaMalloc(
            _aligned_nbytes(self._hidden_nbytes)
        )
        self._token_ipc_ptr = self._lib.cudaMalloc(
            _aligned_nbytes(self._token_nbytes)
        )
        self._lib.cudaMemset(self._hidden_ipc_ptr, 0, self._hidden_nbytes)
        self._lib.cudaMemset(self._token_ipc_ptr, 0, self._token_nbytes)
        self._hidden_handle = export_ipc_handle_ptr(self._hidden_ipc_ptr)
        self._token_handle = export_ipc_handle_ptr(self._token_ipc_ptr)

    @property
    def hidden_ipc_ptr(self) -> int:
        return int(self._hidden_ipc_ptr.value)  # type: ignore[arg-type]

    @property
    def token_ipc_ptr(self) -> int:
        return int(self._token_ipc_ptr.value)  # type: ignore[arg-type]

    def pull_hiddens_from_ipc(self, num_ctx_tokens: int) -> torch.Tensor:
        """IPC buffer → torch hidden_staging; return ``[:num_ctx]`` view."""
        n = int(num_ctx_tokens)
        if n < 0 or n > self.max_tokens:
            raise ValueError(f"num_ctx_tokens={n} out of range")
        nbytes = n * self.hidden_size * _dtype_nbytes(self.dtype)
        if n > 0:
            memcpy_d2d(
                int(self.hidden_staging.data_ptr()),
                self.hidden_ipc_ptr,
                nbytes,
            )
        return self.hidden_staging[:n]

    def push_tokens_to_ipc(self, num_reqs: int) -> None:
        """torch token_staging → IPC buffer (first ``num_reqs`` rows)."""
        n = int(num_reqs)
        if n < 0 or n > self.max_num_seqs:
            raise ValueError(f"num_reqs={n} out of range")
        nbytes = n * self.num_speculative_tokens * _dtype_nbytes(torch.int64)
        if n > 0:
            memcpy_d2d(
                self.token_ipc_ptr,
                int(self.token_staging.data_ptr()),
                nbytes,
            )

    def hello_payload(self) -> dict[str, Any]:
        return {
            "transport": TRANSPORT_CUDA_IPC,
            "max_tokens": self.max_tokens,
            "max_num_seqs": self.max_num_seqs,
            "hidden_size": self.hidden_size,
            "num_speculative_tokens": self.num_speculative_tokens,
            "dtype": str(self.dtype).removeprefix("torch."),
            "hidden_ipc": self._hidden_handle,
            "token_ipc": self._token_handle,
            "hidden_nbytes": self._hidden_nbytes,
            "token_nbytes": self._token_nbytes,
        }

    def close(self) -> None:
        try:
            if self._hidden_ipc_ptr:
                self._lib.cudaFree(self._hidden_ipc_ptr)
            if self._token_ipc_ptr:
                self._lib.cudaFree(self._token_ipc_ptr)
        except Exception as e:
            logger.warning("CUDA IPC staging free failed: %s", e)
        self._hidden_ipc_ptr = ctypes.c_void_p()
        self._token_ipc_ptr = ctypes.c_void_p()


class CudaIpcClientTransport(DisaggDFlashClientTransport):
    """Verify TP0: ZMQ control + CUDA IPC memcpy for hiddens/tokens."""

    def __init__(
        self,
        address: str,
        timeout_ms: int = 5000,
        preferred_max_tokens: int | None = None,
    ):
        self._zmq = ZmqClientTransport(address, timeout_ms=timeout_ms)
        self._preferred_max_tokens = preferred_max_tokens
        self._handshook = False
        self._max_tokens = 0
        self._max_num_seqs = 0
        self._hidden_size = 0
        self._k = 0
        self._hidden_ptr = 0
        self._token_ptr = 0
        self._dtype = torch.bfloat16
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def handshake(self) -> None:
        if self._handshook:
            return
        hello_kwargs: dict[str, Any] = {}
        if self._preferred_max_tokens is not None:
            hello_kwargs["preferred_max_tokens"] = int(self._preferred_max_tokens)
        frames = self._zmq.send_recv(
            encode_hello(TRANSPORT_CUDA_IPC, **hello_kwargs)
        )
        meta = decode_hello_reply(frames)
        if meta.get("error"):
            raise RuntimeError(f"Disagg-DFlash CUDA IPC HELLO failed: {meta}")
        server_t = meta.get("transport")
        if server_t != TRANSPORT_CUDA_IPC:
            raise RuntimeError(
                f"Verify requested cuda_ipc but draft replied transport={server_t!r}. "
                "Start the draft server with --transport cuda_ipc."
            )
        self._hidden_ptr = open_ipc_ptr(meta["hidden_ipc"])
        self._token_ptr = open_ipc_ptr(meta["token_ipc"])
        self._max_tokens = int(meta["max_tokens"])
        self._max_num_seqs = int(meta["max_num_seqs"])
        self._hidden_size = int(meta["hidden_size"])
        self._k = int(meta["num_speculative_tokens"])
        dtype_name = str(meta.get("dtype", "bfloat16"))
        self._dtype = getattr(torch, dtype_name, torch.bfloat16)
        self._handshook = True
        logger.info(
            "Disagg-DFlash CUDA IPC handshake ok: max_tokens=%d max_seqs=%d H=%d K=%d",
            self._max_tokens,
            self._max_num_seqs,
            self._hidden_size,
            self._k,
        )

    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        if not self._handshook:
            self.handshake()

        hiddens = request.context_hiddens
        if not hiddens.is_cuda:
            hiddens = hiddens.to(device=self.device, non_blocking=True)
        hiddens = hiddens.to(dtype=self._dtype).contiguous()
        n_ctx = int(hiddens.shape[0])
        h = int(hiddens.shape[1]) if hiddens.ndim == 2 else 0
        num_reqs = len(request.req_ids)
        if n_ctx > self._max_tokens:
            raise RuntimeError(
                f"Disagg-DFlash IPC staging too small for {n_ctx} context tokens "
                f"(max_tokens={self._max_tokens}). Raise disagg_dflash_ipc_max_num_tokens "
                "/ --ipc-max-num-tokens on draft."
            )
        if num_reqs > self._max_num_seqs:
            raise RuntimeError(
                f"Disagg-DFlash IPC token staging too small for {num_reqs} reqs "
                f"(max_num_seqs={self._max_num_seqs})."
            )
        if h and h != self._hidden_size:
            raise RuntimeError(
                f"Hidden size mismatch: verify H={h}, draft staging H={self._hidden_size}"
            )

        nbytes = n_ctx * self._hidden_size * hiddens.element_size()
        memcpy_d2d(self._hidden_ptr, int(hiddens.data_ptr()), nbytes)

        ipc_req = DisaggDFlashSpeculateRequest(
            req_ids=request.req_ids,
            context_hiddens=hiddens,
            context_positions=request.context_positions,
            query_start_loc=request.query_start_loc,
            num_rejected=request.num_rejected,
            num_sampled=request.num_sampled,
            last_sampled=request.last_sampled,
            next_prefill_tokens=request.next_prefill_tokens,
            temperature=request.temperature,
            seeds=request.seeds,
            num_speculative_tokens=request.num_speculative_tokens,
            payload_mode=PAYLOAD_IPC,
        )
        resp_frames = self._zmq.send_recv(ipc_req.encode())

        k = int(request.num_speculative_tokens)
        out = torch.empty(num_reqs, k, dtype=torch.int64, device=self.device)
        tok_nbytes = num_reqs * k * out.element_size()
        memcpy_d2d(int(out.data_ptr()), self._token_ptr, tok_nbytes)

        DisaggDFlashSpeculateResponse.decode(
            resp_frames,
            draft_tokens=torch.empty(num_reqs, k, dtype=torch.int64),
        )
        return DisaggDFlashSpeculateResponse(
            draft_tokens=out, payload_mode=PAYLOAD_IPC
        )

    def free(self, request: DisaggDFlashFreeRequest) -> None:
        self._zmq.free(request)

    def ping(self) -> bool:
        return self._zmq.ping()

    def close(self) -> None:
        close_ipc_ptr(self._hidden_ptr)
        close_ipc_ptr(self._token_ptr)
        self._hidden_ptr = 0
        self._token_ptr = 0
        self._zmq.close()
