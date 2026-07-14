# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone DFlash draft engine for the Disagg-DFlash server process."""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    download_weights_from_hf,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.disagg_dflash.protocol import (
    CMD_HELLO,
    PAYLOAD_IPC,
    DisaggDFlashFreeRequest,
    DisaggDFlashSpeculateRequest,
    DisaggDFlashSpeculateResponse,
    encode_hello_reply,
    encode_pong,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_causal
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import prepare_kernel_block_sizes

logger = init_logger(__name__)


def _iter_weights(model_path: str, revision: str | None = None):
    try:
        folder = download_weights_from_hf(
            model_path,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.bin"],
            revision=revision,
        )
    except Exception:
        folder = model_path
    st = []
    bins = []
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".safetensors"):
                st.append(p)
            elif f.endswith(".bin"):
                bins.append(p)
    if st:
        yield from safetensors_weights_iterator(st, use_tqdm_on_load=False)
    else:
        for p in bins:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            yield from sd.items()


def _copy_embedding_and_lm_head_from_target(
    draft_model: nn.Module, target_model: str, dtype: torch.dtype
) -> None:
    """Load target embed_tokens / lm_head into the draft model by value."""
    draft_inner = draft_model.model if hasattr(draft_model, "model") else draft_model
    embed = getattr(draft_inner, "embed_tokens", None)
    lm_head = getattr(draft_model, "lm_head", None)

    need_embed = embed is not None
    need_lm = lm_head is not None
    if not need_embed and not need_lm:
        return

    for name, tensor in _iter_weights(target_model):
        if need_embed and (
            name == "model.embed_tokens.weight" or name.endswith("embed_tokens.weight")
        ):
            weight_loader = getattr(embed.weight, "weight_loader", default_weight_loader)
            weight_loader(embed.weight, tensor.to(dtype=dtype))
            need_embed = False
            logger.info("Disagg-DFlash draft: loaded embed_tokens from target")
        if need_lm and (name == "lm_head.weight" or name.endswith("lm_head.weight")):
            weight_loader = getattr(
                lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(lm_head.weight, tensor.to(dtype=dtype))
            need_lm = False
            logger.info("Disagg-DFlash draft: loaded lm_head from target")
        if not need_embed and not need_lm:
            break

    if need_embed:
        logger.warning("Disagg-DFlash draft: embed_tokens not found in target weights")
    if need_lm:
        logger.warning("Disagg-DFlash draft: lm_head not found in target weights")


class _BlockPool:
    def __init__(self, num_blocks: int):
        # Block 0 reserved as null.
        self.num_blocks = num_blocks
        self._free = list(range(1, num_blocks))

    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(
                f"Disagg-DFlash draft OOM: need {n} blocks, have {len(self._free)} "
                f"(total={self.num_blocks - 1} usable). "
                f"Raise --num-gpu-blocks or check FREE RPCs are reaching the draft."
            )
        out = self._free[:n]
        self._free = self._free[n:]
        return out

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)


class DisaggDFlashDraftEngine:
    """Owns DFlash weights + KV on the draft GPU(s)."""

    def __init__(
        self,
        draft_model: str,
        target_model: str,
        num_speculative_tokens: int,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.8,
        dtype: str = "auto",
        block_size: int = 16,
        num_gpu_blocks: int | None = None,
        attention_backend: str | None = None,
        ipc_max_num_tokens: int = 16384,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.num_query_per_req = 1 + num_speculative_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.ipc_max_num_tokens = ipc_max_num_tokens
        self._ipc_staging = None
        # Tick bumped on every speculate; used for LRU reclaim when the pool
        # is empty (FREE RPC dropped / delayed under load).
        self._tick = 0

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # Resolved after draft weights load (fits into free HBM). CLI None → auto.
        self._num_gpu_blocks_arg = num_gpu_blocks

        # Target config is required by DFlashQwen3ForCausalLM (layer count / vocab).
        target_model_config = ModelConfig(
            model=target_model,
            runner="generate",
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
        )
        from vllm.config import SpeculativeConfig
        from vllm.config.attention import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_backend = None
        if attention_backend is not None:
            attn_backend = AttentionBackendEnum[attention_backend.upper()]

        speculative_config = SpeculativeConfig(
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(tensor_parallel_size=1),
            model=draft_model,
            method="dflash",
            num_speculative_tokens=num_speculative_tokens,
            attention_backend=attn_backend,
        )
        draft_model_config = speculative_config.draft_model_config
        dflash_cfg = getattr(draft_model_config.hf_config, "dflash_config", None) or {}
        causal = bool(dflash_cfg.get("causal", False))

        max_num_batched_tokens = min(
            max_num_seqs * self.num_query_per_req * 4,
            32768,
        )
        self.vllm_config = VllmConfig(
            model_config=target_model_config,
            cache_config=CacheConfig(
                block_size=block_size,
                gpu_memory_utilization=gpu_memory_utilization,
                cache_dtype="auto",
                enable_prefix_caching=False,
            ),
            parallel_config=ParallelConfig(tensor_parallel_size=1),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                is_encoder_decoder=target_model_config.is_encoder_decoder,
            ),
            device_config=DeviceConfig(device="cuda"),
            load_config=LoadConfig(),
            speculative_config=speculative_config,
            attention_config=AttentionConfig(
                use_non_causal=not causal,
                backend=attn_backend,
            ),
        )

        # Model-parallel init requires the current VllmConfig to be set.
        with set_current_vllm_config(self.vllm_config):
            self._init_distributed()
            # Load DFlash weights as the draft model_config (same as colocated path).
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=draft_model_config,
            )
            _copy_embedding_and_lm_head_from_target(
                self.model, target_model, self.vllm_config.model_config.dtype
            )
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "_build_fused_kv_buffers"
            ):
                self.model.model._build_fused_kv_buffers()

            self.parallel_drafting_token_id = get_parallel_drafting_token_id(
                draft_model_config.hf_config
            )
            self.dflash_causal = get_dflash_causal(draft_model_config)
            self.hidden_size = draft_model_config.get_hidden_size()
            self.dtype = self.vllm_config.model_config.dtype
            draft_inner = (
                self.model.model if hasattr(self.model, "model") else self.model
            )
            embed = getattr(draft_inner, "embed_tokens", None)
            if embed is None and hasattr(self.model, "get_input_embeddings"):
                embed = self.model.get_input_embeddings()
            if embed is None:
                raise RuntimeError(
                    "Disagg-DFlash draft: could not resolve embed_tokens for vocab size"
                )
            self.embed_vocab_size = int(embed.weight.shape[0])
            if (
                self.parallel_drafting_token_id < 0
                or self.parallel_drafting_token_id >= self.embed_vocab_size
            ):
                raise RuntimeError(
                    "Disagg-DFlash mask_token_id="
                    f"{self.parallel_drafting_token_id} out of range for "
                    f"embed vocab_size={self.embed_vocab_size}"
                )

            # KV / attention setup (must stay inside config context).
            # Use draft model layers for KV specs: temporarily swap model_config
            # for get_kv_cache_spec discovery against the loaded draft module.
            kv_cache_spec = get_kv_cache_spec(self.vllm_config)
            # If target has no draft layers in forward context yet, fall back to
            # inspecting the loaded draft module's attention layers.
            if not kv_cache_spec:
                from vllm.config import get_layers_from_vllm_config
                from vllm.model_executor.layers.attention_layer_base import (
                    AttentionLayerBase,
                )

                # Ensure draft attentions are registered in static_forward_context
                # by re-running get_kv_cache_spec after model init (layers register
                # themselves during construction under set_current_vllm_config).
                kv_cache_spec = get_kv_cache_spec(self.vllm_config)

            from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
            from vllm.v1.kv_cache_interface import KVCacheTensor

            layer_names = list(kv_cache_spec.keys())
            assert layer_names, "DFlash draft model produced no KV cache specs"
            kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
            assert kv_cache_groups, "DFlash draft produced no KV cache groups"

            # Bytes for one logical block id across all independent arenas.
            bytes_per_block = sum(
                int(g.kv_cache_spec.page_size_bytes) for g in kv_cache_groups
            )
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
            # Leave room for activations / fragmentation.
            usable = int(free_bytes * float(gpu_memory_utilization) * 0.90)
            mem_blocks = max(usable // max(bytes_per_block, 1), 1024)

            # Target working set: enough for concurrent seqs at a modest
            # context (1024 tokens). Huge max_model_len × max_num_seqs will
            # not fit on one GPU — reclaim + FREE handle the rest.
            target_blocks = max(
                max_num_seqs * ((1024 + block_size - 1) // block_size) + 1,
                16384,
            )
            if self._num_gpu_blocks_arg is None:
                num_gpu_blocks = min(mem_blocks, target_blocks, 65536)
                logger.info(
                    "Disagg-DFlash draft: auto --num-gpu-blocks=%d "
                    "(mem_cap=%d target=%d; %.1f GiB free, %.1f KiB/block×%d groups)",
                    num_gpu_blocks,
                    mem_blocks,
                    target_blocks,
                    free_bytes / (1024**3),
                    bytes_per_block / 1024,
                    len(kv_cache_groups),
                )
            else:
                num_gpu_blocks = int(self._num_gpu_blocks_arg)
                if num_gpu_blocks > mem_blocks:
                    logger.warning(
                        "Disagg-DFlash draft: --num-gpu-blocks=%d exceeds "
                        "estimated free HBM capacity (%d); clamping",
                        num_gpu_blocks,
                        mem_blocks,
                    )
                    num_gpu_blocks = mem_blocks
                elif num_gpu_blocks < max_num_seqs * 2:
                    logger.warning(
                        "Disagg-DFlash draft: --num-gpu-blocks=%d is very small "
                        "for max_num_seqs=%d (≈%.1f blocks/seq). Under load "
                        "the pool will reclaim/steal finished slots.",
                        num_gpu_blocks,
                        max_num_seqs,
                        num_gpu_blocks / max(max_num_seqs, 1),
                    )

            # One arena PER KV group. Qwen3.5-DFlash produces 6 groups
            # (5× SlidingWindowSpec + 1× FullAttentionSpec). Sharing a single
            # raw tensor across those groups while mirroring the same block IDs
            # made every layer overwrite the same physical pages → ~0% accept.
            # Independent arenas let us safely mirror block IDs across groups.
            kv_cache_tensors = [
                KVCacheTensor(
                    size=int(group.kv_cache_spec.page_size_bytes) * num_gpu_blocks,
                    shared_by=list(group.layer_names),
                )
                for group in kv_cache_groups
            ]
            self.kv_cache_config = KVCacheConfig(
                num_blocks=num_gpu_blocks,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
            )
            self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks

            self.attn_groups, _, _ = init_attn_backend(
                self.kv_cache_config,
                self.vllm_config,
                self.device,
                active_layer_names=set(layer_names),
            )
            self.kernel_block_sizes = prepare_kernel_block_sizes(
                self.kv_cache_config, self.attn_groups
            )
            max_blocks = (max_model_len + block_size - 1) // block_size + 2
            n_groups = len(self.kv_cache_config.kv_cache_groups)
            if len(set(self.kernel_block_sizes)) > 1:
                raise NotImplementedError(
                    "Disagg-DFlash draft currently requires uniform "
                    f"kernel_block_sizes across KV groups; got {self.kernel_block_sizes}"
                )
            self.block_tables = BlockTables(
                block_sizes=[block_size] * n_groups,
                max_num_reqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_blocks_per_group=[max_blocks] * n_groups,
                device=self.device,
                kernel_block_sizes=self.kernel_block_sizes,
            )
            logger.info(
                "Disagg-DFlash draft KV groups: %d (%s); "
                "allocated %d independent KV arenas × %d blocks",
                n_groups,
                [
                    (len(g.layer_names), type(g.kv_cache_spec).__name__)
                    for g in kv_cache_groups
                ],
                len(kv_cache_tensors),
                num_gpu_blocks,
            )
            self._runner_kv_caches: list = []
            init_kv_cache(
                self._runner_kv_caches,
                self.vllm_config.compilation_config.static_forward_context,
                self.kv_cache_config,
                self.attn_groups,
                self.device,
                self.vllm_config.cache_config.cache_dtype,
                self.kernel_block_sizes,
                self.vllm_config,
            )

        self.block_pool = _BlockPool(num_gpu_blocks)
        # req_id -> state
        self._seqs: dict[str, dict[str, Any]] = {}
        self._req_id_to_slot: dict[str, int] = {}
        # Pop from the front so early requests land in low slot rows. Still
        # must gather block-table rows by *actual* slot below — never assume
        # the batch is packed into rows [0, num_reqs).
        self._free_slots: deque[int] = deque(range(max_num_seqs))

        self.input_buffers = InputBuffers(
            max_num_reqs=max_num_seqs,
            max_num_tokens=max_num_seqs * self.num_query_per_req,
            device=self.device,
        )
        self.draft_tokens = torch.zeros(
            max_num_seqs,
            num_speculative_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        logger.info(
            "Disagg-DFlash draft engine ready: draft=%s target_embed_from=%s K=%d",
            draft_model,
            target_model,
            num_speculative_tokens,
        )

    def _init_distributed(self) -> None:
        if not torch.distributed.is_initialized():
            init_method = get_distributed_init_method(get_ip(), get_open_port())
            # Use gloo for the single-process draft server so we do not
            # interfere with the verify fleet's NCCL process groups.
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=init_method,
                local_rank=0,
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1)

    def free(self, request: DisaggDFlashFreeRequest) -> list[bytes]:
        from vllm.v1.spec_decode.disagg_dflash.protocol import encode_pong

        for req_id in request.req_ids:
            state = self._seqs.pop(req_id, None)
            slot = self._req_id_to_slot.pop(req_id, None)
            if state is not None:
                self.block_pool.free(state["blocks"])
            if slot is not None:
                self._free_slots.append(slot)
                # Clear block table row (gpu + CPU mirror used for slot mapping).
                for bt in self.block_tables.block_tables:
                    bt.gpu[slot].zero_()
                for ibt in self.block_tables.input_block_tables:
                    ibt[slot].zero_()
                self.block_tables.num_blocks.gpu[:, slot] = 0
        return encode_pong()

    def _ensure_seq(self, req_id: str) -> int:
        if req_id in self._req_id_to_slot:
            return self._req_id_to_slot[req_id]
        if not self._free_slots:
            # Last-resort reclaim: drop the oldest tracked request.
            # This should not happen if FREE RPCs are delivered; log loudly.
            victim = next(iter(self._req_id_to_slot))
            logger.warning(
                "Disagg-DFlash draft: no free slots (%d in use); "
                "force-freeing oldest req_id=%s (FREE RPC likely dropped)",
                len(self._req_id_to_slot),
                victim,
            )
            self.free(DisaggDFlashFreeRequest(req_ids=[victim]))
        if not self._free_slots:
            raise RuntimeError("Disagg-DFlash draft: no free request slots")
        slot = self._free_slots.popleft()
        self._req_id_to_slot[req_id] = slot
        self._seqs[req_id] = {
            "slot": slot,
            "blocks": [],
            "ctx_len": 0,
            "last_used": self._tick,
        }
        return slot

    def _reclaim_blocks(self, need: int, protect: set[str]) -> None:
        """Free LRU sequences not in ``protect`` until ``need`` blocks are free."""
        if self.block_pool.num_free() >= need:
            return
        victims = sorted(
            (
                (rid, state)
                for rid, state in self._seqs.items()
                if rid not in protect
            ),
            key=lambda x: x[1].get("last_used", 0),
        )
        for rid, state in victims:
            if self.block_pool.num_free() >= need:
                break
            n_blk = len(state["blocks"])
            logger.warning(
                "Disagg-DFlash draft: reclaiming req_id=%s (%d blocks, "
                "free=%d need=%d, %d seqs live) — FREE likely dropped/delayed",
                rid,
                n_blk,
                self.block_pool.num_free(),
                need,
                len(self._seqs),
            )
            self.free(DisaggDFlashFreeRequest(req_ids=[rid]))

    def _ensure_blocks(
        self, req_id: str, needed_tokens: int, protect: set[str] | None = None
    ) -> None:
        state = self._seqs[req_id]
        slot = state["slot"]
        # Cap to draft max_model_len — verify may have a much larger budget.
        needed_tokens = min(needed_tokens, self.max_model_len)
        # When SWA/full share the same page size we mirror one physical block
        # list into every KV group (required for mixed Qwen3.5-DFlash).
        need_blocks = (needed_tokens + self.block_size - 1) // self.block_size
        have = len(state["blocks"])
        changed = False
        if need_blocks > have:
            extra = need_blocks - have
            protected = protect if protect is not None else {req_id}
            try:
                new_blocks = self.block_pool.allocate(extra)
            except RuntimeError:
                self._reclaim_blocks(extra, protect=protected)
                new_blocks = self.block_pool.allocate(extra)
            state["blocks"].extend(new_blocks)
            changed = True
        # Skip host→GPU rewrite when the block list is unchanged.
        if not changed and state.get("bt_synced_n") == have:
            return
        blocks = state["blocks"]
        n = len(blocks)
        blocks_t = torch.as_tensor(blocks, dtype=torch.int32, device=self.device)
        n_groups = len(self.block_tables.block_tables)
        for gid in range(n_groups):
            bt = self.block_tables.block_tables[gid].gpu
            bt[slot, :n] = blocks_t
            self.block_tables.num_blocks.gpu[gid, slot] = n
            self.block_tables.input_block_tables[gid][slot, :n].copy_(blocks_t)
        state["bt_synced_n"] = n

    def _slot_mapping_for_positions(
        self, slot: int, positions: torch.Tensor, group_id: int = 0
    ) -> torch.Tensor:
        """Compute paged slot ids for absolute positions of one request."""
        bs = int(self.block_tables.block_sizes[group_id])
        blocks = self.block_tables.input_block_tables[group_id][slot]
        block_ids = blocks[positions // bs]
        offsets = positions % bs
        return (block_ids.to(dtype=torch.int64) * bs + offsets).to(dtype=torch.int64)

    @staticmethod
    def _as_numpy_i64(t: torch.Tensor) -> np.ndarray:
        """Host numpy view of a meta tensor (never .item() a CUDA tensor)."""
        if t.is_cuda:
            t = t.detach().cpu()
        return np.asarray(t.numpy(), dtype=np.int64)

    @staticmethod
    def _as_numpy_i32(t: torch.Tensor) -> np.ndarray:
        if t.is_cuda:
            t = t.detach().cpu()
        return np.asarray(t.numpy(), dtype=np.int32)

    @torch.inference_mode()
    def speculate(
        self, request: DisaggDFlashSpeculateRequest
    ) -> DisaggDFlashSpeculateResponse:
        num_reqs = len(request.req_ids)
        device = self.device

        # Hiddens / positions needed on GPU for KV write. Meta tensors stay on
        # host — decoding already produced CPU tensors; uploading then .item()
        # was the main draft-side latency tax at n≈64.
        ctx_h = request.context_hiddens
        if not ctx_h.is_cuda:
            ctx_h = ctx_h.to(device=device, dtype=self.dtype, non_blocking=True)
        else:
            ctx_h = ctx_h.to(dtype=self.dtype)
        ctx_pos = request.context_positions
        if not ctx_pos.is_cuda:
            ctx_pos_gpu = ctx_pos.to(device=device, dtype=torch.int64, non_blocking=True)
            ctx_pos_np = self._as_numpy_i64(ctx_pos)
        else:
            ctx_pos_gpu = ctx_pos.to(dtype=torch.int64)
            ctx_pos_np = self._as_numpy_i64(ctx_pos)

        qsl_np = self._as_numpy_i32(request.query_start_loc)
        num_rejected_np = self._as_numpy_i32(request.num_rejected)
        num_sampled_np = self._as_numpy_i32(request.num_sampled)
        last_sampled_np = self._as_numpy_i64(request.last_sampled)
        next_prefill_np = self._as_numpy_i64(request.next_prefill_tokens)

        slots = [self._ensure_seq(rid) for rid in request.req_ids]
        self._tick += 1
        protect = set(request.req_ids)
        for rid in request.req_ids:
            self._seqs[rid]["last_used"] = self._tick

        # Mirror colocated prepare_dflash_inputs:
        # - write KV for all scheduled context tokens (incl. rejected tail slots)
        # - place query after the last *valid* context position
        # - do NOT roll back ctx_len by num_rejected (rejected tokens were never
        #   committed as durable context length; they arrive in this batch)
        context_slot_mapping = torch.zeros(
            ctx_h.shape[0], dtype=torch.int64, device=device
        )
        last_valid_pos_np = np.zeros(num_reqs, dtype=np.int64)

        for i, req_id in enumerate(request.req_ids):
            state = self._seqs[req_id]
            slot = slots[i]
            s = int(qsl_np[i])
            e = int(qsl_np[i + 1])
            n_rej = int(num_rejected_np[i])
            valid_e = e - n_rej
            n_ctx = e - s

            if n_ctx > 0:
                max_pos = int(ctx_pos_np[s:e].max())
                self._ensure_blocks(
                    req_id,
                    max_pos + 1 + self.num_query_per_req,
                    protect=protect,
                )
                context_slot_mapping[s:e] = self._slot_mapping_for_positions(
                    slot, ctx_pos_gpu[s:e]
                )

            if valid_e > s:
                lv = int(ctx_pos_np[valid_e - 1])
                last_valid_pos_np[i] = lv
                state["ctx_len"] = max(state["ctx_len"], lv + 1)
            else:
                # No valid new context this step; keep prior length.
                last_valid_pos_np[i] = max(0, state["ctx_len"] - 1)

        # Precompute context K/V for all scheduled context tokens (rejected
        # slots are overwritten; query placement ignores them via last_valid_pos).
        if ctx_h.shape[0] > 0:
            n_groups = self.block_tables.slot_mappings.shape[0]
            if n_groups <= 1:
                context_slots: torch.Tensor | list[torch.Tensor | None] = (
                    context_slot_mapping
                )
            else:
                # Same page size across groups → reuse group-0 slot map.
                # (Qwen3.5-DFlash mirrors one physical block list into every group.)
                same_bs = len({int(b) for b in self.block_tables.block_sizes}) == 1
                if same_bs:
                    group_slots = [context_slot_mapping] * n_groups
                else:
                    group_slots = []
                    for gid in range(n_groups):
                        sm = torch.zeros(
                            ctx_h.shape[0], dtype=torch.int64, device=device
                        )
                        for i, req_id in enumerate(request.req_ids):
                            slot = slots[i]
                            s = int(qsl_np[i])
                            e = int(qsl_np[i + 1])
                            if e > s:
                                sm[s:e] = self._slot_mapping_for_positions(
                                    slot, ctx_pos_gpu[s:e], group_id=gid
                                )
                        group_slots.append(sm)
                # Map each draft attention layer to its KV group.
                layer_names = []
                if hasattr(self.model, "get_draft_kv_cache_layer_names"):
                    layer_names = list(self.model.get_draft_kv_cache_layer_names())
                if not layer_names:
                    layer_names = [
                        ln
                        for g in self.kv_cache_config.kv_cache_groups
                        for ln in g.layer_names
                    ]
                name_to_gid = {
                    ln: gid
                    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
                    for ln in group.layer_names
                }
                context_slots = [
                    group_slots[name_to_gid[name]]
                    if name in name_to_gid
                    else group_slots[0]
                    for name in layer_names
                ]
            self.model.precompute_and_store_context_kv(
                ctx_h,
                ctx_pos_gpu,
                context_slots,
            )

        # Build query tokens: bonus + mask tokens (batched GPU fills).
        num_query = num_reqs * self.num_query_per_req
        vocab_size = int(self.embed_vocab_size)
        mask_id = int(self.parallel_drafting_token_id)
        if mask_id < 0 or mask_id >= vocab_size:
            raise RuntimeError(
                f"Disagg-DFlash mask_token_id={mask_id} out of range for "
                f"embed vocab_size={vocab_size}"
            )
        input_ids = torch.full(
            (num_query,),
            mask_id,
            dtype=torch.int32,
            device=device,
        )
        positions = torch.zeros(num_query, dtype=torch.int64, device=device)
        query_slot_mapping = torch.zeros(num_query, dtype=torch.int64, device=device)
        seq_lens_np = np.zeros(num_reqs, dtype=np.int32)

        # Bonus selection on host (meta is already CPU).
        bonus_np = np.where(
            num_sampled_np[:num_reqs] > 0,
            last_sampled_np[:num_reqs],
            next_prefill_np[:num_reqs],
        ).astype(np.int64, copy=False)
        oob = (bonus_np < 0) | (bonus_np >= vocab_size)
        if np.any(oob):
            for i in np.nonzero(oob)[0].tolist():
                logger.warning(
                    "Disagg-DFlash draft: OOB bonus token %d (req=%s, vocab=%d); "
                    "clamping into [0, %d]",
                    int(bonus_np[i]),
                    request.req_ids[i],
                    vocab_size,
                    vocab_size - 1,
                )
                bonus_np[i] = 0 if bonus_np[i] < 0 else vocab_size - 1

        q_offsets = torch.arange(
            self.num_query_per_req, device=device, dtype=torch.int64
        )
        max_pos_cap = self.max_model_len - 1
        max_pos_cap_t = torch.tensor(max_pos_cap, device=device)
        for i, req_id in enumerate(request.req_ids):
            state = self._seqs[req_id]
            slot = slots[i]
            base = i * self.num_query_per_req
            input_ids[base] = int(bonus_np[i])
            lv = int(last_valid_pos_np[i])
            self._ensure_blocks(
                req_id, lv + 1 + self.num_query_per_req, protect=protect
            )
            qpos = torch.minimum(lv + 1 + q_offsets, max_pos_cap_t)
            positions[base : base + self.num_query_per_req] = qpos
            query_slot_mapping[base : base + self.num_query_per_req] = (
                self._slot_mapping_for_positions(slot, qpos)
            )
            seq_lens_np[i] = lv + 1 + self.num_query_per_req
            state["ctx_len"] = max(state["ctx_len"], lv + 1)

        seq_lens = torch.as_tensor(seq_lens_np, device=device, dtype=torch.int32)

        self.input_buffers.input_ids[:num_query].copy_(input_ids)
        self.input_buffers.positions[:num_query].copy_(positions)
        self.input_buffers.query_start_loc[: num_reqs + 1] = (
            torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
            * self.num_query_per_req
        )
        self.input_buffers.seq_lens[:num_reqs].copy_(seq_lens)

        # Slot mappings for every KV group (SWA vs full may be separate groups).
        for gid in range(self.block_tables.slot_mappings.shape[0]):
            self.block_tables.slot_mappings[gid, :num_query].copy_(query_slot_mapping)

        query_start_loc_cpu = (
            torch.arange(num_reqs + 1, dtype=torch.int32) * self.num_query_per_req
        )
        max_seq_len = int(seq_lens_np.max()) if num_reqs > 0 else 1

        # Per-layer causal when draft has mixed SWA/full attention.
        causal: bool | dict[int, bool] = self.dflash_causal
        if hasattr(self.model, "get_draft_attn_causal") and hasattr(
            self.model, "get_draft_kv_cache_layer_names"
        ):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            layer_causal = self.model.get_draft_attn_causal()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(self.kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            causal = {
                name_to_gid[name]: layer_causal[i]
                for i, name in enumerate(layer_names)
                if name in name_to_gid
            }

        # CRITICAL: gather block-table rows by the request's *allocated slot*,
        # not rows [0, num_reqs). Slots are an arena address — after warmup or
        # any FREE/reuse they are not dense at the front. Using bt[:num_reqs]
        # made attention page in zeros while K/V lived on the real slot rows →
        # ~0% acceptance despite valid drafts flowing.
        slot_idx = torch.as_tensor(slots, device=device, dtype=torch.long)
        attn_block_tables = [
            bt.index_select(0, slot_idx)
            for bt in self.block_tables.input_block_tables
        ]

        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_query,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=self.num_query_per_req,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=max_seq_len,
            block_tables=attn_block_tables,
            slot_mappings=self.block_tables.slot_mappings[:, :num_query],
            kv_cache_config=self.kv_cache_config,
            causal=causal,
        )
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        slot_mappings = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_query], self.kv_cache_config
        )

        batch_descriptor = BatchDescriptor(num_tokens=num_query)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_query,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden = self.model(
                input_ids=self.input_buffers.input_ids[:num_query],
                positions=self.input_buffers.positions[:num_query],
                inputs_embeds=None,
            )

        # Sample from mask-token positions (skip bonus at offset 0).
        if num_reqs > 0:
            bases = (
                torch.arange(num_reqs, device=device, dtype=torch.long)
                * self.num_query_per_req
            )
            offs = torch.arange(
                1, self.num_query_per_req, device=device, dtype=torch.long
            )
            sample_idx_t = (bases.unsqueeze(1) + offs).reshape(-1)
        else:
            sample_idx_t = torch.empty(0, device=device, dtype=torch.long)
        logits = self.model.compute_logits(hidden[sample_idx_t])
        tokens = logits.argmax(dim=-1).view(num_reqs, self.num_speculative_tokens)
        self.draft_tokens[:num_reqs].copy_(tokens)

        if request.payload_mode == PAYLOAD_IPC:
            staging = self.ensure_ipc_staging()
            staging.token_staging[:num_reqs].copy_(
                self.draft_tokens[:num_reqs], non_blocking=True
            )
            # Publish tokens into the cudaMalloc IPC buffer for verify to read.
            staging.push_tokens_to_ipc(num_reqs)
            return DisaggDFlashSpeculateResponse(
                draft_tokens=staging.token_staging[:num_reqs],
                payload_mode=PAYLOAD_IPC,
            )

        return DisaggDFlashSpeculateResponse(
            draft_tokens=self.draft_tokens[:num_reqs].detach().cpu()
        )

    def ensure_ipc_staging(self):
        """Allocate (once) draft-owned CUDA IPC staging buffers."""
        if self._ipc_staging is not None:
            return self._ipc_staging
        from vllm.v1.spec_decode.disagg_dflash.transport_cuda_ipc import (
            CudaIpcStaging,
        )

        h = self.vllm_config.model_config.get_hidden_size()
        self._ipc_staging = CudaIpcStaging(
            max_tokens=self.ipc_max_num_tokens,
            max_num_seqs=self.max_num_seqs,
            hidden_size=h,
            num_speculative_tokens=self.num_speculative_tokens,
            dtype=self.dtype,
            device=self.device,
        )
        logger.info(
            "Disagg-DFlash draft: CUDA IPC staging ready "
            "(max_tokens=%d max_seqs=%d H=%d K=%d dtype=%s)",
            self.ipc_max_num_tokens,
            self.max_num_seqs,
            h,
            self.num_speculative_tokens,
            self.dtype,
        )
        return self._ipc_staging

    def handle_hello(self, hello_meta: dict) -> list[bytes]:
        transport = str(hello_meta.get("transport", "zmq")).lower()
        if transport == "zmq":
            return encode_hello_reply({"transport": "zmq"})
        if transport == "cuda_ipc":
            preferred = hello_meta.get("preferred_max_tokens")
            if preferred is not None:
                try:
                    pref = int(preferred)
                    if pref > self.ipc_max_num_tokens:
                        logger.info(
                            "Disagg-DFlash draft: raising ipc_max_num_tokens "
                            "%d → %d from verify HELLO",
                            self.ipc_max_num_tokens,
                            pref,
                        )
                        self.ipc_max_num_tokens = pref
                        # Force re-allocate if already staged smaller.
                        if (
                            self._ipc_staging is not None
                            and self._ipc_staging.max_tokens < pref
                        ):
                            self._ipc_staging = None
                except (TypeError, ValueError):
                    pass
            staging = self.ensure_ipc_staging()
            return encode_hello_reply(staging.hello_payload())
        if transport in ("nccl", "nixl"):
            return encode_hello_reply(
                {
                    "transport": "zmq",
                    "error": f"transport {transport!r} not implemented",
                }
            )
        return encode_hello_reply(
            {"transport": "zmq", "error": f"unknown transport {transport!r}"}
        )

    def warmup(self) -> None:
        """Run one dummy speculate so torch.compile / cudagraph happen before serve."""
        from vllm.v1.spec_decode.disagg_dflash.protocol import (
            DisaggDFlashFreeRequest,
            DisaggDFlashSpeculateRequest,
        )

        logger.info("Disagg-DFlash draft: running compile/warmup speculate...")
        h = self.vllm_config.model_config.get_hidden_size()
        req = DisaggDFlashSpeculateRequest(
            req_ids=["__warmup__"],
            context_hiddens=torch.zeros(1, h, dtype=self.dtype),
            context_positions=torch.zeros(1, dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            num_rejected=torch.zeros(1, dtype=torch.int32),
            num_sampled=torch.ones(1, dtype=torch.int32),
            last_sampled=torch.zeros(1, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
            temperature=torch.zeros(1, dtype=torch.float32),
            seeds=torch.zeros(1, dtype=torch.int64),
            num_speculative_tokens=self.num_speculative_tokens,
        )
        self.speculate(req)
        self.free(DisaggDFlashFreeRequest(req_ids=["__warmup__"]))
        logger.info("Disagg-DFlash draft: warmup complete")

    def handle(self, cmd: str, payload) -> list[bytes]:
        if cmd == CMD_HELLO:
            return self.handle_hello(payload)
        if cmd == "speculate":
            # IPC pull already synchronized inside pull_hiddens_from_ipc /
            # memcpy_d2d when using cuda_ipc; ZMQ path needs no extra sync.
            return self.speculate(payload).encode()
        if cmd == "free":
            return self.free(payload)
        return encode_pong()
