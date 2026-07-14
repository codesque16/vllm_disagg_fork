# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side Disagg-DFlash proxy (Model Runner V2 / BaseSpeculator)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.disagg_dflash.common import pack_speculate_request
from vllm.v1.spec_decode.disagg_dflash.projector import (
    build_and_load_projector,
    reduce_aux_hidden_states,
)
from vllm.v1.spec_decode.disagg_dflash.protocol import DisaggDFlashFreeRequest
from vllm.v1.spec_decode.disagg_dflash.transport import (
    DisaggDFlashClientTransport,
    create_client_transport,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


def _profile_enabled() -> bool:
    return os.environ.get("VLLM_DISAGG_DFLASH_PROFILE", "0") == "1"


def _cross_step_enabled() -> bool:
    # Default ON: defer propose_finish into the next execute_model so draft
    # RTT overlaps EngineCore / schedule / preamble. Set to 0 to force
    # in-step begin+finish (Stage 2 only).
    return os.environ.get("VLLM_DISAGG_DFLASH_CROSS_STEP", "1") != "0"


@dataclass
class DisaggProposePending:
    """In-flight Disagg-DFlash propose between fire and wait."""

    num_reqs: int
    early_done: bool = False
    profile: bool = False
    t_proj0: float = 0.0
    t_proj1: float = 0.0
    t_pack0: float = 0.0
    t_pack1: float = 0.0
    t_fire: float = 0.0
    speculate_ms: dict[str, float] = field(default_factory=dict)
    # Fire-time request-state indices — write drafts here at finish even if
    # the next execute_model batch order differs (cross-step pipeline).
    idx_mapping: torch.Tensor | None = None
    req_ids: list[str] = field(default_factory=list)


class DisaggDFlashProxy(BaseSpeculator):
    """Remote DFlash drafter. Does not load draft transformer weights."""

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.use_disagg_dflash()
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        self._tp_rank = get_tp_group().rank_in_group
        self._tp_size = get_tp_group().world_size
        self._client: DisaggDFlashClientTransport | None = None
        self._projector: nn.Module | None = None
        self._known_req_ids: set[str] = set()
        self._transport_name = self.speculative_config.disagg_dflash_transport
        self._profile_steps = 0
        self._profile_every = int(
            os.environ.get("VLLM_DISAGG_DFLASH_PROFILE_EVERY", "50")
        )
        self._cross_step = _cross_step_enabled()
        # Cross-step: one in-flight speculate whose finish is deferred into the
        # next execute_model (before FREE / prepare_inputs).
        self._deferred_pending: DisaggProposePending | None = None
        # Side-stream projector so reduce_aux can run under rejection sampling.
        self._proj_stream = torch.cuda.Stream(device=device)
        self._proj_ready = torch.cuda.Event(enable_timing=False)
        self._proj_reduced: torch.Tensor | None = None
        self._proj_num_tokens: int = 0
        self._proj_t0: float = 0.0
        self._proj_inflight: bool = False

        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        # Token-only wire protocol: rejection sampling uses target logits only.
        self.draft_logits: torch.Tensor | None = None
        # Dummy attribute so isinstance(..., DraftModelSpeculator) paths skip us.
        self.model = None

    def load_model(self, target_model: nn.Module) -> None:
        self._projector = build_and_load_projector(self.vllm_config, self.device)
        if self._tp_rank == 0:
            address = self.speculative_config.disagg_dflash_address
            assert address is not None
            self._client = create_client_transport(
                address=address,
                timeout_ms=self.speculative_config.disagg_dflash_timeout_ms,
                transport=self._transport_name,
                ipc_max_num_tokens=(
                    self.speculative_config.disagg_dflash_ipc_max_num_tokens
                ),
            )
            if not self._client.ping():
                logger.warning(
                    "Disagg-DFlash draft server did not answer ping at %s", address
                )
            self._client.handshake()
        logger.info(
            "Disagg-DFlash proxy ready (tp_rank=%d, address=%s, transport=%s, "
            "cross_step=%s)",
            self._tp_rank,
            self.speculative_config.disagg_dflash_address,
            self._transport_name,
            self._cross_step,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        return

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        return

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        return

    def begin_reduce(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_tokens: int,
    ) -> None:
        """Launch L*H→H projector on a side stream (overlaps with sample())."""
        if self._projector is None:
            self._proj_inflight = False
            self._proj_reduced = last_hidden_states[:num_tokens]
            self._proj_num_tokens = num_tokens
            return
        profile = _profile_enabled() and self._tp_rank == 0
        self._proj_t0 = time.perf_counter() if profile else 0.0
        self._proj_num_tokens = num_tokens
        self._proj_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._proj_stream):
            self._proj_reduced = reduce_aux_hidden_states(
                self._projector,
                last_hidden_states[:num_tokens],
                (
                    [h[:num_tokens] for h in aux_hidden_states]
                    if aux_hidden_states is not None
                    else None
                ),
            )
        self._proj_ready.record(self._proj_stream)
        self._proj_inflight = True

    def _take_reduced(self, profile: bool) -> tuple[torch.Tensor, float, float]:
        """Wait for side-stream reduce (or no-op) and return (tensor, t0, t1)."""
        t0 = self._proj_t0 if profile else 0.0
        if self._proj_inflight:
            self._proj_ready.synchronize()
            torch.cuda.current_stream(self.device).wait_stream(self._proj_stream)
            self._proj_inflight = False
        t1 = time.perf_counter() if profile else 0.0
        assert self._proj_reduced is not None
        return self._proj_reduced, t0, t1

    def free_requests(self, req_ids: list[str]) -> None:
        if not req_ids:
            return
        # DEALER is in-order: never FREE while a speculate reply is pending.
        self.drain_deferred(req_states=None)
        # Always notify draft for finished IDs we have seen; also drop unknown
        # IDs from local tracking so verify-side set cannot grow unbounded.
        to_free = [r for r in req_ids if r in self._known_req_ids]
        for r in req_ids:
            self._known_req_ids.discard(r)
        if not to_free or self._tp_rank != 0 or self._client is None:
            return
        try:
            self._client.free(DisaggDFlashFreeRequest(req_ids=to_free))
        except Exception as e:
            logger.warning("Disagg-DFlash FREE failed: %s", e)

    @property
    def cross_step_enabled(self) -> bool:
        return self._cross_step

    def defer_pending(self, pending: DisaggProposePending) -> None:
        """Hold fire-time pending until the next execute_model drains it."""
        if self._deferred_pending is not None and not self._deferred_pending.early_done:
            raise RuntimeError(
                "Disagg-DFlash cross-step: new propose_begin while previous "
                "speculate is still in flight"
            )
        self._deferred_pending = pending

    def drain_deferred(self, req_states: Any | None) -> torch.Tensor | None:
        """Finish a deferred speculate; optionally write into req_states.draft_tokens.

        Returns draft tokens when a pending fire was drained; else None.
        """
        pending = self._deferred_pending
        if pending is None:
            return None
        self._deferred_pending = None
        draft = self.propose_finish(pending)
        if req_states is not None and pending.idx_mapping is not None:
            req_states.draft_tokens[pending.idx_mapping] = draft
        return draft

    def _resolve_bonus_tokens(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Build per-request bonus token for the draft (batch-local)."""
        idx = input_batch.idx_mapping[:num_reqs].long()
        # Batch-local sampler output [num_reqs, K+1] (Stage 2, before postprocess)
        # vs request-state table [max_num_reqs] / [max_num_reqs, 1].
        if last_sampled.dim() == 2 and last_sampled.shape[0] == num_reqs:
            if last_sampled.shape[-1] == 1:
                ls = last_sampled[:, 0].to(dtype=torch.int64)
            else:
                ns_i64 = num_sampled[:num_reqs].to(dtype=torch.int64)
                gather_idx = (ns_i64 - 1).clamp(min=0).unsqueeze(1)
                ls = (
                    last_sampled.gather(1, gather_idx).squeeze(1).to(dtype=torch.int64)
                )
        elif last_sampled.dim() == 1:
            if last_sampled.shape[0] == num_reqs:
                ls = last_sampled[:num_reqs].to(dtype=torch.int64)
            else:
                ls = last_sampled.index_select(0, idx).to(dtype=torch.int64)
        elif last_sampled.dim() == 2 and last_sampled.shape[-1] == 1:
            ls = last_sampled[:, 0].index_select(0, idx).to(dtype=torch.int64)
        else:
            ls = last_sampled.reshape(-1).index_select(0, idx).to(dtype=torch.int64)

        if next_prefill_tokens.dim() == 1:
            if next_prefill_tokens.shape[0] == num_reqs:
                npf = next_prefill_tokens.to(dtype=torch.int64)
            else:
                npf = next_prefill_tokens.index_select(0, idx).to(dtype=torch.int64)
        else:
            npf = (
                next_prefill_tokens.reshape(-1)
                .index_select(0, idx)
                .to(dtype=torch.int64)
            )

        ns = num_sampled[:num_reqs]
        bonus = torch.where(ns > 0, ls, npf).reshape(-1)[:num_reqs].contiguous()

        vocab = self.vllm_config.model_config.get_vocab_size()
        oob = (bonus < 0) | (bonus >= vocab)
        if bool(oob.any().item()):
            # Prefer the first logit slot (combine wrote last_sampled there)
            # over the rejected draft tail.
            qsl = input_batch.query_start_loc
            logits_indices = input_batch.logits_indices
            cu = input_batch.cu_num_logits
            before = bonus[oob][:8].detach().tolist()
            for i in torch.nonzero(oob, as_tuple=False).view(-1).tolist():
                repaired = False
                if cu is not None and logits_indices is not None:
                    logit0 = int(cu[i].item())
                    tok = int(input_batch.input_ids[logits_indices[logit0]].item())
                    if 0 <= tok < vocab:
                        bonus[i] = tok
                        repaired = True
                if not repaired:
                    s = int(qsl[i].item())
                    e = int(qsl[i + 1].item())
                    n_rej = int(num_rejected[i].item())
                    valid_e = e - n_rej
                    if valid_e > s:
                        tok = int(input_batch.input_ids[valid_e - 1].item())
                        bonus[i] = tok if 0 <= tok < vocab else 0
                    else:
                        bonus[i] = 0
            logger.warning(
                "Disagg-DFlash: repaired %d/%d OOB bonus tokens "
                "(vocab=%d); sample before=%s after=%s "
                "(248320==vocab_size is the top-k sentinel, not a token)",
                int(oob.sum().item()),
                num_reqs,
                vocab,
                before,
                bonus[oob][:8].detach().tolist(),
            )
        return bonus

    @torch.inference_mode()
    def propose_begin(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        context_only: bool = False,
    ) -> DisaggProposePending:
        """Fire projector + remote speculate; does not wait for draft tokens.

        Call ``propose_finish`` after overlapping verify-side post-sample work.
        """
        del (
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping[:num_reqs].clone()
        req_ids_list = list(input_batch.req_ids[:num_reqs])
        # context_only is Design A (P/D); mutually exclusive with this proxy.
        if dummy_run or is_profile or context_only:
            self.draft_tokens[:num_reqs].zero_()
            return DisaggProposePending(
                num_reqs=num_reqs,
                early_done=True,
                idx_mapping=idx_mapping,
                req_ids=req_ids_list,
            )

        req_ids = req_ids_list
        # Verify kernel warmup uses "_warmup_{i}_" IDs and historically only
        # finished them via execute_model (no sample_tokens). Never allocate
        # draft slots for those synthetic requests.
        if any(rid.startswith("_warmup_") for rid in req_ids):
            self.draft_tokens[:num_reqs].zero_()
            return DisaggProposePending(
                num_reqs=num_reqs,
                early_done=True,
                idx_mapping=idx_mapping,
                req_ids=req_ids_list,
            )

        num_tokens = input_batch.num_tokens
        profile = _profile_enabled() and self._tp_rank == 0
        if (
            self._proj_inflight
            or self._proj_reduced is not None
        ) and self._proj_num_tokens == num_tokens:
            reduced, t_proj0, t_proj1 = self._take_reduced(profile)
        else:
            if self._proj_inflight:
                # Stale async reduce (token count changed); wait & drop.
                self._proj_ready.synchronize()
                self._proj_inflight = False
            t_proj0 = time.perf_counter() if profile else 0.0
            reduced = reduce_aux_hidden_states(
                self._projector,
                last_hidden_states[:num_tokens],
                (
                    [h[:num_tokens] for h in aux_hidden_states]
                    if aux_hidden_states is not None
                    else None
                ),
            )
            if profile:
                torch.cuda.synchronize()
            t_proj1 = time.perf_counter() if profile else 0.0
        self._proj_reduced = None

        positions = input_batch.positions[:num_tokens]
        self._known_req_ids.update(req_ids)

        pending = DisaggProposePending(
            num_reqs=num_reqs,
            profile=profile,
            t_proj0=t_proj0,
            t_proj1=t_proj1,
            idx_mapping=idx_mapping,
            req_ids=req_ids_list,
        )

        if self._tp_rank == 0:
            assert self._client is not None
            idx = input_batch.idx_mapping[:num_reqs].long()
            bonus = self._resolve_bonus_tokens(
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                num_reqs,
            )

            t_pack0 = time.perf_counter() if profile else 0.0
            req = pack_speculate_request(
                req_ids=req_ids,
                reduced_hiddens=reduced,
                positions=positions,
                query_start_loc=input_batch.query_start_loc,
                num_rejected=num_rejected,
                # Force draft to take the last_sampled/bonus path.
                num_sampled=torch.ones(num_reqs, dtype=torch.int32, device=bonus.device),
                last_sampled=bonus,
                next_prefill_tokens=bonus,
                temperature=(
                    temperature.index_select(0, idx)
                    if temperature.numel() > num_reqs
                    else temperature[:num_reqs]
                ),
                seeds=(
                    seeds.index_select(0, idx)
                    if seeds.numel() > num_reqs
                    else seeds[:num_reqs]
                ),
                num_speculative_tokens=self.num_speculative_steps,
                num_reqs=num_reqs,
            )
            t_pack1 = time.perf_counter() if profile else 0.0
            self._client.speculate_begin(req)
            t_fire = time.perf_counter() if profile else 0.0
            pending.t_pack0 = t_pack0
            pending.t_pack1 = t_pack1
            pending.t_fire = t_fire
            if profile:
                pending.speculate_ms = {
                    "projector_ms": (t_proj1 - t_proj0) * 1000.0,
                    "pack_ms": (t_pack1 - t_pack0) * 1000.0,
                    "fire_ms": (t_fire - t_pack1) * 1000.0,
                }

        return pending

    @torch.inference_mode()
    def propose_finish(self, pending: DisaggProposePending) -> torch.Tensor:
        """Wait for in-flight draft tokens and broadcast to the TP group."""
        num_reqs = pending.num_reqs
        if pending.early_done:
            return self.draft_tokens[:num_reqs]

        profile = pending.profile
        speculate_ms = dict(pending.speculate_ms)
        t_wait0 = time.perf_counter() if profile else 0.0

        draft_tokens: torch.Tensor
        if self._tp_rank == 0:
            assert self._client is not None
            resp = self._client.speculate_wait()
            t_wait1 = time.perf_counter() if profile else 0.0
            draft_tokens = resp.draft_tokens.to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            if profile:
                speculate_ms["await_ms"] = (t_wait1 - t_wait0) * 1000.0
                if pending.t_fire > 0:
                    speculate_ms["verify_overlap_ms"] = (
                        t_wait0 - pending.t_fire
                    ) * 1000.0
                client_ms = getattr(self._client, "last_timings_ms", None)
                if isinstance(client_ms, dict):
                    speculate_ms.update(client_ms)
        else:
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )

        t_bcast0 = time.perf_counter() if profile else 0.0
        if self._tp_size > 1:
            torch.distributed.broadcast(
                draft_tokens,
                src=get_tp_group().ranks[0],
                group=get_tp_group().device_group,
            )
        if profile:
            torch.cuda.synchronize()
            t_bcast1 = time.perf_counter()
            speculate_ms["broadcast_ms"] = (t_bcast1 - t_bcast0) * 1000.0
            speculate_ms["propose_total_ms"] = (t_bcast1 - pending.t_proj0) * 1000.0
            self._profile_steps += 1
            if self._profile_steps % max(1, self._profile_every) == 0:
                logger.info(
                    "Disagg-DFlash profile (n=%d transport=%s cross_step=%s): %s",
                    num_reqs,
                    self._transport_name,
                    self._cross_step,
                    {k: round(v, 2) for k, v in speculate_ms.items()},
                )

        self.draft_tokens[:num_reqs].copy_(draft_tokens[:num_reqs])
        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        context_only: bool = False,
    ) -> torch.Tensor:
        pending = self.propose_begin(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            context_only=context_only,
        )
        return self.propose_finish(pending)
