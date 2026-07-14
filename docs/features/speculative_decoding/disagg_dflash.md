# Disaggregated DFlash (experimental)

Offload the DFlash draft model to a **separate process / GPU** so verify
(target) GPUs no longer co-locate draft compute.

When `disagg_dflash_address` is **unset** (default), colocated DFlash is
**unchanged**. This path is orthogonal to Design A `disagg_spec_role` (P/D).

## Design

| Side | Role |
| --- | --- |
| Verify | Target model + rejection sampling. Loads only the small draft `fc` projector (`L·H → H`) and RPCs to the draft server. |
| Draft | Full DFlash weights + target `embed_tokens` / `lm_head`. Owns draft KV. |

Each speculate round sends **only this step's scheduled tokens** (already
incremental vs full prefix), as **reduced** `[N, H]` hiddens — not `L·H`.

### Transports

| Transport | Data plane | Notes |
| --- | --- | --- |
| `zmq` (default) | Multipart ZMQ frames for all tensors | Works cross-node; simplest bring-up |
| `cuda_ipc` | CUDA IPC staging for `context_hiddens` / `draft_tokens`; ZMQ for control + small meta tensors | Same host only; TP0 ↔ draft GPU need P2P/NVLink (or PCIe P2P) |
| `nccl` / `nixl` | — | Reserved stubs (not implemented) |

## Run

### 1. Start the draft server (dedicated GPU)

```bash
# Example: draft on physical GPU 4 (ZMQ data plane)
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.dflash_draft_server \
  --draft-model z-lab/Qwen3.5-27B-DFlash \
  --target-model Qwen/Qwen3.5-27B \
  --num-speculative-tokens 7 \
  --bind tcp://0.0.0.0:50051 \
  --attention-backend FLASH_ATTN

# Same-node CUDA IPC data plane
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.dflash_draft_server \
  --draft-model z-lab/Qwen3.5-27B-DFlash \
  --target-model Qwen/Qwen3.5-27B \
  --num-speculative-tokens 7 \
  --bind tcp://0.0.0.0:50051 \
  --transport cuda_ipc \
  --ipc-max-num-tokens 16384 \
  --attention-backend FLASH_ATTN
```

### 2. Start the verify / serve process (target GPUs)

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "dflash",
    "model": "z-lab/Qwen3.5-27B-DFlash",
    "num_speculative_tokens": 7,
    "disagg_dflash_address": "tcp://127.0.0.1:50051",
    "disagg_dflash_timeout_ms": 180000,
    "disagg_dflash_transport": "cuda_ipc",
    "attention_backend": "FLASH_ATTN"
  }'
```

Bring-up helper: `specdec/launch_disagg_dflash_servers.sh`.

### 3. Colocated control (no disagg)

Omit `disagg_dflash_address` — same as stock DFlash.

## Config knobs

| Key | Default | Meaning |
| --- | --- | --- |
| `disagg_dflash_address` | `None` | ZMQ address of draft server; enables disagg |
| `disagg_dflash_timeout_ms` | `180000` | RPC timeout (first compile can be slow) |
| `disagg_dflash_transport` | `zmq` | `zmq` \| `cuda_ipc` (stubs: `nccl`, `nixl`) |
| `disagg_dflash_ipc_max_num_tokens` | `None` | Preferred IPC staging size; draft raises `--ipc-max-num-tokens` if HELLO asks for more |

## Notes

- Qwen3.5-DFlash requires Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`).
- Draft and verify `num_speculative_tokens` must match.
- Do not set `disagg_spec_role` together with `disagg_dflash_address`.
- `cuda_ipc` requires the same host; draft `--transport` and verify
  `disagg_dflash_transport` must both be `cuda_ipc`.
- Same-node TCP is fine for the control plane.
- Profiling (verify TP0): `VLLM_DISAGG_DFLASH_PROFILE=1` logs projector /
  IPC / ZMQ RTT / broadcast ms every `VLLM_DISAGG_DFLASH_PROFILE_EVERY`
  steps (default 50).
- Experimental; validate acceptance length vs colocated DFlash before production use.
