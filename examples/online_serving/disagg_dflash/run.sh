#!/usr/bin/env bash
# Example: Disagg-DFlash bring-up on one node (edit GPU ids / models as needed).
set -euo pipefail

DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-27B-DFlash}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-27B}"
NUM_SPEC="${NUM_SPEC:-7}"
BIND="${BIND:-tcp://127.0.0.1:50051}"
DRAFT_GPU="${DRAFT_GPU:-4}"
VERIFY_GPUS="${VERIFY_GPUS:-0,1,2,3}"
TP="${TP:-4}"
# zmq (default) or cuda_ipc for same-node GPU staging
TRANSPORT="${TRANSPORT:-zmq}"
IPC_MAX_TOKENS="${IPC_MAX_TOKENS:-16384}"

export VLLM_USE_V2_MODEL_RUNNER=1
export NCCL_NET="${NCCL_NET:-Socket}"

echo "Starting DFlash draft server on GPU ${DRAFT_GPU} (transport=${TRANSPORT}) ..."
CUDA_VISIBLE_DEVICES="${DRAFT_GPU}" python -m vllm.entrypoints.dflash_draft_server \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL}" \
  --num-speculative-tokens "${NUM_SPEC}" \
  --bind "${BIND}" \
  --transport "${TRANSPORT}" \
  --ipc-max-num-tokens "${IPC_MAX_TOKENS}" \
  --attention-backend FLASH_ATTN &
DRAFT_PID=$!
trap 'kill ${DRAFT_PID} 2>/dev/null || true' EXIT

# Wait for draft warmup (compile).
for i in $(seq 1 180); do
  if ! kill -0 "${DRAFT_PID}" 2>/dev/null; then
    echo "Draft server died."
    exit 1
  fi
  # Rough wait — prefer launch_disagg_dflash_servers.sh which greps logs.
  sleep 2
  if (( i >= 15 )); then
    break
  fi
done

SPEC_JSON="{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC},\"disagg_dflash_address\":\"${BIND}\",\"disagg_dflash_timeout_ms\":180000,\"disagg_dflash_transport\":\"${TRANSPORT}\",\"attention_backend\":\"FLASH_ATTN\"}"
if [[ "${TRANSPORT}" == "cuda_ipc" ]]; then
  SPEC_JSON="{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${NUM_SPEC},\"disagg_dflash_address\":\"${BIND}\",\"disagg_dflash_timeout_ms\":180000,\"disagg_dflash_transport\":\"cuda_ipc\",\"disagg_dflash_ipc_max_num_tokens\":${IPC_MAX_TOKENS},\"attention_backend\":\"FLASH_ATTN\"}"
fi

echo "Starting verify server on GPUs ${VERIFY_GPUS} ..."
CUDA_VISIBLE_DEVICES="${VERIFY_GPUS}" vllm serve "${TARGET_MODEL}" \
  --tensor-parallel-size "${TP}" \
  --speculative-config "${SPEC_JSON}"
