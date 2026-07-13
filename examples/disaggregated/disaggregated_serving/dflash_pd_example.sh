#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Design A: P/D disaggregation with DFlash speculative decoding.
#
# Prefill (producer) builds target KV + DFlash context KV, skips drafting.
# Decode (consumer) receives both KV groups via NixlConnector and resumes
# DFlash drafting without re-prefilling the drafter context.
#
# Qwen3.5-27B-DFlash has mixed sliding/full attention and requires the V2
# model runner. Hybrid Mamba models also need DS conv layout for NIXL.
#
# Usage:
#   export MODEL=Qwen/Qwen3.5-27B
#   export DRAFT=z-lab/Qwen3.5-27B-DFlash
#   export TP=4
#   ./dflash_pd_example.sh
#
# Then point a proxy at prefill:8100 and decode:8200, e.g.:
#   python3 examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py \
#     --model "$MODEL" --prefill localhost:8100 --decode localhost:8200 --port 8000

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-27B-DFlash}"
TP="${TP:-4}"
NUM_SPEC="${NUM_SPEC:-4}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"

SPEC_BASE="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":${NUM_SPEC}"
SPEC_PRODUCER="${SPEC_BASE},\"disagg_spec_role\":\"producer\"}"
SPEC_CONSUMER="${SPEC_BASE},\"disagg_spec_role\":\"consumer\"}"

cleanup() {
  echo "Stopping P/D DFlash servers..."
  [[ -n "${PREFILL_PID:-}" ]] && kill "${PREFILL_PID}" 2>/dev/null || true
  [[ -n "${DECODE_PID:-}" ]] && kill "${DECODE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Qwen3.5 / hybrid Mamba models require DS conv layout for NIXL transfer.
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
# Mixed sliding/full DFlash drafters require the V2 model runner.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

# Prefer producer/consumer roles (kv_both is deprecated).
KV_CFG_P="${KV_CFG_P:-{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\"}}"
KV_CFG_D="${KV_CFG_D:-{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\"}}"

echo "Starting prefill (producer) on :${PREFILL_PORT} ..."
CUDA_VISIBLE_DEVICES="${PREFILL_GPUS:-0,1,2,3}" \
VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-127.0.0.1}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_PORT:-5559}" \
  vllm serve "${MODEL}" \
    --host 127.0.0.1 \
    --tensor-parallel-size "${TP}" \
    --port "${PREFILL_PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --kv-transfer-config "${KV_CFG_P}" \
    --speculative-config "${SPEC_PRODUCER}" &
PREFILL_PID=$!

echo "Starting decode (consumer) on :${DECODE_PORT} ..."
CUDA_VISIBLE_DEVICES="${DECODE_GPUS:-4,5,6,7}" \
VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-127.0.0.1}" \
VLLM_NIXL_SIDE_CHANNEL_PORT="${DECODE_SIDE_PORT:-5659}" \
  vllm serve "${MODEL}" \
    --host 127.0.0.1 \
    --tensor-parallel-size "${TP}" \
    --port "${DECODE_PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --kv-transfer-config "${KV_CFG_D}" \
    --speculative-config "${SPEC_CONSUMER}" &
DECODE_PID=$!

echo "Prefill PID=${PREFILL_PID}, Decode PID=${DECODE_PID}"
echo "Waiting for servers (Ctrl-C to stop)..."
wait
