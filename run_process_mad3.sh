#!/usr/bin/env bash
set -euo pipefail

# ==============================
# MoonCast MAD3 batch TTS on 4 GPUs
# - Runs process_mad3.py in N shards (GPU per shard)
# - Uses CUDA_VISIBLE_DEVICES to pin one GPU per process
# - Logs per shard
# ==============================

# ---- user knobs ----
PY="python"
SCRIPT="process_mad3.py"

# Where your JSONs are
INPUT_DIR="input_data/Output_STEP1+2_ENGLISH_TTS"
# Where you want wav outputs
OUTPUT_DIR="output_data/ENGLISH_mooncast_Gem"
# Where manifest + wav voices live
MANIFEST="voices/manifest.json"

# GPUs to use (edit as needed)
GPUS=(3 5 6 7)
MAX_PARALLEL=${#GPUS[@]}

# Logs
LOGDIR="logs_process_mad3"
mkdir -p "${LOGDIR}"
mkdir -p "${OUTPUT_DIR}"

SEED=1234

# ------------------------------
# New speed/quality knobs (edit here)
# ------------------------------

# WAV padding at start/end
PAD_MS=250

# Dynamic max_new_tokens heuristic
MIN_NEW_TOKENS=150
MAX_NEW_TOKENS_CAP=10000
WORDS_PER_SEC=2.5
TOKENS_PER_SEC=60.0
SAFETY=1.8

# Retry-on-truncation behavior
MAX_RETRIES=1
RETRY_MULT=2.0
RETRY_ADD=200
RETRY_FULL_CAP_ON_LAST=false   # set to true if you want last retry to jump to CAP

# Dtype (auto/bf16/fp16). bf16 is good on A100/H100; fp16 on older GPUs.
DTYPE="auto"

submit() {
  local shard_id="$1"
  local num_shards="$2"
  local gpu="$3"

  local name="shard${shard_id}_of_${num_shards}_gpu${gpu}"
  local log="${LOGDIR}/${name}.log"

  echo "[LAUNCH] ${name}"

  # Build optional flag
  local EXTRA_RETRY_FLAG=()
  if [ "${RETRY_FULL_CAP_ON_LAST}" = "true" ]; then
    EXTRA_RETRY_FLAG+=(--retry_full_cap_on_last)
  fi

  nohup env CUDA_VISIBLE_DEVICES="${gpu}" TOKENIZERS_PARALLELISM=false \
    ${PY} "${SCRIPT}" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --manifest "${MANIFEST}" \
      --shard_id "${shard_id}" \
      --num_shards "${num_shards}" \
      --seed "${SEED}" \
      --skip_existing \
      --pad_ms "${PAD_MS}" \
      --min_new_tokens "${MIN_NEW_TOKENS}" \
      --max_new_tokens_cap "${MAX_NEW_TOKENS_CAP}" \
      --words_per_sec "${WORDS_PER_SEC}" \
      --tokens_per_sec "${TOKENS_PER_SEC}" \
      --safety "${SAFETY}" \
      --max_retries "${MAX_RETRIES}" \
      --retry_mult "${RETRY_MULT}" \
      --retry_add "${RETRY_ADD}" \
      --dtype "${DTYPE}" \
      "${EXTRA_RETRY_FLAG[@]}" \
      > "${log}" 2>&1 &

  # concurrency guard
  while [ "$(jobs -r -p | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    sleep 3
  done
}

# Launch N shards
NUM_SHARDS=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  submit "${i}" "${NUM_SHARDS}" "${GPUS[$i]}"
done

wait
echo "All jobs finished. Logs in: ${LOGDIR}"
echo "Outputs in: ${OUTPUT_DIR}"

# Usage:
#   chmod +x run_process_mad3_4gpu.sh
#   nohup ./run_process_mad3_4gpu.sh > master_process_mad3.out 2>&1 &
