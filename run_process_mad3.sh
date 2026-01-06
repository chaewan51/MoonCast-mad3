#!/usr/bin/env bash
set -euo pipefail

# ==============================
# MoonCast MAD3 batch TTS on 4 GPUs
# - Runs process_mad3.py in 4 shards (GPU per shard)
# - Uses CUDA_VISIBLE_DEVICES to pin one GPU per process
# - Simple concurrency guard + logs per shard
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

# ------------------------------
# Assumptions about process_mad3.py:
# It supports these CLI args (you add them if missing):
#   --input_dir, --output_dir, --manifest
#   --shard_id, --num_shards
#   --seed (optional)
#   --skip_existing (optional)
# ------------------------------

SEED=1234

submit() {
  local shard_id="$1"
  local num_shards="$2"
  local gpu="$3"

  local name="shard${shard_id}_of_${num_shards}_gpu${gpu}"
  local log="${LOGDIR}/${name}.log"

  echo "[LAUNCH] ${name}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu}" TOKENIZERS_PARALLELISM=false \
    ${PY} "${SCRIPT}" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --manifest "${MANIFEST}" \
      --shard_id "${shard_id}" \
      --num_shards "${num_shards}" \
      --seed "${SEED}" \
      --skip_existing \
      > "${log}" 2>&1 &

  # concurrency guard
  while [ "$(jobs -r -p | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    sleep 3
  done
}

# Launch 4 shards
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
