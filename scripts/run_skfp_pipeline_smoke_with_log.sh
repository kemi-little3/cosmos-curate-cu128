#!/usr/bin/env bash
set -euo pipefail

DATA_PIPELINE_ROOT=${DATA_PIPELINE_ROOT:-/mlp-01/duanmengxuan/data_pipeline}
COSMOS_ROOT=${COSMOS_ROOT:-${DATA_PIPELINE_ROOT}/repo/cosmos-curate-cu128}
LOG_DIR=${LOG_DIR:-${DATA_PIPELINE_ROOT}/tmp/skfp_pipeline_logs}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-${LOG_DIR}/skfp_resident_window_smoke_${TIMESTAMP}.log}

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${COSMOS_ROOT}"

echo "[skfp-smoke] started_at=$(date -Is)"
echo "[skfp-smoke] log_file=${LOG_FILE}"
echo "[skfp-smoke] cosmos_root=${COSMOS_ROOT}"
echo "[skfp-smoke] data_pipeline_root=${DATA_PIPELINE_ROOT}"

export RAY_ADDRESS=${RAY_ADDRESS:-192.168.8.176:6379}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VIPE_CMD=${VIPE_CMD:-${DATA_PIPELINE_ROOT}/vipe/envs/vipe/bin/vipe}
export VIPE_REPO=${VIPE_REPO:-${DATA_PIPELINE_ROOT}/vipe/pose_estimation/repo/vipe}
export TORCH_HOME=${TORCH_HOME:-${DATA_PIPELINE_ROOT}/vipe/models/torch}
export HF_HOME=${HF_HOME:-${DATA_PIPELINE_ROOT}/vipe/models/huggingface}
export GENERATE_T5_EMBEDDINGS=${GENERATE_T5_EMBEDDINGS:-0}

echo "[skfp-smoke] RAY_ADDRESS=${RAY_ADDRESS}"
echo "[skfp-smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[skfp-smoke] VIPE_CMD=${VIPE_CMD}"
echo "[skfp-smoke] VIPE_REPO=${VIPE_REPO}"
echo "[skfp-smoke] input_video_path=${SKFP_SMOKE_INPUT_VIDEO_PATH:-${DATA_PIPELINE_ROOT}/input}"
echo "[skfp-smoke] input_video_list_json_path=${SKFP_SMOKE_INPUT_VIDEO_LIST_JSON_PATH:-${DATA_PIPELINE_ROOT}/input/skfp_smoke_input_video_list.json}"
echo "[skfp-smoke] output_clip_path=${SKFP_SMOKE_OUTPUT_CLIP_PATH:-${DATA_PIPELINE_ROOT}/tmp/skfp_pipeline_smoke_resident_window}"
echo "[skfp-smoke] skfp_gpus_per_worker=${SKFP_SMOKE_GPUS_PER_WORKER:-0.75}"

echo "[skfp-smoke] ray preflight begin"
pixi run --as-is python -c "import os, ray; print('preflight RAY_ADDRESS=', os.environ.get('RAY_ADDRESS')); ray.init(address=os.environ['RAY_ADDRESS']); print('preflight cluster=', ray.cluster_resources()); print('preflight available=', ray.available_resources())"
echo "[skfp-smoke] ray preflight end"

pixi run --as-is python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path "${SKFP_SMOKE_INPUT_VIDEO_PATH:-${DATA_PIPELINE_ROOT}/input}" \
  --input-video-list-json-path "${SKFP_SMOKE_INPUT_VIDEO_LIST_JSON_PATH:-${DATA_PIPELINE_ROOT}/input/skfp_smoke_input_video_list.json}" \
  --output-clip-path "${SKFP_SMOKE_OUTPUT_CLIP_PATH:-${DATA_PIPELINE_ROOT}/tmp/skfp_pipeline_smoke_resident_window}" \
  --model-weights-path "${SKFP_SMOKE_MODEL_WEIGHTS_PATH:-/mlp-01/models}" \
  --limit "${SKFP_SMOKE_LIMIT:-1}" \
  --limit-clips "${SKFP_SMOKE_LIMIT_CLIPS:-1}" \
  --execution-mode STREAMING \
  --motion-filter disable \
  --aesthetic-threshold 0 \
  --artificial-text-filter disable \
  --vlm-filter disable \
  --qwen-video-classifier disable \
  --camera-character-coupling-filter disable \
  --no-generate-embeddings \
  --no-generate-captions \
  --enable-skfp-pose \
  --skfp-run-mode resident-window \
  --skfp-root "${SKFP_SMOKE_ROOT:-${DATA_PIPELINE_ROOT}/SparseKeyframePose}" \
  --skfp-stride "${SKFP_SMOKE_STRIDE:-32}" \
  --skfp-min-keyframes "${SKFP_SMOKE_MIN_KEYFRAMES:-3}" \
  --skfp-max-keyframes "${SKFP_SMOKE_MAX_KEYFRAMES:-64}" \
  --skfp-vipe-python "${SKFP_SMOKE_VIPE_PYTHON:-${DATA_PIPELINE_ROOT}/vipe/envs/vipe/bin/python}" \
  --skfp-vipe-adapter-script "${SKFP_SMOKE_VIPE_ADAPTER_SCRIPT:-${DATA_PIPELINE_ROOT}/vipe/pose_estimation/adapter/scripts/run_adapter.py}" \
  --skfp-vipe-work-root "${SKFP_SMOKE_VIPE_WORK_ROOT:-${DATA_PIPELINE_ROOT}/tmp/skfp_pipeline_vipe_runs}" \
  --skfp-fail-policy "${SKFP_SMOKE_FAIL_POLICY:-warn-only}" \
  --skfp-gpus-per-worker "${SKFP_SMOKE_GPUS_PER_WORKER:-0.75}" \
  --skfp-num-workers "${SKFP_SMOKE_NUM_WORKERS:-1}" \
  --generate-cosmos-predict-dataset predict2 \
  --frame-number "${SKFP_SMOKE_FRAME_NUMBER:-81}" \
  --transnetv2-max-length-s "${SKFP_SMOKE_TRANSNETV2_MAX_LENGTH_S:-65}" \
  --transnetv2-frame-decoder-mode ffmpeg_cpu \
  --transnetv2-gpus-per-worker "${SKFP_SMOKE_TRANSNETV2_GPUS_PER_WORKER:-0}" \
  --transcode-target-fps "${SKFP_SMOKE_TRANSCODE_TARGET_FPS:-16}" \
  --transcode-target-width "${SKFP_SMOKE_TRANSCODE_TARGET_WIDTH:-640}" \
  --transcode-target-height "${SKFP_SMOKE_TRANSCODE_TARGET_HEIGHT:-352}" \
  --num-download-workers-per-node "${SKFP_SMOKE_NUM_DOWNLOAD_WORKERS_PER_NODE:-1}" \
  --num-clip-writer-workers-per-node "${SKFP_SMOKE_NUM_CLIP_WRITER_WORKERS_PER_NODE:-1}" \
  --vllm-prepare-num-cpus-per-worker "${SKFP_SMOKE_VLLM_PREPARE_NUM_CPUS_PER_WORKER:-1}"

echo "[skfp-smoke] finished_at=$(date -Is)"
echo "[skfp-smoke] log_file=${LOG_FILE}"
