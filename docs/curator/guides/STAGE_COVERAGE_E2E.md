# Split E2E Stage Coverage Run

This note records the local split pipeline command used to exercise the
filtering, classification, captioning, embedding, pose, and Cosmos Predict2
dataset writing stages.

## Scope

Enabled:

- Motion scoring: `--motion-filter score-only`
- Aesthetic scoring: `--aesthetic-threshold 0.0`
- Artificial text filtering: `--artificial-text-filter enable`
- Qwen semantic filtering in score-only mode: `--qwen-filter score-only`
- Qwen video classification: `--qwen-video-classifier enable`
- VLM captioning
- InternVideo2 embeddings
- ViPE pose estimation: `--enable-vipe-pose`
- Cosmos Predict2 dataset output: `--generate-cosmos-predict-dataset predict2`

Disabled or intentionally omitted:

- First-person POV filter
- Dedup
- Sharding
- Super-resolution
- SAM

## ViPE Environment

The ViPE process runs with the external ViPE conda environment:

```bash
/home/hexuming/miniconda3/envs/vipe/bin/python
```

The model caches must be exposed to the container and exported before running
the pipeline:

```bash
TORCH_HOME=/data1/hexuming/pose_estimation/ViPE_models/torch
HF_HOME=/data1/hexuming/pose_estimation/ViPE_models/huggingface
TOKENIZERS_PARALLELISM=false
```

The command below uses the derived `cosmos-curate-vipe:slim` image built from
`docker/vipe-runtime.Dockerfile`. It keeps the original `cosmos-curate:slim`
image unchanged and only adds the GLib/GL/X11 runtime libraries needed by
OpenCV in the external ViPE conda environment.

## Command

```bash
source ~/.zshrc && conda activate cosmos-curate && cosmos-curate local launch \
  --image-name cosmos-curate-vipe \
  --image-tag slim \
  --curator-path . \
  --pixi-path . \
  --extra-volumes /home/hexuming/miniconda3/envs/vipe:/home/hexuming/miniconda3/envs/vipe,/data1/hexuming/pose_estimation:/data1/hexuming/pose_estimation \
  -- env \
    TORCH_HOME=/data1/hexuming/pose_estimation/ViPE_models/torch \
    HF_HOME=/data1/hexuming/pose_estimation/ViPE_models/huggingface \
    TOKENIZERS_PARALLELISM=false \
    pixi run --as-is python -m cosmos_curate.pipelines.video.run_pipeline split \
      --input-video-path /config/raw_videos \
      --output-clip-path /config/output_clips_stage_coverage_no_pov_pose_cache \
      --limit 1 \
      --motion-filter score-only \
      --aesthetic-threshold 0.0 \
      --artificial-text-filter enable \
      --qwen-filter score-only \
      --qwen-video-classifier enable \
      --enable-vipe-pose \
      --vipe-python /home/hexuming/miniconda3/envs/vipe/bin/python \
      --generate-cosmos-predict-dataset predict2
```

## Verified Result

Output path:

```bash
/data1/hexuming/data_pipeline/cosmos_curate_local_workspace/output_clips_stage_coverage_no_pov_pose_cache
```

Summary:

- `num_input_videos`: 1
- `num_processed_videos`: 1
- `total_num_clips_transcoded`: 4
- `total_num_clips_filtered_by_artificial_text`: 3
- `total_num_clips_passed`: 1
- `total_num_clips_with_caption`: 1
- `total_windows_with_pose`: 3
- `total_windows_pose_failed`: 0
- `total_num_clips_filtered_by_qwen_first_person_pov`: 0
- `pipeline_run_time`: 16.34 minutes

Pose files were written under:

```bash
cosmos_predict2_video2world_dataset/poses/
```

Each processed window contains:

- `intrinsics.npy`
- `poses.npy`
- `relative_poses.npy`
- `pose_meta.json`
