#!/usr/bin/env python3
"""Run Cosmos to save OpenAICaptionStage inputs into the project data/inputs directory."""

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib import error, request


DEFAULT_COSMOS_URL = "http://localhost:8000/v1/run_pipeline"
DEFAULT_INPUT_VIDEO_PATH = "/mlp-01/lihaoyue/caption/data/samples"
DEFAULT_MOTION_FILTER = "disable"
DEFAULT_MODEL_NAME = "Qwen3.6-27B"
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 20
DEFAULT_EXECUTION_MODE = "STREAMING"
DEFAULT_STAGE_NAME = "OpenAICaptionStage"
DEFAULT_API_CAPTION_NUM_WORKERS_PER_NODE = 1
DEFAULT_LIMIT = 5


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the Cosmos pipeline path needed to save OpenAICaptionStage inputs, "
            "then copy the saved tasks into the project data/inputs directory."
        )
    )
    parser.add_argument("--cosmos-url", default=DEFAULT_COSMOS_URL, help="Cosmos API endpoint.")
    parser.add_argument("--input-video-path", default=DEFAULT_INPUT_VIDEO_PATH, help="Raw video root for Cosmos.")
    parser.add_argument(
        "--run-name",
        default=datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S"),
        help="Logical run name used under data/outputs and data/inputs.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of raw videos to process.")
    parser.add_argument(
        "--motion-filter",
        default=DEFAULT_MOTION_FILTER,
        choices=["disable", "score-only", "enable"],
        help="Motion filter mode for the pipeline.",
    )
    parser.add_argument("--openai-model-name", default=DEFAULT_MODEL_NAME, help="Served model name for vLLM.")
    parser.add_argument("--openai-caption-retries", type=int, default=DEFAULT_RETRIES, help="Caption retry count.")
    parser.add_argument(
        "--openai-retry-delay-seconds",
        type=int,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="Delay between caption retries.",
    )
    parser.add_argument(
        "--reqid",
        default=None,
        help="Optional NVCF-REQID header value. Defaults to cosmos-caption-<run-name>.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Caption project root that contains data/inputs and data/outputs.",
    )
    parser.add_argument(
        "--stage-name",
        default=DEFAULT_STAGE_NAME,
        help="Saved stage directory name to copy from outputs/tasks into inputs/tasks.",
    )
    parser.add_argument(
        "--execution-mode",
        default=DEFAULT_EXECUTION_MODE,
        choices=["AUTO", "BATCH", "STREAMING"],
        help="Cosmos execution mode.",
    )
    parser.add_argument(
        "--api-caption-batch-size",
        type=int,
        default=1,
        help="OpenAI-compatible caption stage batch size / concurrency.",
    )
    parser.add_argument(
        "--api-caption-num-workers-per-node",
        type=int,
        default=DEFAULT_API_CAPTION_NUM_WORKERS_PER_NODE,
        help="Number of OpenAI caption workers per node.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the full Cosmos JSON response body on success.",
    )
    return parser.parse_args()


def build_payload(args: argparse.Namespace, output_clip_path: Path) -> dict[str, object]:
    return {
        "pipeline": "split",
        "args": {
            "input_video_path": args.input_video_path,
            "output_clip_path": str(output_clip_path),
            "limit": args.limit,
            "execution_mode": args.execution_mode,
            "splitting_algorithm": "transnetv2",
            "transnetv2_frame_decoder_mode": "ffmpeg_cpu",
            "transnetv2_min_length_s": 5,
            "transnetv2_min_length_frames": 81,
            "transnetv2_max_length_s": 65,
            "transnetv2_max_length_mode": "stride",
            "transnetv2_crop_s": 0.5,
            "frame_number": 81,
            "transcode_target_fps": 16,
            "transcode_target_width": 640,
            "transcode_target_height": 352,
            "generate_embeddings": False,
            "generate_previews": False,
            "generate_captions": True,
            "motion_filter": args.motion_filter,
            "motion_global_mean_threshold": 0.00080,
            "motion_per_patch_min_256_threshold": 1e-06,
            "aesthetic_threshold": 4.0,
            "artificial_text_filter": "disable",
            "vlm_filter": "disable",
            "vlm_filter_endpoint": "openai",
            "vlm_filter_openai_model_name": "auto",
            "vlm_filter_openai_retries": 2,
            "vlm_filter_openai_retry_delay_seconds": 20,
            "vlm_filter_prompt_variant": "text-contamination",
            "vlm_filter_categories": "post-production text",
            "qwen_video_classifier": "disable",
            "camera_character_coupling_filter": "disable",
            "captioning_algorithm": "openai",
            "openai_model_name": args.openai_model_name,
            "openai_caption_retries": args.openai_caption_retries,
            "openai_retry_delay_seconds": args.openai_retry_delay_seconds,
            "stage_save": [args.stage_name],
            "stage_save_sample_rate": 1.0,
            "api_caption_batch_size": args.api_caption_batch_size,
            "api_caption_num_workers_per_node": args.api_caption_num_workers_per_node,
        },
    }


def post_json(url: str, reqid: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "NVCF-REQID": reqid,
        },
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cosmos API returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach Cosmos API at {url}: {exc}") from exc


def copy_stage_inputs(stage_output_dir: Path, stage_input_dir: Path) -> None:
    if not stage_output_dir.exists():
        raise FileNotFoundError(f"Saved stage input directory not found: {stage_output_dir}")
    stage_input_dir.parent.mkdir(parents=True, exist_ok=True)
    if stage_input_dir.exists():
        shutil.rmtree(stage_input_dir)
    shutil.copytree(stage_output_dir, stage_input_dir)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    outputs_root = project_root / "data" / "outputs"
    inputs_root = project_root / "data" / "inputs"

    run_output_dir = outputs_root / args.run_name
    run_input_stage_dir = inputs_root / args.run_name / "tasks" / args.stage_name
    saved_stage_dir = run_output_dir / "tasks" / args.stage_name
    reqid = args.reqid or f"cosmos-caption-{args.run_name}"

    run_output_dir.parent.mkdir(parents=True, exist_ok=True)
    inputs_root.mkdir(parents=True, exist_ok=True)

    payload = build_payload(args, run_output_dir)
    print(f"Running Cosmos pipeline with reqid={reqid}")
    print(f"Output directory: {run_output_dir}")
    response = post_json(args.cosmos_url, reqid, payload)

    copy_stage_inputs(saved_stage_dir, run_input_stage_dir)

    print(f"Copied saved caption-stage inputs to: {run_input_stage_dir}")
    if args.print_response:
        print(json.dumps(response, indent=2, ensure_ascii=True))
    else:
        print("Cosmos pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
