#!/usr/bin/env python3
"""Line-oriented resident SparseKeyframePose worker used by pipeline SKFP stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RESPONSE_PREFIX = "__SKFP_RPC__ "


def now() -> float:
    return time.perf_counter()


def report_level() -> str:
    level = os.environ.get("SKFP_REPORT_LEVEL", "summary").strip().lower()
    return "debug" if level == "debug" else "summary"


def gpu_csv_enabled() -> bool:
    return os.environ.get("SKFP_ENABLE_GPU_CSV", "0").strip().lower() in {"1", "true", "yes", "on"}


def gpu_index_for_sampler() -> str | None:
    explicit = os.environ.get("SKFP_GPU_INDEX")
    if explicit:
        return explicit.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",", 1)[0].strip()
        return first or None
    return "0"


def _pick_timing(timings: dict[str, Any], key: str) -> float | None:
    value = timings.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize_stage_report(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "mode",
        "job_count",
        "planned_job_count",
        "worker_startup_s",
        "worker_total_s",
        "jobs_total_s",
        "stage_jobs_total_s",
        "stage_residual_s",
    ):
        if key in report:
            summary[key] = report[key]
    jobs: list[dict[str, Any]] = []
    for job in report.get("jobs", []) or []:
        if not isinstance(job, dict):
            continue
        timings = job.get("timings") if isinstance(job.get("timings"), dict) else {}
        item: dict[str, Any] = {
            "clip_uuid": job.get("clip_uuid"),
            "input_mode": job.get("input_mode"),
            "job_total_s": job.get("job_total_s"),
            "pipeline_run_s": job.get("pipeline_run_s") or _pick_timing(timings, "pipeline_run_s"),
            "artifact_convert_s": job.get("artifact_convert_s") or _pick_timing(timings, "artifact_convert_s"),
            "status": job.get("status") or ("error" if job.get("error") else "ok"),
        }
        if job.get("error"):
            item["error"] = job.get("error")
        jobs.append({key: value for key, value in item.items() if value is not None})
    summary["jobs"] = jobs
    return summary


def emit(payload: dict[str, Any]) -> None:
    print(RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


class ResidentSkfpWorker:
    def __init__(
        self,
        *,
        skfp_root: Path,
        vipe_python: str,
        vipe_adapter_script: Path,
        vipe_work_root: Path | None,
        stride: int,
        min_keyframes: int,
        max_keyframes: int,
    ) -> None:
        self.skfp_root = skfp_root.resolve()
        self.vipe_python = vipe_python
        self.vipe_adapter_script = vipe_adapter_script.resolve()
        self.vipe_work_root = vipe_work_root.resolve() if vipe_work_root is not None else None
        self.stride = stride
        self.min_keyframes = min_keyframes
        self.max_keyframes = max_keyframes
        self.pose_root = self.vipe_adapter_script.parents[2]
        self.vipe_repo = self.pose_root / "repo" / "vipe"
        self.resident_worker_script = self.pose_root / "scripts" / "vipe_resident_worker.py"
        self.report: dict[str, Any] = {
            "mode": "skfp_resident_rpc_worker",
            "skfp_root": str(self.skfp_root),
            "vipe_python": self.vipe_python,
            "vipe_adapter_script": str(self.vipe_adapter_script),
            "vipe_work_root": str(self.vipe_work_root) if self.vipe_work_root is not None else None,
            "vipe_repo": str(self.vipe_repo),
            "resident_worker_script": str(self.resident_worker_script),
            "stride": self.stride,
            "min_keyframes": self.min_keyframes,
            "max_keyframes": self.max_keyframes,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jobs": [],
        }

    def setup(self) -> None:
        t0 = now()
        sys.path.insert(0, str(self.skfp_root))
        import_t0 = now()
        from sparse_keyframe_pose.runner import run_sparse_keyframe_pose  # noqa: PLC0415

        self._run_sparse_keyframe_pose = run_sparse_keyframe_pose
        self.report["worker_import_s"] = now() - import_t0
        self.report["stage_setup_s"] = now() - t0

    def process_stage_jobs(self, request: dict[str, Any]) -> dict[str, Any]:
        jobs = request.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("stage_jobs request requires a non-empty jobs list")
        label = str(request.get("label") or "skfp")
        work_root = self.vipe_work_root or Path(tempfile.mkdtemp(prefix="skfp_vipe_stage_jobs_"))
        run_root = work_root / "stage_worker_runs" / f"{label}_{len(self.report['jobs']):06d}"
        jobs_jsonl = run_root / "jobs.jsonl"
        report_json = run_root / "report.json"
        level = report_level()
        gpu_csv = run_root / "gpu.csv"
        log_path = run_root / "stage_worker.log"
        run_root.mkdir(parents=True, exist_ok=True)
        with jobs_jsonl.open("w") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        cmd = [
            self.vipe_python,
            str(self.resident_worker_script),
            "stage-worker",
            "--jobs-jsonl",
            str(jobs_jsonl),
            "--report-json",
            str(report_json),
            "--vipe-repo",
            str(self.vipe_repo),
            "--report-level",
            level,
        ]
        if gpu_csv_enabled() or level == "debug":
            cmd.extend(["--gpu-samples-csv", str(gpu_csv)])
            gpu_index = gpu_index_for_sampler()
            if gpu_index is not None:
                cmd.extend(["--gpu-index", gpu_index])
        job_t0 = now()
        if level == "debug":
            with log_path.open("w") as log_f:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
        else:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            detail = f"; log={log_path}" if level == "debug" else "; set SKFP_REPORT_LEVEL=debug to capture stage_worker.log"
            raise RuntimeError(f"ViPE resident stage-worker failed with code {proc.returncode}{detail}")
        report = json.loads(report_json.read_text())
        report["stage_jobs_total_s"] = now() - job_t0
        if level == "debug":
            report["jobs_jsonl"] = str(jobs_jsonl)
            report["stage_worker_log"] = str(log_path)
            if gpu_csv.exists():
                report["gpu_samples_csv"] = str(gpu_csv)
        else:
            report = summarize_stage_report(report)
        self.report["jobs"].append({"clip_uuid": label, "stage_jobs": True, "job_total_s": report["stage_jobs_total_s"], "report": report})
        self.report["job_count"] = len(self.report["jobs"])
        return report

    def process_job(self, job: dict[str, Any]) -> dict[str, Any]:
        clip_dir = Path(job["clip_dir"]).resolve()
        out_root = Path(job["out_root"]).resolve()
        clip_uuid = str(job["clip_uuid"])
        output_dir = out_root / clip_uuid / "skfp"
        output_dir.mkdir(parents=True, exist_ok=True)
        job_t0 = now()
        run_kwargs: dict[str, Any] = {
            "video_path": clip_dir / "clip.mp4",
            "output_dir": output_dir,
            "stride": self.stride,
            "min_keyframes": self.min_keyframes,
            "max_keyframes": self.max_keyframes,
            "estimator_name": "vipe",
            "vipe_adapter_script": self.vipe_adapter_script,
            "vipe_work_root": self.vipe_work_root,
            "vipe_python_bin": self.vipe_python,
        }
        for key in ("num_frames", "fps", "width", "height"):
            if key in job and job[key] is not None:
                run_kwargs[key] = job[key]
        result = self._run_sparse_keyframe_pose(**run_kwargs)
        dense_dir = output_dir / "dense"
        pose_meta_path = dense_dir / "pose_meta.json"
        pose_meta = json.loads(pose_meta_path.read_text()) if pose_meta_path.exists() else {}
        job_report: dict[str, Any] = {
            "clip_uuid": clip_uuid,
            "clip_dir": str(clip_dir),
            "output_dir": str(output_dir),
            "dense_dir": str(dense_dir),
            "pose_status": getattr(result, "pose_status", "ok"),
            "pose_meta": pose_meta,
            "job_total_s": now() - job_t0,
        }
        self.report["jobs"].append(job_report)
        self.report["job_count"] = len(self.report["jobs"])
        return job_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skfp-root", type=Path, required=True)
    parser.add_argument("--vipe-python", required=True)
    parser.add_argument("--vipe-adapter-script", type=Path, required=True)
    parser.add_argument("--vipe-work-root", type=Path)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-keyframes", type=int, default=3)
    parser.add_argument("--max-keyframes", type=int, default=0)
    args = parser.parse_args()

    worker = ResidentSkfpWorker(
        skfp_root=args.skfp_root,
        vipe_python=args.vipe_python,
        vipe_adapter_script=args.vipe_adapter_script,
        vipe_work_root=args.vipe_work_root,
        stride=args.stride,
        min_keyframes=args.min_keyframes,
        max_keyframes=args.max_keyframes,
    )
    try:
        worker.setup()
        emit({"status": "ready", "report": worker.report})
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            request = json.loads(raw)
            if request.get("command") == "shutdown":
                emit({"status": "shutdown", "report": worker.report})
                return 0
            try:
                if request.get("command") == "stage_jobs":
                    report = worker.process_stage_jobs(request)
                    emit({"status": "ok", "report": report, "worker_report": worker.report})
                    continue
                report = worker.process_job(request)
                emit({"status": "ok", "job": report, "output_dir": report["output_dir"], "dense_dir": report["dense_dir"], "report": worker.report})
            except Exception as exc:  # noqa: BLE001
                emit({"status": "error", "error": repr(exc), "report": worker.report})
    except Exception as exc:  # noqa: BLE001
        emit({"status": "fatal", "error": repr(exc), "report": getattr(worker, "report", {})})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
