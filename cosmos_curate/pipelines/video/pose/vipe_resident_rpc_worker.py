#!/usr/bin/env python3
"""Line-oriented resident ViPE worker used by pipeline ViPE stages.

The parent process keeps this script alive inside the external ViPE Python
environment. Each JSON request on stdin runs one ViPE job and one JSON response
is written to stdout with a protocol prefix so ordinary ViPE logs can coexist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

RESPONSE_PREFIX = "__VIPE_RPC__ "
CURRENT_JOB_TIMINGS: dict[str, Any] | None = None


def now() -> float:
    return time.perf_counter()


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def add_timing(name: str, elapsed: float) -> None:
    if CURRENT_JOB_TIMINGS is None:
        return
    CURRENT_JOB_TIMINGS[name] = float(CURRENT_JOB_TIMINGS.get(name, 0.0)) + float(elapsed)
    CURRENT_JOB_TIMINGS[f"{name}_count"] = int(CURRENT_JOB_TIMINGS.get(f"{name}_count", 0)) + 1


def install_timing_patches(report: dict[str, Any]) -> None:
    import vipe.pipeline.processors as processors
    import vipe.slam.components.backend as backend_mod
    import vipe.slam.components.buffer as buffer_mod
    import vipe.slam.components.inner_filler as inner_filler_mod
    import vipe.slam.system as slam_system

    model_cache: dict[tuple[Any, ...], Any] = {}
    model_init = report.setdefault("model_init_s", {})

    orig_geocalib = processors.GeoCalib
    orig_droid = slam_system.DroidNet
    orig_depth = slam_system.make_depth_model
    orig_build_components = slam_system.SLAMSystem._build_components
    orig_slam_run = slam_system.SLAMSystem.run
    orig_backend_run = backend_mod.SLAMBackend.run
    orig_backend_run_if_needed = backend_mod.SLAMBackend.run_if_necessary
    orig_inner_compute = inner_filler_mod.InnerFiller.compute
    orig_ba = buffer_mod.GraphBuffer.bundle_adjustment

    def timed_model(key: tuple[Any, ...], label: str, factory):
        if key not in model_cache:
            sync_cuda()
            t0 = now()
            model_cache[key] = factory()
            sync_cuda()
            model_init[label] = float(model_init.get(label, 0.0)) + now() - t0
        return model_cache[key]

    def cached_geocalib(*args, **kwargs):
        label = f"geocalib:{kwargs.get('weights', args[0] if args else 'default')}"
        return timed_model(("geocalib", tuple(args), tuple(sorted(kwargs.items()))), label, lambda: orig_geocalib(*args, **kwargs))

    def cached_droid(*args, **kwargs):
        return timed_model(("droid", tuple(args), tuple(sorted(kwargs.items()))), "droidnet", lambda: orig_droid(*args, **kwargs))

    def cached_depth(*args, **kwargs):
        label = f"depth:{args[0] if args else kwargs.get('model_name', 'default')}"
        return timed_model(("depth", tuple(args), tuple(sorted(kwargs.items()))), label, lambda: orig_depth(*args, **kwargs))

    def timed_build_components(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_build_components(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("slam_build_components_s", now() - t0)

    def timed_slam_run(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_slam_run(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("slam_run_total_s", now() - t0)

    def timed_backend_run(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_backend_run(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("backend_run_s", now() - t0)

    def timed_backend_run_if_needed(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_backend_run_if_needed(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("backend_run_if_necessary_s", now() - t0)

    def timed_inner_compute(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_inner_compute(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("inner_filler_compute_s", now() - t0)

    def timed_ba(self, *args, **kwargs):
        sync_cuda()
        t0 = now()
        try:
            return orig_ba(self, *args, **kwargs)
        finally:
            sync_cuda()
            add_timing("bundle_adjustment_s", now() - t0)

    processors.GeoCalib = cached_geocalib
    slam_system.DroidNet = cached_droid
    slam_system.make_depth_model = cached_depth
    slam_system.SLAMSystem._build_components = timed_build_components
    slam_system.SLAMSystem.run = timed_slam_run
    backend_mod.SLAMBackend.run = timed_backend_run
    backend_mod.SLAMBackend.run_if_necessary = timed_backend_run_if_needed
    inner_filler_mod.InnerFiller.compute = timed_inner_compute
    buffer_mod.GraphBuffer.bundle_adjustment = timed_ba


def compose_vipe_config(vipe_repo: Path, clip_mp4: Path, output_path: Path):
    import hydra

    with hydra.initialize_config_dir(config_dir=str(vipe_repo / "configs"), version_base=None):
        return hydra.compose(
            config_name="default",
            overrides=[
                "pipeline=default",
                "streams=raw_mp4_stream",
                f"streams.base_path={clip_mp4}",
                f"pipeline.output.path={output_path}",
                "pipeline.output.save_artifacts=true",
                "pipeline.output.save_viz=false",
                "pipeline.post.depth_align_model=null",
            ],
        )


class ResidentVipeWorker:
    def __init__(self, *, adapter_root: Path, vipe_repo: Path) -> None:
        self.adapter_root = adapter_root.resolve()
        self.vipe_repo = vipe_repo.resolve()
        self.report: dict[str, Any] = {
            "mode": "pipeline_resident_rpc_worker",
            "adapter_root": str(self.adapter_root),
            "vipe_repo": str(self.vipe_repo),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jobs": [],
        }

    def setup(self) -> None:
        setup_t0 = now()
        sys.path.insert(0, str(self.adapter_root))
        sys.path.insert(0, str(self.vipe_repo))
        os.chdir(self.vipe_repo)

        import_t0 = now()
        import torch  # noqa: F401
        import vipe.pipeline.processors  # noqa: F401
        import vipe.slam.system  # noqa: F401

        self.report["worker_import_s"] = now() - import_t0

        patch_t0 = now()
        install_timing_patches(self.report)
        self.report["worker_patch_s"] = now() - patch_t0
        self.report["stage_setup_s"] = now() - setup_t0

    def process_job(self, job: dict[str, str]) -> dict[str, Any]:
        global CURRENT_JOB_TIMINGS

        from pose_adapter.adapters.vipe_adapter import _find_artifact, _select_rows_by_indices
        from pose_adapter.geometry import invert_se3_batch, normalize_max_translation, relative_to_first
        from pose_adapter.schema import UnifiedOutput, save

        clip_dir = Path(job["clip_dir"]).resolve()
        out_root = Path(job["out_root"]).resolve()
        clip_uuid = job["clip_uuid"]
        output_dir = out_root / clip_uuid / "vipe"
        output_dir.mkdir(parents=True, exist_ok=True)

        job_report: dict[str, Any] = {
            "clip_uuid": clip_uuid,
            "clip_dir": str(clip_dir),
            "output_dir": str(output_dir),
            "timings": {},
        }
        CURRENT_JOB_TIMINGS = job_report["timings"]
        job_t0 = now()
        try:
            with (clip_dir / "meta.json").open() as f:
                meta = json.load(f)
            wanted = np.load(clip_dir / "frame_indices.npy").astype(np.int32)
            timestamps = np.load(clip_dir / "timestamps.npy")
            orig_w, orig_h = meta["original_resolution"]

            with tempfile.TemporaryDirectory(prefix="vipe_resident_out_") as tmpdir:
                tmp = Path(tmpdir)

                t0 = now()
                cfg = compose_vipe_config(self.vipe_repo, clip_dir / "clip.mp4", tmp)
                job_report["timings"]["compose_config_s"] = now() - t0

                t0 = now()
                from vipe.pipeline import make_pipeline
                from vipe.streams.raw_mp4_stream import RawMp4Stream

                pipeline = make_pipeline(cfg.pipeline)
                job_report["timings"]["make_pipeline_s"] = now() - t0

                t0 = now()
                stream = RawMp4Stream(clip_dir / "clip.mp4")
                job_report["timings"]["raw_stream_init_s"] = now() - t0

                sync_cuda()
                t0 = now()
                pipeline.run(stream)
                sync_cuda()
                job_report["timings"]["pipeline_run_s"] = now() - t0

                pose_npz = _find_artifact(tmp, "pose", ".npz")
                intr_npz = _find_artifact(tmp, "intrinsics", ".npz")
                pose_data = np.load(pose_npz)
                intr_data = np.load(intr_npz)
                pose_sel, pose_fallback = _select_rows_by_indices(pose_data["data"], pose_data["inds"], wanted)
                intr_sel, intr_fallback = _select_rows_by_indices(intr_data["data"], intr_data["inds"], wanted)

            poses_c2w = pose_sel.astype(np.float64)
            poses_w2c = invert_se3_batch(poses_c2w)
            rel = relative_to_first(poses_c2w)
            rel_norm, scale_factor, total_path = normalize_max_translation(rel)

            warnings: list[str] = []
            if pose_fallback:
                warnings.append(f"vipe pose nearest-neighbour fallback at {len(pose_fallback)} positions")
            if intr_fallback:
                warnings.append(f"vipe intrinsics nearest-neighbour fallback at {len(intr_fallback)} positions")

            elapsed = now() - job_t0
            quality = {
                "method": "vipe",
                "status": "ok",
                "pose_convention": "c2w",
                "camera_coordinate": "opencv",
                "camera_model": "pinhole",
                "scale_status": "normalized",
                "scale_rule": "max_translation_norm",
                "scale_factor": scale_factor,
                "total_path_length_before_norm": total_path,
                "resolution_reference": [orig_w, orig_h],
                "intrinsics_shared_per_frame": False,
                "num_frames": int(poses_c2w.shape[0]),
                "frame_source_fps": meta["sampled_fps"],
                "original_fps": meta["original_fps"],
                "t_meaning": "sampled_frames",
                "elapsed_s": elapsed,
                "warnings": warnings,
                "runner": "resident_rpc_worker",
                "worker_timings": job_report["timings"],
                "worker_model_init_s": self.report.get("model_init_s", {}),
            }
            save(
                output_dir,
                UnifiedOutput(
                    intrinsics=intr_sel.astype(np.float32),
                    poses_c2w=poses_c2w,
                    poses_w2c=poses_w2c,
                    relative_poses=rel_norm,
                    scale_factor=scale_factor,
                    frame_indices=wanted,
                    timestamps=timestamps,
                    quality=quality,
                ),
            )
        finally:
            CURRENT_JOB_TIMINGS = None

        job_report["job_total_s"] = now() - job_t0
        self.report["jobs"].append(job_report)
        self.report["job_count"] = len(self.report["jobs"])
        return job_report


def emit(payload: dict[str, Any]) -> None:
    print(RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-script", type=Path, required=True)
    parser.add_argument("--vipe-repo", type=Path, required=True)
    args = parser.parse_args()

    worker = ResidentVipeWorker(
        adapter_root=args.adapter_script.resolve().parent.parent,
        vipe_repo=args.vipe_repo,
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
                report = worker.process_job(request)
                emit({"status": "ok", "job": report, "report": worker.report})
            except Exception as exc:  # noqa: BLE001
                emit({"status": "error", "error": repr(exc), "report": worker.report})
    except Exception as exc:  # noqa: BLE001
        emit({"status": "fatal", "error": repr(exc), "report": getattr(worker, "report", {})})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
