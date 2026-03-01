"""Alpamayo VLA inference workflow for HLX.

Runs the Alpamayo-R1-10B model inference on video data and produces
trajectory predictions with reasoning traces.
"""
import logging
import os
from pathlib import Path

from hlx.wf import DedicatedNode, Node, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

from config_alpamayo import (
    ALPAMAYO_CKPT_PREFIX,
    BASE_WORKDIR,
    CKPT_FUSE_MOUNT_NAME,
    CKPT_FUSE_MOUNT_ROOT,
    CKPT_LOCAL_PATH,
    COMPUTE_NODE,
    CONTAINER_IMAGE,
    SCRATCH_OUTPUT_BASE,
    VLA_OUTPUT_PREFIX,
    VP_BUCKET,
    VIDEO_DATA_FUSE_MOUNT_NAME,
    VIDEO_DATA_GCS_PREFIX,
)
from helpers_alpamayo import (
    VLAVideoMetrics,
    ensure_symlink,
    _load_model,
    _run_alpamayo_inference,
    _stage_video_data,
    _upload_outputs,
    _write_report,
)

logger = logging.getLogger(__name__)


@task(
    compute=DedicatedNode(
        node=COMPUTE_NODE,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/root/.cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    },
    mounts=[
        FuseBucket(
            bucket=VP_BUCKET,
            name=CKPT_FUSE_MOUNT_NAME,
            prefix=ALPAMAYO_CKPT_PREFIX,
        ),
        FuseBucket(
            bucket=VP_BUCKET,
            name=VIDEO_DATA_FUSE_MOUNT_NAME,
            prefix=VIDEO_DATA_GCS_PREFIX,
        ),
    ],
)
def run_alpamayo_inference_task(
    video_data_gcs_path: str,
    output_run_id: str,
    model_id: str = "nvidia/Alpamayo-R1-10B",
    num_traj_samples: int = 1,
    video_name: str = "auto",
    black_non_target_cameras: bool = True,
) -> dict:
    """
    Run Alpamayo VLA inference on video data.
    
    Args:
        video_data_gcs_path: GCS path to video data (e.g., gs://bucket/path/to/videos)
        output_run_id: Unique identifier for this run
        model_id: HuggingFace model ID or local path
        num_traj_samples: Number of trajectory samples per video
        video_name: Filter to a specific video by stem name ("auto" = all videos)
        black_non_target_cameras: If True, replace all non-target camera frames with
            black (zero) frames so the model prediction is influenced only by the
            generated front camera video.
    
    Returns:
        Dictionary with output GCS path and metrics
    """
    logger.info("=" * 80)
    logger.info("ALPAMAYO VLA INFERENCE TASK")
    logger.info("=" * 80)
    logger.info(f"Video Data Source: {video_data_gcs_path}")
    logger.info(f"Output Run ID: {output_run_id}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Video Name Filter: {video_name}")
    logger.info(f"Black Non-Target Cameras: {black_non_target_cameras}")
    logger.info(f"Container Image: {CONTAINER_IMAGE}")
    
    # Setup checkpoint symlink
    logger.info(f"Setting up checkpoint symlink from {CKPT_FUSE_MOUNT_ROOT} to {CKPT_LOCAL_PATH}")
    ensure_symlink(CKPT_FUSE_MOUNT_ROOT, CKPT_LOCAL_PATH)
    
    # Discover video files via FuseBucket mount
    video_tuples = _stage_video_data(video_data_gcs_path, video_name=video_name)
    
    if not video_tuples:
        raise ValueError(f"No videos found in {video_data_gcs_path}")
    
    logger.info(f"Processing {len(video_tuples)} videos")
    
    # Load model ONCE for all videos
    model, processor, helper_mod, device = _load_model(model_id)
    
    # Create output directory
    local_output_dir = os.path.join(SCRATCH_OUTPUT_BASE, output_run_id)
    Path(local_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run inference on each video (in-process, reusing loaded model)
    all_metrics = []
    for i, (video_path, prompt_dir) in enumerate(video_tuples, 1):
        logger.info(f"Processing video {i}/{len(video_tuples)}: {video_path} (prompt_dir={prompt_dir!r})")
        
        # Each video gets its own subdirectory:
        #   <run_id>/<prompt_dir>/<video_stem>/   (mirrors VP output layout)
        if prompt_dir:
            video_output_dir = os.path.join(local_output_dir, prompt_dir, Path(video_path).stem)
        else:
            video_output_dir = os.path.join(local_output_dir, Path(video_path).stem)
        metrics = _run_alpamayo_inference(
            video_path=video_path,
            output_dir=video_output_dir,
            model=model,
            processor=processor,
            helper_mod=helper_mod,
            device=device,
            num_traj_samples=num_traj_samples,
            black_non_target_cameras=black_non_target_cameras,
        )
        all_metrics.append(metrics)
    
    # Write report
    report_path = _write_report(
        output_dir=local_output_dir,
        run_id=output_run_id,
        metrics=all_metrics,
        video_data_source=video_data_gcs_path,
    )
    
    # Upload outputs to GCS
    gcs_output_path = _upload_outputs(local_output_dir, VLA_OUTPUT_PREFIX, output_run_id)
    
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"Output Location: {gcs_output_path}")
    logger.info(f"Report: {gcs_output_path}/{output_run_id}_report.txt")
    logger.info("=" * 80)
    
    return {
        "output_gcs_path": gcs_output_path,
        "report_path": f"{gcs_output_path}/{output_run_id}_report.txt",
        "num_videos": len(video_tuples),
        "num_successful": sum(1 for m in all_metrics if m.success),
        "num_failed": sum(1 for m in all_metrics if not m.success),
    }


@workflow
def alpamayo_vla_inference_wf(
    video_data_gcs_path: str,
    output_run_id: str,
    model_id: str = "nvidia/Alpamayo-R1-10B",
    num_traj_samples: int = 1,
    video_name: str = "auto",
    black_non_target_cameras: bool = True,
) -> dict:
    """
    Alpamayo VLA inference workflow.
    
    Args:
        video_data_gcs_path: GCS path to video data
        output_run_id: Unique identifier for this run
        model_id: Model identifier
        num_traj_samples: Number of trajectory samples
        video_name: Filter to a specific video by stem name ("auto" = all videos)
        black_non_target_cameras: If True, black out all non-target cameras
    
    Returns:
        Dictionary with results
    """
    return run_alpamayo_inference_task(
        video_data_gcs_path=video_data_gcs_path,
        output_run_id=output_run_id,
        model_id=model_id,
        num_traj_samples=num_traj_samples,
        video_name=video_name,
        black_non_target_cameras=black_non_target_cameras,
    )
