"""SAM3 segmentation workflow for HLX.

Runs the SAM3 video segmentation pipeline (process_videos_sam3.py) inside a
container for road segmentation on autonomous driving videos.

SAM3 uses text-based prompting instead of point-based prompting (SAM2),
enabling semantic understanding of what to segment.
"""
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from hlx.wf import DedicatedNode, Node, fuse_prefetch_metadata, task, workflow
from hlx.wf.mounts import MOUNTPOINT, FuseBucket

logger = logging.getLogger(__name__)

# Container image
CONTAINER_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam3"
CONTAINER_IMAGE = os.environ.get("SAM3_CONTAINER_IMAGE", f"{CONTAINER_IMAGE_DEFAULT}:latest")

# Paths inside container
BASE_WORKDIR = "/workspace/sam3"

# GCS bucket paths for outputs
SAM3_OUTPUT_BUCKET_BASE = os.environ.get(
    "SAM3_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam3",
)
SAM3_PREPROCESSED_BUCKET_BASE = os.environ.get(
    "SAM3_PREPROCESSED_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# GCS bucket paths for checkpoints (mounted via FuseBucket)
SAM3_BUCKET = "mbadas-sandbox-research-9bb9c7f"
SAM3_CHECKPOINT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam3_checkpoint"

SAM3_FUSE_MOUNT_NAME = "sam3-checkpoints"
SAM3_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, SAM3_FUSE_MOUNT_NAME)
MOUNTED_CHECKPOINT_PATH = os.path.join(SAM3_FUSE_MOUNT_ROOT, "checkpoints", "sam3.pt")


def _resolve_chunk_uri(chunk_uri: str) -> List[str]:
    """Resolve a chunks:// URI into a list of gs:// video file paths."""
    from urllib.parse import urlparse, parse_qs
    import gcsfs

    parsed = urlparse(chunk_uri)
    base_path = parsed.netloc + parsed.path
    params = parse_qs(parsed.query)
    chunk_start = int(params.get("start", [0])[0])
    chunk_end = int(params.get("end", [0])[0])
    per_chunk = int(params.get("per_chunk", [1])[0])

    logger.info(
        "Resolving chunks:// URI — base=%s, chunks %d–%d, %d files/chunk",
        base_path, chunk_start, chunk_end, per_chunk,
    )

    fs = gcsfs.GCSFileSystem()
    resolved: List[str] = []

    for chunk_idx in range(chunk_start, chunk_end + 1):
        chunk_folder = f"{base_path}/chunk_{chunk_idx:04d}"
        try:
            files = fs.ls(chunk_folder, detail=False)
            mp4_files = sorted(f for f in files if f.endswith(".mp4"))
            selected = mp4_files[:per_chunk]
            for f in selected:
                resolved.append(f"gs://{f}")
            logger.info(
                "  chunk_%04d: %d mp4 files found, selected %d",
                chunk_idx, len(mp4_files), len(selected),
            )
        except FileNotFoundError:
            logger.warning("  chunk_%04d: folder not found at %s — skipping", chunk_idx, chunk_folder)

    logger.info("Resolved %d video files from %d chunks", len(resolved), chunk_end - chunk_start + 1)
    if not resolved:
        raise ValueError(
            f"No video files found for chunks:// URI: {chunk_uri}. "
            f"Checked {base_path}/chunk_NNNN/ for .mp4 files."
        )
    return resolved


@task(
    compute=DedicatedNode(
        node=Node.A100_80GB_1GPU,
        ephemeral_storage="max",
        max_duration="3d",
    ),
    container_image=CONTAINER_IMAGE,
    environment={"PYTHONUNBUFFERED": "1"},
    mounts=[
        FuseBucket(
            bucket=SAM3_BUCKET,
            name=SAM3_FUSE_MOUNT_NAME,
            prefix=SAM3_CHECKPOINT_PREFIX,
        ),
    ],
)
def run_sam3_segmentation(
    run_id: str,
    sam3_video_uris: str = "default",
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
    text_prompt: str = "road surface",
) -> str:
    """Run SAM3 video segmentation on a list of video URIs.

    Args:
        run_id: Unique identifier for this run
        sam3_video_uris: Video input specification (chunks://, gs://, comma-separated)
        upload_to_gcp: Whether to upload results to GCS
        upload_to_local: Whether to keep local copies
        max_frames: Maximum frames to extract per video
        text_prompt: Text prompt for SAM3 concept-based segmentation

    Returns:
        Summary message with GCS paths
    """
    # Parse video URIs
    video_uris: Optional[List[str]] = None
    if sam3_video_uris and sam3_video_uris != "default":
        if sam3_video_uris.startswith("chunks://"):
            video_uris = _resolve_chunk_uri(sam3_video_uris)
        elif sam3_video_uris.rstrip("/").startswith("gs://") and "," not in sam3_video_uris:
            video_uris = [sam3_video_uris]
        else:
            video_uris = [u.strip() for u in sam3_video_uris.split(",") if u.strip()]

    if video_uris is None:
        raise ValueError("No video URIs provided.")

    # Construct output paths
    output_bucket = f"{SAM3_OUTPUT_BUCKET_BASE}/{run_id}"
    preprocessed_bucket = f"{SAM3_PREPROCESSED_BUCKET_BASE}/{run_id}"

    logger.info("=" * 60)
    logger.info("SAM3 Segmentation Workflow (Text-Prompted)")
    logger.info(f"Processing {len(video_uris)} videos")
    logger.info(f"RUN_ID: {run_id}")
    logger.info(f"Text prompt: '{text_prompt}'")
    logger.info(f"Output bucket: {output_bucket}")
    logger.info(f"Preprocessed bucket: {preprocessed_bucket}")
    logger.info("=" * 60)

    # Download input videos from GCS
    import gcsfs

    video_cache_dir = Path("/tmp/sam3_video_cache")
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    fs = gcsfs.GCSFileSystem()
    local_video_paths = []
    for uri in video_uris:
        video_filename = uri.split("/")[-1]
        local_video_path = video_cache_dir / video_filename

        if local_video_path.exists():
            logger.info(f"Already downloaded: {local_video_path}")
            local_video_paths.append(str(local_video_path))
            continue

        gcs_path = uri
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path[len("gs://"):]
        elif gcs_path.startswith("https://storage.googleapis.com/"):
            gcs_path = gcs_path[len("https://storage.googleapis.com/"):]

        logger.info(f"Downloading gs://{gcs_path} -> {local_video_path}")
        try:
            fs.get(gcs_path, str(local_video_path))
            local_video_paths.append(str(local_video_path))
        except Exception as e:
            logger.error(f"Failed to download gs://{gcs_path}: {e}")
            raise

    logger.info(f"Downloaded {len(local_video_paths)} videos")
    video_uris = local_video_paths

    # Build command to run the processing script
    script_path = os.path.join(BASE_WORKDIR, "process_vide_sam3_hlxwf.py")
    cmd = [
        sys.executable,
        script_path,
        "--video-uris", *video_uris,
        "--output-bucket", output_bucket,
        "--preprocessed-bucket", preprocessed_bucket,
        "--max-frames", str(max_frames),
        "--run-id", run_id,
        "--text-prompt", text_prompt,
    ]

    if upload_to_gcp:
        cmd.append("--upload-gcp")
    if upload_to_local:
        cmd.append("--upload-local")

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE_WORKDIR,
    )

    if result.stdout:
        logger.info("STDOUT:")
        logger.info(result.stdout)

    if result.stderr:
        logger.warning("STDERR:")
        logger.warning(result.stderr)

    if result.returncode != 0:
        stdout_tail = (result.stdout or "")[-4000:]
        stderr_tail = (result.stderr or "")[-4000:]
        raise RuntimeError(
            f"SAM3 processing failed with exit code {result.returncode}.\n"
            f"--- stdout (tail) ---\n{stdout_tail}\n"
            f"--- stderr (tail) ---\n{stderr_tail}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = f"""
SAM3 Segmentation Complete!
===========================
Model: SAM3 (Segment Anything with Concepts)
Text prompt: '{text_prompt}'
Videos processed: {len(video_uris)}
Timestamp: {timestamp}

GCS Outputs:
- Raw outputs: {output_bucket}/
- VideoPainter format: {preprocessed_bucket}/

Each video has:
  - Binary masks (PNG)
  - Visualization overlays (JPG + MP4)
  - VideoPainter preprocessing (MP4 + NPZ masks + meta.csv)
"""
    logger.info(summary)
    return summary


@workflow
def sam3_segmentation_wf(
    run_id: str,
    sam3_video_uris: str = "default",
    upload_to_gcp: bool = True,
    upload_to_local: bool = False,
    max_frames: int = 150,
    text_prompt: str = "road surface",
) -> str:
    """Workflow: Process multiple videos with SAM3 for road segmentation.

    SAM3 uses text-based prompting for concept-driven segmentation,
    replacing SAM2's point-based initialization approach.

    Args:
        run_id: Unique identifier for this run
        sam3_video_uris: Video input specification
        upload_to_gcp: Upload results to GCS (default: True)
        upload_to_local: Keep local copies (default: False)
        max_frames: Max frames per video (default: 150)
        text_prompt: Text prompt for segmentation (default: 'road surface')

    Returns:
        Summary of processed videos and GCS output locations
    """
    return run_sam3_segmentation(
        run_id=run_id,
        sam3_video_uris=sam3_video_uris,
        upload_to_gcp=upload_to_gcp,
        upload_to_local=upload_to_local,
        max_frames=max_frames,
        text_prompt=text_prompt,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Segmentation Workflow")
    parser.add_argument(
        "--sam3-video-uris", type=str, default="default",
        help="Video input: chunks:// URI, gs:// path, comma-separated URIs, or 'default'",
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Maximum frames to extract per video",
    )
    parser.add_argument(
        "--text-prompt", type=str, default="road surface",
        help="Text prompt for SAM3 segmentation",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip GCS upload",
    )

    args = parser.parse_args()

    result = sam3_segmentation_wf(
        sam3_video_uris=args.sam3_video_uris,
        upload_to_gcp=not args.no_upload,
        upload_to_local=args.no_upload,
        max_frames=args.max_frames,
        text_prompt=args.text_prompt,
    )

    print(result)
