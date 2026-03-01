"""Configuration constants for the VideoPainter HLX workflow.

All GCS bucket paths, container images, mount names, default paths, and
derived settings live here so that workflow_vp.py contains only the
HLX @task / @workflow definitions.
"""

import os

from hlx.wf import Node
from hlx.wf.mounts import MOUNTPOINT

# ----------------------------------------------------------------------------------
# RUN SUFFIX (set via VP_RUN_SUFFIX env var from scripts/build_and_run.sh)
# ----------------------------------------------------------------------------------
X = (os.environ.get("VP_RUN_SUFFIX") or "").strip()


# ----------------------------------------------------------------------------------
# VLM (Qwen2.5-VL-7B) MOUNT
# ----------------------------------------------------------------------------------
# We only support mounting the 7B VLM checkpoint via a dedicated FuseBucket and
# symlinking it into the expected local path:
#   /workspace/VideoPainter/ckpt/vlm/Qwen2.5-VL-7B-Instruct -> <mounted_folder>
#
# This avoids relying on HuggingFace Hub downloads at runtime and avoids needing
# the VLM checkpoint to live inside the main ckpt mount.
LLM_MODEL_SIZE = "7B"

USE_QWEN2_5_VL_7B = True

# Fixed to 1 GPU for 7B model
COMPUTE_NODE = Node.A100_80GB_1GPU
# To dedicate the second GPU to Qwen (VLM) when available, keep Flux on the same
# GPU as CogVideoX by default.
VP_FLUX_DEVICE_DEFAULT = "cuda:0"


VP_RUN_SUFFIX = X

# Allow the runner script to pin an exact image tag (avoids stale ':latest' pulls).
# The build_and_run.sh script sets VP_CONTAINER_IMAGE with RUN_TAG in the image name
# e.g. europe-west4-docker.pkg.dev/.../harimt_vp<suffix>_<run_tag>:<run_tag>
REMOTE_IMAGE = (
	f"europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp{VP_RUN_SUFFIX}"
	if VP_RUN_SUFFIX
	else "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp"
)
CONTAINER_IMAGE_DEFAULT = f"{REMOTE_IMAGE}:latest"
CONTAINER_IMAGE = os.environ.get("VP_CONTAINER_IMAGE", CONTAINER_IMAGE_DEFAULT)

# ----------------------------------------------------------------------------------
# PATHS (inside container) and GCS mount
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/VideoPainter"
DEFAULT_DATA_DIR = os.path.join(BASE_WORKDIR, "data")
DEFAULT_CKPT_DIR = os.path.join(BASE_WORKDIR, "ckpt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_WORKDIR, "output_vp")

OUTPUT_SUBDIR = "output/vp"
SCRATCH_BASE = "/tmp/videopainter_output"
SCRATCH_DATA_BASE = "/tmp/videopainter_data"

# GCS bucket paths for checkpoints
VP_BUCKET = "mbadas-sandbox-research-9bb9c7f"
VP_BUCKET_PREFIX = "workspace/user/hbaskar/Video_inpainting/videopainter"

# Optional VLM checkpoint folder (mounted separately so it doesn't interfere with
# existing ckpt or data mounts).
# NOTE: In our bucket layout, VLM checkpoints live under `ckpt/vlm/...`.
VLM_7B_GCS_PREFIX = os.path.join(VP_BUCKET_PREFIX, "ckpt", "vlm", "Qwen2.5-VL-7B-Instruct")

# Trained FluxFill checkpoint folder (configurable via environment variable)
TRAINED_FLUXFILL_GCS_PREFIX = os.environ.get(
	"TRAINED_FLUXFILL_GCS_PATH",
	"workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_144012",
)

# GCS bucket path for SAM2 preprocessed data.
# IMPORTANT: we mount the *base* prefix so `data_run_id` can be chosen dynamically.
DEFAULT_DATA_RUN_ID = os.environ.get("DATA_RUN_ID", "10")
VP_DATA_PREFIX = "workspace/user/hbaskar/outputs/preprocessed_data_vp"

# FuseBucket mount paths
VP_FUSE_MOUNT_NAME = "vp-bucket"
VP_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_FUSE_MOUNT_NAME)
MOUNTED_CKPT_PATH = os.path.join(VP_FUSE_MOUNT_ROOT, "ckpt")

VP_VLM_7B_FUSE_MOUNT_NAME = "vp-vlm-7b"
VP_VLM_7B_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_VLM_7B_FUSE_MOUNT_NAME)

VLM_7B_DEST_PATH = os.path.join(DEFAULT_CKPT_DIR, "vlm", "Qwen2.5-VL-7B-Instruct")

TRAINED_FLUXFILL_FUSE_MOUNT_NAME = "vp-trained-fluxfill"
TRAINED_FLUXFILL_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, TRAINED_FLUXFILL_FUSE_MOUNT_NAME)
TRAINED_FLUXFILL_DEST_PATH = os.path.join(DEFAULT_CKPT_DIR, "trained_fluxfill_lora")

VP_DATA_FUSE_MOUNT_NAME = "data"
VP_DATA_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VP_DATA_FUSE_MOUNT_NAME)

# Override via VP_OUTPUT_BASE env var in build_and_run.sh
GCS_OUTPUT_BASE = os.environ.get(
	"VP_OUTPUT_BASE",
	f"gs://{VP_BUCKET}/workspace/user/hbaskar/outputs/vp",
)

DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CKPT_DIR, "CogVideoX-5b-I2V")
DEFAULT_BRANCH_PATH = os.path.join(DEFAULT_CKPT_DIR, "VideoPainter/checkpoints/branch")
DEFAULT_IMG_INPAINT_PATH = os.path.join(DEFAULT_CKPT_DIR, "flux_inp")


DEFAULT_META = os.path.join(DEFAULT_DATA_DIR, "meta.csv")
DEFAULT_VIDEO_ROOT = os.path.join(DEFAULT_DATA_DIR, "raw_videos")
