"""Alpamayo VLA workflow configuration constants."""
import os

from hlx.wf import Node
from hlx.wf.mounts import MOUNTPOINT

# ----------------------------------------------------------------------------------
# GCS BUCKET
# ----------------------------------------------------------------------------------
VP_BUCKET = "mbadas-sandbox-research-9bb9c7f"
VLA_BASE_PREFIX = "workspace/user/hbaskar/Video_inpainting/vla"

# ----------------------------------------------------------------------------------
# CHECKPOINT PATHS IN GCS
# ----------------------------------------------------------------------------------
ALPAMAYO_CKPT_PREFIX = os.path.join(VLA_BASE_PREFIX, "alpamayo", "checkpoints")

# ----------------------------------------------------------------------------------
# OUTPUT PATH IN GCS
# ----------------------------------------------------------------------------------
VLA_OUTPUT_PREFIX = os.environ.get(
    "ALPAMAYO_OUTPUT_BASE",
    "workspace/user/hbaskar/outputs/alpamayo",
)

# ----------------------------------------------------------------------------------
# CONTAINER IMAGE
# ----------------------------------------------------------------------------------
REMOTE_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla"
CONTAINER_IMAGE = os.environ.get("ALPAMAYO_CONTAINER_IMAGE", f"{REMOTE_IMAGE_DEFAULT}:latest")

# ----------------------------------------------------------------------------------
# COMPUTE
# ----------------------------------------------------------------------------------
COMPUTE_NODE = Node.A100_80GB_1GPU

# ----------------------------------------------------------------------------------
# FUSEBUCKET MOUNT PATHS
# ----------------------------------------------------------------------------------
CKPT_FUSE_MOUNT_NAME = "alpamayo-ckpt"
CKPT_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, CKPT_FUSE_MOUNT_NAME)

VIDEO_DATA_GCS_PREFIX = os.environ.get(
    "ALPAMAYO_VIDEO_DATA_PREFIX",
    "workspace/user/hbaskar/outputs/vp",
)
VIDEO_DATA_FUSE_MOUNT_NAME = "alpamayo-video-data"
VIDEO_DATA_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, VIDEO_DATA_FUSE_MOUNT_NAME)

# ----------------------------------------------------------------------------------
# LOCAL PATHS INSIDE CONTAINER
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/alpamayo"
CKPT_LOCAL_PATH = os.path.join(BASE_WORKDIR, "checkpoints")
SCRATCH_OUTPUT_BASE = "/tmp/alpamayo_output"
