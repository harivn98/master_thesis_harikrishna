"""SAM2 workflow configuration constants."""
import os

from hlx.wf.mounts import MOUNTPOINT

# ----------------------------------------------------------------------------------
# CONTAINER IMAGE
# ----------------------------------------------------------------------------------
CONTAINER_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2"
CONTAINER_IMAGE = os.environ.get("SAM2_CONTAINER_IMAGE", f"{CONTAINER_IMAGE_DEFAULT}:latest")

# ----------------------------------------------------------------------------------
# PATHS (inside container)
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/sam2"
DEFAULT_CHECKPOINT = os.path.join(BASE_WORKDIR, "checkpoints", "sam2.1_hiera_large.pt")
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ----------------------------------------------------------------------------------
# GCS OUTPUT PATHS
# ----------------------------------------------------------------------------------
SAM2_OUTPUT_BUCKET_BASE = os.environ.get(
    "SAM2_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam2",
)
SAM2_PREPROCESSED_BUCKET_BASE = os.environ.get(
    "SAM2_PREPROCESSED_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# ----------------------------------------------------------------------------------
# GCS / FUSEBUCKET CHECKPOINT MOUNTS
# ----------------------------------------------------------------------------------
SAM2_BUCKET = "mbadas-sandbox-research-9bb9c7f"
SAM2_CHECKPOINT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"

SAM2_FUSE_MOUNT_NAME = "sam2-checkpoints"
SAM2_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, SAM2_FUSE_MOUNT_NAME)
MOUNTED_CHECKPOINT_PATH = os.path.join(SAM2_FUSE_MOUNT_ROOT, "checkpoints", "sam2.1_hiera_large.pt")
