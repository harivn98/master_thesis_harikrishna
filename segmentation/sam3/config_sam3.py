"""SAM3 workflow configuration constants."""
import os

from hlx.wf.mounts import MOUNTPOINT

# ----------------------------------------------------------------------------------
# CONTAINER IMAGE
# ----------------------------------------------------------------------------------
CONTAINER_IMAGE_DEFAULT = "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam3"
CONTAINER_IMAGE = os.environ.get("SAM3_CONTAINER_IMAGE", f"{CONTAINER_IMAGE_DEFAULT}:latest")

# ----------------------------------------------------------------------------------
# PATHS (inside container)
# ----------------------------------------------------------------------------------
BASE_WORKDIR = "/workspace/sam3"

# ----------------------------------------------------------------------------------
# GCS OUTPUT PATHS
# ----------------------------------------------------------------------------------
SAM3_OUTPUT_BUCKET_BASE = os.environ.get(
    "SAM3_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/sam3",
)
SAM3_PREPROCESSED_BUCKET_BASE = os.environ.get(
    "SAM3_PREPROCESSED_OUTPUT_BASE",
    "gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# ----------------------------------------------------------------------------------
# GCS / FUSEBUCKET CHECKPOINT MOUNTS
# ----------------------------------------------------------------------------------
SAM3_BUCKET = "mbadas-sandbox-research-9bb9c7f"
SAM3_CHECKPOINT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam3_checkpoint"

SAM3_FUSE_MOUNT_NAME = "sam3-checkpoints"
SAM3_FUSE_MOUNT_ROOT = os.path.join(MOUNTPOINT, SAM3_FUSE_MOUNT_NAME)
MOUNTED_CHECKPOINT_PATH = os.path.join(SAM3_FUSE_MOUNT_ROOT, "checkpoints", "sam3.pt")
