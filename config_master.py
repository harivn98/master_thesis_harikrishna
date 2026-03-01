"""Configuration constants for the Master Pipeline Orchestrator.

All GCS bucket paths, container images, mount names, and stage-specific
configuration live here so that workflow_master.py contains only the
HLX @task / @workflow definitions.
"""

import os

# ==============================================================================
# GCS BUCKET (shared across all stages)
# ==============================================================================
GCS_BUCKET = "mbadas-sandbox-research-9bb9c7f"

# ==============================================================================
# CONTAINER IMAGES  — set by scripts/build_and_run.sh before `hlx wf run`
# Image names include the run ID: <base>_<run_id>:<run_id>
# ==============================================================================
SAM2_CONTAINER_IMAGE = os.environ.get(
    "SAM2_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:latest",
)
SAM3_CONTAINER_IMAGE = os.environ.get(
    "SAM3_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam3:latest",
)
VP_CONTAINER_IMAGE = os.environ.get(
    "VP_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp:latest",
)
ALPAMAYO_CONTAINER_IMAGE = os.environ.get(
    "ALPAMAYO_CONTAINER_IMAGE",
    "europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/alpamayo_vla:latest",
)

# ==============================================================================
# STAGE 1 — SAM2 GCS / mount configuration
# ==============================================================================
SAM2_CKPT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam2_checkpoint"
SAM2_FUSE_NAME   = "sam2-checkpoints"

# ==============================================================================
# STAGE 1 (alt) — SAM3 GCS / mount configuration
# ==============================================================================
SAM3_CKPT_PREFIX = "workspace/user/hbaskar/Video_inpainting/sam3_checkpoint"
SAM3_FUSE_NAME   = "sam3-checkpoints"

SAM3_OUTPUT_BASE = os.environ.get(
    "SAM3_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/sam3",
)

# Default SAM model selection (overridden by build_and_run.sh)
SAM_MODEL = os.environ.get("SAM_MODEL", "sam2")
SAM3_TEXT_PROMPT = os.environ.get("SAM3_TEXT_PROMPT", "road surface")

SAM2_OUTPUT_BASE = os.environ.get(
    "SAM2_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/sam2",
)
SAM2_PREPROCESSED_OUTPUT_BASE = os.environ.get(
    "SAM2_PREPROCESSED_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/preprocessed_data_vp",
)

# ==============================================================================
# STAGE 2 — VideoPainter GCS / mount configuration
# ==============================================================================
VP_BUCKET_PREFIX      = "workspace/user/hbaskar/Video_inpainting/videopainter"
VP_DATA_PREFIX        = "workspace/user/hbaskar/outputs/preprocessed_data_vp"
VP_FUSE_NAME          = "vp-bucket"
VP_DATA_FUSE_NAME     = "data"
VP_VLM_7B_FUSE_NAME   = "vp-vlm-7b"
VLM_7B_GCS_PREFIX     = os.path.join(VP_BUCKET_PREFIX, "ckpt", "vlm", "Qwen2.5-VL-7B-Instruct")

TRAINED_FLUXFILL_FUSE_NAME = "vp-trained-fluxfill"
TRAINED_FLUXFILL_GCS_PREFIX = os.environ.get(
    "TRAINED_FLUXFILL_GCS_PATH",
    "workspace/user/hbaskar/Video_inpainting/videopainter/training/"
    "trained_checkpoint/fluxfill_single_white_solid_clearroad_20260212_151908",
)

VP_OUTPUT_BASE = os.environ.get(
    "VP_OUTPUT_BASE",
    f"gs://{GCS_BUCKET}/workspace/user/hbaskar/outputs/vp",
)

# ==============================================================================
# STAGE 3 — Alpamayo GCS / mount configuration
# ==============================================================================
VLA_BASE_PREFIX        = "workspace/user/hbaskar/Video_inpainting/vla"
ALPAMAYO_CKPT_PREFIX   = os.path.join(VLA_BASE_PREFIX, "alpamayo", "checkpoints")
ALPAMAYO_CKPT_FUSE     = "alpamayo-ckpt"
ALPAMAYO_DATA_FUSE     = "alpamayo-video-data"
ALPAMAYO_DATA_PREFIX   = "workspace/user/hbaskar/outputs/vp"

ALPAMAYO_OUTPUT_BASE = os.environ.get(
    "ALPAMAYO_OUTPUT_BASE",
    "workspace/user/hbaskar/outputs/alpamayo",
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
