#!/bin/bash

set -euo pipefail

# ==================================================================================
# FluxFill LoRA training — 5 lane-type checkpoints from 5 sorted datasets
# ==================================================================================
#
# Trains one FluxFill LoRA per lane-marking combination, producing 5 independent
# checkpoints.  Each checkpoint specialises the base FluxFill model for a single
# lane type (e.g. "single white solid").
#
# Inputs:  5 filtered datasets produced by build_and_run_sorting_data.sh
#          (td-chunk<START>-<END>_<count>_<color>_<pattern>/)
#
# Outputs: 5 LoRA checkpoint folders under the same GCS checkpoint base:
#          trained_checkpoint/fluxfill_<count>_<color>_<pattern>_<TIMESTAMP>/
#
# Usage:
#   bash scripts/build_and_run_training_all.sh                    # chunks 0–99
#   bash scripts/build_and_run_training_all.sh 0 49               # chunks 0–49
#
# Each training job runs on A100_80GB_1GPU via the existing HLX workflow
# (training_workflow.fluxfill_training_wf).
# ==================================================================================

TEAM_SPACE="research"
DOMAIN="prod"

# ---------- Chunk range (must match the sorting run) ------------------------------
CHUNK_START="${1:-0}"
CHUNK_END="${2:-99}"

# ---------- GCS paths -------------------------------------------------------------
DATA_BASE="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data"
OUTPUT_CHECKPOINT_DIR="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/trained_checkpoint"

# Input prefix (output of build_and_run_sorting_data.sh)
INPUT_PREFIX="td-chunk${CHUNK_START}-${CHUNK_END}"

# ---------- Training hyperparameters ----------------------------------------------
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# ---------- Docker ----------------------------------------------------------------
X="${TRAIN_RUN_SUFFIX:-train_5lora}"
export VP_RUN_SUFFIX="${X}"
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_vp${X}"

# ---------- 5 lane-type combinations (must match build_and_run_sorting_data.sh) ---
COMBOS=(
	"single white  solid"
	"double white  solid"
	"single yellow solid"
	"double yellow solid"
	"single white  dashed"
)

# Timestamp shared across all 5 runs for traceability
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# ---------- Build & push image (once) ---------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "============================================================"
echo " FluxFill LoRA Training — 5 Lane-Type Checkpoints"
echo "============================================================"
echo "  Input prefix : ${INPUT_PREFIX}"
echo "  Combinations : ${#COMBOS[@]}"
echo "  Epochs       : ${NUM_TRAIN_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size   : ${TRAIN_BATCH_SIZE} (× ${GRAD_ACCUM} grad accum)"
echo "  Timestamp    : ${TIMESTAMP}"
echo "============================================================"
echo ""

echo "Building docker image (VideoPainter)..."
docker compose build

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_IMAGE_TAGGED="${REMOTE_IMAGE}:${RUN_TAG}"

echo "Tagging and pushing image: ${REMOTE_IMAGE_TAGGED}"
docker tag videopainter:latest "${REMOTE_IMAGE_TAGGED}"
docker tag videopainter:latest "${REMOTE_IMAGE}:latest"
docker push "${REMOTE_IMAGE_TAGGED}"
docker push "${REMOTE_IMAGE}:latest"

export VP_CONTAINER_IMAGE="${REMOTE_IMAGE_TAGGED}"

# ---------- Submit one training workflow per combination ---------------------------
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	SUFFIX="${COUNT}_${COLOR}_${PATTERN}"

	INPUT_DATA_DIR="${DATA_BASE}/${INPUT_PREFIX}_${SUFFIX}"
	OUTPUT_RUN_ID="fluxfill_${SUFFIX}_${TIMESTAMP}"

	echo ""
	echo "================================================================================"
	echo " [$(( i + 1 ))/${#COMBOS[@]}]  Training: ${COUNT} / ${COLOR} / ${PATTERN}"
	echo "   Input dataset : ${INPUT_DATA_DIR}"
	echo "   Output ckpt   : ${OUTPUT_CHECKPOINT_DIR}/${OUTPUT_RUN_ID}/"
	echo "================================================================================"

	WF_CMD="hlx wf run \
		--team-space \"${TEAM_SPACE}\" \
		--domain \"${DOMAIN}\" \
		--execution-name \"train-fluxfill-${SUFFIX//[_]/-}-${TIMESTAMP//[_]/-}\" \
		training_workflow.fluxfill_training_wf \
		--input_data_dir \"${INPUT_DATA_DIR}\" \
		--output_checkpoint_dir \"${OUTPUT_CHECKPOINT_DIR}\" \
		--output_run_id \"${OUTPUT_RUN_ID}\" \
		--mixed_precision \"${MIXED_PRECISION}\" \
		--train_batch_size \"${TRAIN_BATCH_SIZE}\" \
		--gradient_accumulation_steps \"${GRAD_ACCUM}\" \
		--learning_rate \"${LEARNING_RATE}\" \
		--lr_scheduler \"${LR_SCHEDULER}\" \
		--lr_warmup_steps \"${LR_WARMUP_STEPS}\" \
		--num_train_epochs \"${NUM_TRAIN_EPOCHS}\" \
		--checkpointing_steps \"${CHECKPOINTING_STEPS}\""

	if [ -n "${MAX_TRAIN_STEPS}" ]; then
		WF_CMD="${WF_CMD} --max_train_steps \"${MAX_TRAIN_STEPS}\""
	fi

	eval "${WF_CMD}"
	echo "  → Submitted: ${OUTPUT_CHECKPOINT_DIR}/${OUTPUT_RUN_ID}/"
done

echo ""
echo "============================================================"
echo " All ${#COMBOS[@]} training workflows submitted."
echo "============================================================"
echo ""
echo "Checkpoint folders (once training completes):"
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	SUFFIX="${COUNT}_${COLOR}_${PATTERN}"
	echo "  ${OUTPUT_CHECKPOINT_DIR}/fluxfill_${SUFFIX}_${TIMESTAMP}/"
done
echo ""
echo "To use a checkpoint for inference, set TRAINED_FLUXFILL_GCS_PATH in build_and_run.sh:"
echo "  e.g. TRAINED_FLUXFILL_GCS_PATH=workspace/user/hbaskar/.../fluxfill_single_white_solid_${TIMESTAMP}"
