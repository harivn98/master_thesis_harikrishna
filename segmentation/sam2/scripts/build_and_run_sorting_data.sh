#!/bin/bash

set -euo pipefail

# FluxFill dataset filtering/sorting – runs filter combinations in sequence.
# Builds the Docker image once, then submits one HLX workflow per combination.
#
# Usage:
#   bash scripts/build_and_run_sorting_data.sh [CHUNK_START] [CHUNK_END]
#
# Defaults match the output of build_and_run_training_data.sh (chunks 0–99).
# The input path is auto-derived:  <DATA_BASE>/td-chunk<START>-<END>
#
# Examples:
#   bash scripts/build_and_run_sorting_data.sh          # chunks 0–99
#   bash scripts/build_and_run_sorting_data.sh 0 49     # chunks 0–49
#   bash scripts/build_and_run_sorting_data.sh 50 99    # chunks 50–99

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REMOTE_IMAGE="europe-west4-docker.pkg.dev/mb-adas-2015-p-a4db/research/harimt_sam2:${TIMESTAMP}"
TEAM_SPACE="research"
DOMAIN="prod"

# ---------- Chunk range (must match the data-generation run) ------------------
CHUNK_START="${1:-0}"
CHUNK_END="${2:-99}"

# ---------- Input datasets (auto-derived from chunk range) --------------------
# data_generation.py writes to:
#   gs://<BUCKET>/…/data/td-chunk<START>-<END>/
# with train.csv referencing chunk_XXXX/images/*.png and chunk_XXXX/masks/*.png
DATA_BASE="gs://mbadas-sandbox-research-9bb9c7f/workspace/user/hbaskar/Video_inpainting/videopainter/training/data"
INPUT_DIR="${DATA_BASE}/td-chunk${CHUNK_START}-${CHUNK_END}"

# CSV ordering + optional cap
SORT="prompt"    # prompt|image
LIMIT="0"        # 0 means no limit

# Keep only samples with clear lane markings (no 'unknown' attributes)
REQUIRE_CLEAR_ROAD="1"   # 1|0

# ── Filter combinations ──────────────────────────────────────────────────────
#  FORMAT: "COUNT COLOR PATTERN"
#  These match the Qwen-generated lane-attribute vocabulary from data_generation.py:
#    count:   single | double
#    color:   white  | yellow
#    pattern: solid  | dashed
COMBOS=(
	"single white  solid"
	"double white  solid"
	"single yellow solid"
	"double yellow solid"
	"single white  dashed"
)

# Output folder prefix — derived from input chunk range for traceability
# e.g. td-chunk0-99_single_white_solid
OUTPUT_PREFIX="td-chunk${CHUNK_START}-${CHUNK_END}"

# Ensure we run from the sam2 repo root (docker-compose.yaml lives there)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "============================================================"
echo "Sorting training data from: ${INPUT_DIR}"
echo "  Chunk range  : ${CHUNK_START}..${CHUNK_END}"
echo "  Combinations : ${#COMBOS[@]}"
echo "  Sort by      : ${SORT}"
echo "  Require clear: ${REQUIRE_CLEAR_ROAD}"
echo "============================================================"

# ── Build & push image (once) ────────────────────────────────────────────────
echo "Building docker image (sam2/frontend)..."
DOCKER_BUILDKIT=1 docker compose build

echo "Tagging and pushing image: ${REMOTE_IMAGE}"
docker tag sam2/frontend "${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "Image built with timestamp: ${TIMESTAMP}"

export SAM2_CONTAINER_IMAGE="${REMOTE_IMAGE}"
export FLUXFILL_DATA_CONTAINER_IMAGE="${REMOTE_IMAGE}"

# ── Submit one workflow per combination ───────────────────────────────────────
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	SUFFIX="${COUNT}_${COLOR}_${PATTERN}"
	OUTPUT_DIR="${DATA_BASE}/${OUTPUT_PREFIX}_${SUFFIX}"

	echo ""
	echo "================================================================================"
	echo " [$(( i + 1 ))/${#COMBOS[@]}]  ${COUNT} / ${COLOR} / ${PATTERN}"
	echo "   Input  : ${INPUT_DIR}"
	echo "   Output : ${OUTPUT_DIR}"
	echo "================================================================================"

	hlx wf run \
		--team-space "${TEAM_SPACE}" \
		--domain "${DOMAIN}" \
		--execution-name "filter-fluxfill-${SUFFIX//[_]/-}-${TIMESTAMP}" \
		filter_fluxfill_dataset.filter_fluxfill_dataset_wf \
		--input_dir "${INPUT_DIR}" \
		--suffix "${SUFFIX}" \
		--output_dir "${OUTPUT_DIR}" \
		--count "${COUNT}" \
		--color "${COLOR}" \
		--pattern "${PATTERN}" \
		--sort "${SORT}" \
		--limit "${LIMIT}" \
		--require_clear_road "${REQUIRE_CLEAR_ROAD}"

	echo "  → Submitted: ${OUTPUT_DIR}/"
done

echo ""
echo "All ${#COMBOS[@]} filter workflows submitted."
echo ""
echo "Input:  ${INPUT_DIR}"
echo "Output folders:"
for i in "${!COMBOS[@]}"; do
	read -r COUNT COLOR PATTERN <<< "${COMBOS[$i]}"
	echo "  ${DATA_BASE}/${OUTPUT_PREFIX}_${COUNT}_${COLOR}_${PATTERN}/"
done

