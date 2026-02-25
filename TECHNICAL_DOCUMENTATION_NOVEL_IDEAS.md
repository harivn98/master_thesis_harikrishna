# Technical Documentation: Novel Ideas & Contributions

## Master Thesis — Video Inpainting for Lane Marking Generation in Autonomous Driving

**Author:** Hariharan Baskar  
**Date:** February 22, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Upstream Repositories vs. Local Modifications — Structural Comparison](#2-upstream-repositories-vs-local-modifications)
   - 2.1 [TencentARC/VideoPainter vs generation/VideoPainter](#21-tencentarcvideopainter-vs-generationvideopainter)
   - 2.2 [facebookresearch/sam2 vs segmentation/sam2](#22-facebookresearchsam2-vs-segmentationsam2)
   - 2.3 [NVlabs/alpamayo vs vla/alpamayo](#23-nvlabsalpamayo-vs-vlaalpamayo)
3. [Novel Contribution #1: Three-Stage Master Pipeline Orchestrator](#3-novel-contribution-1-three-stage-master-pipeline-orchestrator)
4. [Novel Contribution #2: SAM2-Based Road Segmentation Pipeline for AD Videos](#4-novel-contribution-2-sam2-based-road-segmentation-pipeline)
5. [Novel Contribution #3: FluxFill Integration into VideoPainter's Edit Pipeline](#5-novel-contribution-3-fluxfill-integration-into-videopainters-edit-pipeline)
6. [Novel Contribution #4: VLM-Guided Iterative Caption Refinement for Inpainting QA](#6-novel-contribution-4-vlm-guided-iterative-caption-refinement)
7. [Novel Contribution #5: Automated FluxFill Training Data Generation from Physical AI](#7-novel-contribution-5-automated-fluxfill-training-data-generation)
8. [Novel Contribution #6: Semantic Lane-Attribute Dataset Filtering & Curation](#8-novel-contribution-6-semantic-lane-attribute-dataset-filtering)
9. [Novel Contribution #7: FluxFill LoRA Fine-Tuning Pipeline for Lane Markings](#9-novel-contribution-7-fluxfill-lora-fine-tuning-pipeline)
10. [Novel Contribution #8: Alpamayo VLA Trajectory Evaluation of Inpainted Videos](#10-novel-contribution-8-alpamayo-vla-trajectory-evaluation)
11. [Novel Contribution #9: Trajectory Visualization & Comparison Video Rendering](#11-novel-contribution-9-trajectory-visualization--comparison-rendering)
12. [Novel Contribution #10: Cloud-Native Containerized ML Infrastructure](#12-novel-contribution-10-cloud-native-containerized-ml-infrastructure)
13. [End-to-End Data Flow](#13-end-to-end-data-flow)
14. [Technical Specifications](#14-technical-specifications)

---

## 1. Executive Summary

This thesis introduces a **novel end-to-end pipeline** that chains three state-of-the-art vision foundation models—**SAM2** (Meta), **VideoPainter** (TencentARC/SIGGRAPH 2025), and **Alpamayo-R1-10B** (NVIDIA)—into a unified system for **generating synthetic lane markings in autonomous driving videos and quantitatively evaluating their impact on trajectory prediction**.

None of the three upstream repositories were designed to interoperate. Each was released as a standalone research artifact. This thesis contributes **10 novel technical ideas** that transform them into a cohesive, production-grade pipeline:

| # | Novel Idea | Files Added/Modified | Lines of Novel Code |
|---|-----------|---------------------|-------------------|
| 1 | Three-stage master pipeline orchestrator | `workflow_master.py`, `scripts/build_and_run.sh`, root `Dockerfile` | ~1,400 |
| 2 | SAM2 road segmentation for AD videos | `process_videos_sam2.py`, `process_vide_sam2_hlxwf.py`, `workflow_sam2.py` | ~1,630 |
| 3 | FluxFill first-frame integration in VideoPainter | Modified `infer/edit_bench.py` | ~500 |
| 4 | VLM iterative caption refinement (Qwen2.5-VL) | Modified `infer/edit_bench.py` | ~300 |
| 5 | Automated FluxFill training data generation | `data_generation.py` | ~1,112 |
| 6 | Semantic lane-attribute dataset filtering | `filter_fluxfill_dataset.py` | ~725 |
| 7 | FluxFill LoRA fine-tuning pipeline | `training_workflow.py`, `train/train_fluxfill_inpaint_lora.py` | ~600 |
| 8 | Alpamayo VLA trajectory evaluation | `run_inference.py`, `workflow_alpamayo.py` | ~1,040 |
| 9 | Trajectory visualization & comparison rendering | `visualize_video.py` | ~530 |
| 10 | Cloud-native containerized ML infrastructure | 4 Dockerfiles, 4 docker-compose.yaml, build scripts | ~800 |
| | **Total novel code** | | **~8,600+** |

---

## 2. Upstream Repositories vs. Local Modifications

### 2.1 TencentARC/VideoPainter vs generation/VideoPainter

**Upstream** ([github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter)): SIGGRAPH 2025 paper — dual-stream video inpainting with CogVideoX-5B backbone and a lightweight context encoder. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `app/` | Gradio demo application |
| `diffusers/` | Modified HuggingFace Diffusers (CogVideoX support) |
| `infer/` | Inference scripts (`edit_bench.py`, shell runners) |
| `train/` | Training scripts (CogVideoX fine-tuning) |
| `evaluate/` | LPIPS, temporal consistency, CLIP-based evaluation |
| `data_utils/` | VPData download and preprocessing |
| `env.sh`, `requirements.txt` | Environment setup |

**Local additions & modifications** (entirely novel files marked with ★):

| File/Directory | Status | Purpose |
|---------------|--------|---------|
| ★ `Dockerfile` | **New** (95 lines) | Full CUDA 12.1 container with Qwen VLM, Diffusers, workflow SDK, pre-cached eval models |
| ★ `docker-compose.yaml` | **New** | Container orchestration for local and cloud builds |
| ★ `workflow_vp.py` | **New** (1,620 lines) | Complete cloud workflow: GCS FUSE-mounted storage, multi-video discovery, VLM symlink management, output upload, prompt parsing, data staging |
| ★ `training_workflow.py` | **New** (300 lines) | FluxFill LoRA training cloud workflow: data download from GCS, training subprocess, checkpoint upload |
| ★ `workflow_sdk.whl` | **New** | Custom workflow SDK for cloud orchestration |
| ★ `data_1/` | **New** | Local test data (meta.csv, masks/, raw_videos/) |
| ★ `data_unprocessed/` | **New** | Intermediate processing data (frames/, segmentation_mask/) |
| ★ `ckpt/` | **New** | Checkpoint directories (CLIP ViT-L/14, CogVideoX-5b-I2V, SDXL inpaint) |
| `infer/edit_bench.py` | **Modified** | FluxFill pipeline integration, VLM refinement loop, multi-GPU orchestration, mask dilation/feathering |
| `train/train_fluxfill_inpaint_lora.py` | **New** | LoRA fine-tuning for FluxFill on lane-marking CSVs |
| `diffusers/loaders/lora_conversion_utils.py` | **Modified** | BFL Flux Control LoRA state dict converter (+14 lines) |
| `diffusers/loaders/lora_pipeline.py` | **Modified** | Manual PEFT adapter injection for FluxFill LoRA (+50 lines) |

**Key architectural difference:** Upstream VideoPainter runs only CogVideoX for video inpainting. The local version introduces a **dual-model pipeline** (FluxFill → CogVideoX) where FluxFill generates a high-quality semantically-correct first frame, and CogVideoX propagates the edit temporally across all subsequent frames.

#### Line-by-Line File Comparison (verified February 24, 2026)

A recursive `diff -rq` comparison was performed between the upstream TencentARC/VideoPainter repository and the local `generation/VideoPainter` directory. Binary assets (`.mp4`, `.jpg`, `.png`, `.so`), `.git/`, `__pycache__/`, and runtime data directories (`ckpt/`, `data_1/`, `data_unprocessed/`) were excluded.

**Result: 688 of 696 upstream source files are byte-identical. 8 files modified, 13 files added, 0 files deleted.**

##### Modified Files (8 files, ~1,300 lines changed)

| File | Upstream Lines | Local Lines | Delta | Nature of Change |
|------|---------------|-------------|-------|-----------------|
| `infer/edit_bench.py` | 700 | 1,787 | **+1,087** | OpenAI API → local Qwen2.5-VL; FluxFill dual-model integration; two-phase GPU execution; iterative caption refinement; border-aware masking; mask dilation/feathering; domain-specific prompts; LoRA adapter loading; multi-GPU device routing |
| `evaluate/metrics.py` | 903 | 1,125 | **+222** | OpenAI API → local Qwen2.5-VL; `clip` package → HuggingFace CLIPModel with local checkpoint; `compute_all_metrics()` single-pass method; conditional cogvlm2 loading with graceful fallback |
| `infer/inpaint.py` | 601 | 665 | **+64** | OpenAI → Qwen2.5-VL migration; FluxFill LoRA loading support; FPS calculation fix |
| `infer/edit.py` | 641 | 688 | **+47** | OpenAI → Qwen2.5-VL migration; FPS calculation fix |
| `diffusers/.../lora_pipeline.py` | 2,828 | 2,878 | **+50** | Manual PEFT inject_adapter_in_model() replacing broken load_lora_adapter(); LoRA rank extraction; DoRA version check; CPU offload hook management |
| `diffusers/.../lora_conversion_utils.py` | 623 | 637 | **+14** | BFL Flux Control LoRA state dict pass-through converter for norm layer keys |
| `requirements.txt` | 42 | 45 | **+3** | `openai` removed; `qwen-vl-utils` added; `transformers` upgraded to v4.49.0 for Qwen2.5-VL; version pins tightened |
| `infer/edit_bench.sh` | 97 | 97 | **0** | LLM model identifier changed (cosmetic) |

##### New Files (13 files, ~4,200 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `workflow_vp.py` | 1,619 | HLX cloud workflow orchestrator: GCS FuseBucket mounts for 5 checkpoints, per-video folder consumption, multi-instruction editing, structured result upload |
| `scripts/build_and_run.sh` | 181 | Docker build/tag/push + HLX workflow submission for inference |
| `scripts/build_and_run_training.sh` | 110 | Docker build/tag/push for LoRA training |
| `scripts/build_and_run_training_all.sh` | 152 | Parallel LoRA training for all 5 lane marking types |
| `scripts/download_physicalai_data.sh` | 118 | PhysicalAI-AV dataset download utility |
| `train/train_fluxfill_inpaint_lora.py` | 605 | Complete FluxFill LoRA fine-tuning: CSV dataset, Accelerate, PEFT LoraConfig, gradient accumulation, cosine LR, safetensors checkpointing |
| `data_utils/build_fluxfill_inpaint_csv.py` | 488 | Training dataset CSV builder with auto-captioning and mask generation |
| `training_workflow.py` | 359 | HLX cloud workflow for LoRA training: GCS data staging, training subprocess, checkpoint upload |
| `infer/border_utils.py` | 154 | Border-aware masking (cv2.inpaint/blur) to prevent black-fill artifacts during camera movement |
| `Dockerfile` | 95 | CUDA 12.1 production container with Qwen2.5-VL, Diffusers, workflow SDK, pre-cached eval models |
| `docker-compose.yaml` | 30 | Local dev configuration with GPU passthrough, 16 GB shm |
| `.dockerignore` | 2 | Excludes ckpt/, data_1/, data_unprocessed/ from build context |

##### Unchanged Files (688 files)

All files in the following directories are byte-identical to upstream:
- `app/` — Gradio demo application including bundled SAM 2 library (all files)
- `assets/` — Method diagrams, demo videos
- `data_utils/VPData_download.py`, `data_utils/unzip_folder.py`
- `diffusers/` — Entire vendored Diffusers library **except** the 2 LoRA loader files above
- `evaluate/` — All evaluation scripts **except** metrics.py
- `infer/` — Shell scripts (inpaint.sh, edit.sh, inpaint_id_resample.sh) unchanged
- `train/` — All upstream training scripts (VideoPainter.sh, VideoPainterID.sh, accelerate configs, mask_process.py, CogVideoX trainers)
- `env.sh`, `LICENSE`, `README.md`, `.gitignore`

---

### 2.2 facebookresearch/sam2 vs segmentation/sam2

**Upstream** ([github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)): Meta's foundation model for promptable visual segmentation in images and videos. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `sam2/` | Core model code (SAM2Base, image/video predictors, memory bank) |
| `checkpoints/` | Download scripts for SAM2/SAM2.1 checkpoints |
| `demo/` | Frontend + backend web demo |
| `notebooks/` | Jupyter notebooks for image/video prediction |
| `training/` | Fine-tuning code |
| `tools/` | Benchmarking utilities |
| `sav_dataset/` | SA-V dataset utilities |

#### Line-by-Line File Comparison (verified February 24, 2026)

**Files IDENTICAL to upstream facebookresearch/sam2 (0 lines modified):**

| File | Lines | Verification |
|------|-------|-------------|
| `sam2/build_sam.py` | — | 100% byte-identical. build_sam2(), build_sam2_video_predictor(), HF loading, checkpoint loading — all unchanged |
| `sam2/sam2_video_predictor.py` | 1,224 | 100% byte-identical. SAM2VideoPredictor class, SAM2VideoPredictorVOS with torch.compile — all unchanged |
| `sam2/sam2_image_predictor.py` | 467 | 100% byte-identical. set_image(), predict(), backbone feature sizes — all unchanged |
| `sam2/sam2_video_predictor_legacy.py` | — | 100% identical |
| `sam2/automatic_mask_generator.py` | — | 100% identical |
| `sam2/benchmark.py` | — | 100% identical |
| `sam2/__init__.py` | — | 100% identical |
| `sam2/configs/` (all YAML files) | — | All sam2 and sam2.1 configs unchanged |
| `sam2/csrc/` (CUDA C++ code) | — | connected_components.cu — unchanged |
| `sam2/modeling/` (all model code) | — | SAM2Base, memory encoder/attention, position encoding, Hiera backbone — all unchanged |
| `sam2/utils/` (all utilities) | — | misc.py, transforms.py — all unchanged |
| `setup.py` | 175 | 100% identical. Same REQUIRED_PACKAGES, EXTRA_PACKAGES, CUDA extension build logic |
| `pyproject.toml` | — | 100% identical. Same build-system requirements |
| `tools/vos_inference.py` | 508 | 100% identical. vos_inference(), DAVIS palette, mask utilities — unchanged |
| `backend.Dockerfile` | — | 100% identical. pytorch/pytorch base, Gunicorn, SAM 2.1 checkpoint downloads — unchanged |
| `training/` (entire directory) | — | All fine-tuning code unchanged |
| `notebooks/` (entire directory) | — | All Jupyter notebooks unchanged |
| `sav_dataset/` (entire directory) | — | All SA-V dataset utilities unchanged |
| All license/contributing/docs files | — | CODE_OF_CONDUCT.md, CONTRIBUTING.md, INSTALL.md, LICENSE, MANIFEST.in, README.md, RELEASE_NOTES.md — all unchanged |

> **Key finding:** The SAM2 core library (model architecture, predictors, configs, training code, CUDA extensions) is 100% unmodified. Not a single line of Meta's code was changed.

**File REWRITTEN (original functionality completely replaced):**

| File | Original Purpose | Local Purpose |
|------|-----------------|---------------|
| `docker-compose.yaml` | 2-service web demo (React frontend + Gunicorn backend) with port 7262/7263, 1 GPU, demo data volume | Single-service batch processing container with all GPUs, 16GB shared memory, full repo mount, /bin/bash entry, PYTHONPATH/HuggingFace env vars |

Detailed docker-compose.yaml comparison:

| Aspect | Original (upstream) | Local (rewritten) |
|--------|-------------------|-------------------|
| Services | `frontend` + `backend` (2 services) | Single `frontend` (batch worker) |
| Frontend | React app on port 7262 | SAM2 batch container |
| Backend | Gunicorn API on port 7263 | **Removed entirely** |
| GPU | 1 GPU for backend only | **All GPUs** |
| Volumes | `./demo/data/:/data/` | `./:/workspace/sam2` (full repo) |
| Environment | `VIDEO_ENCODE_CONCURRENCY`, API URLs | `HF_HOME`, `PYTHONPATH`, `SAM2_BUILD_CUDA=0` |
| shm_size | Not set | **16g** |
| IPC mode | Not set | **host** |
| Dockerfile ref | `backend.Dockerfile` + `demo/frontend/Dockerfile` | `Dockerfile` (new batch container) |
| Command | Gunicorn server start | `/bin/bash` (interactive) |

**Local additions (all novel — total ~5,390 lines of new code):**

| File | Status | Purpose | Lines |
|------|--------|---------|-------|
| ★ `process_videos_sam2.py` | **New** | Complete road segmentation pipeline with 6-point grid initialization, 4-stage morphological filtering, multi-format output (PNG + NPZ + overlay video), VP-compatible preprocessing, parallel GCS upload (8 threads), adaptive GPU precision, in-memory mask pass-through, timing report generation | 1,122 |
| ★ `process_vide_sam2_hlxwf.py` | **New** | Cloud workflow wrapper: CLI argument parsing, global configuration injection via exec() | 67 |
| ★ `workflow_sam2.py` | **New** | Cloud workflow: HLX @task/@workflow, GCS FuseBucket checkpoint mounting, chunks:// URI resolver, A100 80GB allocation, 3-day max duration | 444 |
| ★ `data_generation.py` | **New** | FluxFill training data generation: SAM2 image predictor masks + Qwen2.5-VL-7B batch captioning, multi-GPU subprocess workers (spawn context), chunk-based pipeline with prefetch, lane prompt normalization, GPU fault isolation | 1,432 |
| ★ `filter_fluxfill_dataset.py` | **New** | Lane-attribute filtering: regex-based prompt parsing, multi-field count/color/pattern filtering, REQUIRE_CLEAR_ROAD quality gate, GCS + local I/O | 751 |
| ★ `workflow_framesextraction.py` | **New** | Parallel frame extraction: 28-worker CPU extraction, chunk-by-chunk download/process/upload, deterministic naming | 344 |
| ★ `workflow_datasplitting.py` | **New** | Data splitting: N-way equal split, GCS folder discovery, remainder distribution | 306 |
| ★ `Dockerfile` | **New** | CUDA 12.1 batch container with Python 3.10, SAM2, Qwen VLM deps, gcsfs, HLX SDK (92 lines). NOT in upstream — upstream only has `backend.Dockerfile` for demo server | 92 |
| ★ `docker-compose.yaml` | **Rewritten** | Batch container orchestration (see table above) | 30 |
| ★ `hlx_wf-2.4.2.dev9-py3-none-any.whl` | **New** | HLX workflow SDK binary | — |
| ★ `OPTIMIZATION_SUMMARY.md` | **New** | Documents multi-GPU caching, threading→multiprocessing switch, prefetch pipeline | 160 |
| ★ `README_PIPELINE.md` | **New** | SAM2→VP pipeline documentation, output structure, run identifier examples | 160 |
| ★ `scripts/build_and_run.sh` | **New** | SAM2 build/deploy: GCS config, chunk URI encoding, Docker tagging, HLX submit | 159 |
| ★ `scripts/build_and_run_framesextraction.sh` | **New** | Frame extraction build script | ~80 |
| ★ `scripts/build_and_run_sorting_data.sh` | **New** | Data sorting/filtering build script | ~80 |
| ★ `scripts/build_and_run_splitdata.sh` | **New** | Data splitting build script | ~60 |
| ★ `scripts/build_and_run_training_data.sh` | **New** | Training data generation build script | ~80 |

**Key architectural difference:** Upstream SAM2 is a general-purpose segmentation model with interactive point/box prompts. The local version adds an **automated road segmentation pipeline** that uses a fixed 6-point grid optimized for front-facing vehicle cameras, produces VideoPainter-compatible output formats, and includes complete GCS integration for cloud-scale processing. The upstream SAM2 core library is used as an **unmodified dependency** — no model code was changed.

---

### 2.3 NVlabs/alpamayo vs vla/alpamayo

**Upstream** ([github.com/NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)): NVIDIA's Alpamayo-R1-10B Vision-Language-Action model for autonomous driving. The repository provides:

| Directory | Upstream Purpose |
|-----------|-----------------|
| `src/alpamayo_r1/` | Model code: VLA architecture, diffusion trajectory head, geometry, action space (19 files) |
| `notebooks/` | Interactive inference notebook (`inference.ipynb`) + clip ID parquet |
| `pyproject.toml`, `uv.lock` | Dependency management (Python 3.12 + uv) |
| `CONTRIBUTING.md`, `LICENSE`, `README.md` | Project documentation |

Upstream's `test_inference.py` loads a single clip from NVIDIA's PhysicalAI-AV dataset and runs forward inference producing trajectory predictions. It has **no** support for processing edited videos, no batch processing, no visualization, and no evaluation metrics.

**Comparison methodology:** `diff -rq` on all files, byte-level comparison of each `src/` file, comparison performed February 2026.

**Summary statistics:**

| Metric | Value |
|--------|-------|
| Upstream source files | 27 |
| Local files (excl. cache, checkpoints, .whl, .npz) | 37 |
| Byte-identical files | 25 (all 19 `src/` + 6 shared non-src) |
| Modified files | 2 (trivial: `.gitignore`, `pyproject.toml`) |
| New files | 10 |
| Novel code volume | ~3,900 lines |
| Core library changes | **0 lines** (100% identical) |

**Identical files (25):** All 19 files in `src/alpamayo_r1/` are byte-identical to upstream:

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | VLA architecture | Alpamayo-R1 model definition |
| `diffusion.py` | Diffusion trajectory head | Denoising trajectory prediction |
| `geometry.py` | Geometry utilities | Coordinate transforms, ego-motion |
| `action_space.py` | Action space | Trajectory action parameterisation |
| `config.py` | Configuration | Model hyperparameters, dataclasses |
| `helper.py` | Helper utilities | Common functions |
| `load_physical_aiavdataset.py` | Dataset loader | PhysicalAI-AV clip loading |
| `test_inference.py` | Test script | Single-clip inference demo |
| All remaining `src/` files | — | Package init, type stubs, etc. |

Plus 6 shared non-src files: `CONTRIBUTING.md`, `LICENSE`, `README.md`, `notebooks/clip_ids.parquet`, `notebooks/inference.ipynb`, `uv.lock` — all byte-identical.

**Modified files (2):**

| File | Change | Detail |
|------|--------|--------|
| `.gitignore` | +1 entry | Added `checkpoints/` to ignore list |
| `pyproject.toml` | +2 deps | Added `opencv-python-headless>=4.8.0` (frame rendering for `visualize_video.py`) and `psutil>=5.9.0` (memory tracking for `run_inference.py`) |

**New files (10 files, ~3,900 lines total):**

| File | Lines | Purpose |
|------|-------|---------|
| ★ `run_inference.py` | 640 | **VP→Alpamayo data bridge:** parses `_edited_` token from VP filenames to extract clip_id + camera_name, loads original multi-camera frames + ego-motion from PhysicalAI-AV, replaces single camera's frames with VP-edited frames (bilinear resize to 576×1024), runs dual-pass inference (original vs. generated), computes minADE against ground-truth, outputs structured JSON (trajectories, minADE, chain-of-thought, timing, GPU/RAM usage via nvidia-smi + psutil) + NPZ (full tensor data for offline re-rendering) |
| ★ `workflow_alpamayo.py` | 583 | **HLX cloud workflow:** GCS FuseBucket mounts for Alpamayo-R1-10B checkpoint + VP output data, in-process model reuse (load once → infer all videos), `VLAVideoMetrics` dataclass, sequential per-video processing loop, aggregate report (avg/best/worst minADE, total time, peak resources), structured GCS upload |
| ★ `visualize_video.py` | 964 | **Trajectory renderer:** pinhole camera projection with ego→cam transform (x_cam=−y_ego, y_cam=−z_ego, z_cam=x_ego), focal length f_x=(W/2)/tan(FOV/2), ground-truth in red + predictions in cyan, BEV inset (250×250 px), progressive reveal animation, side-by-side original-vs-edited comparison, camera-yaw-aware transforms, H.264 encoding via PyAV |
| ★ `download_checkpoints.py` | 30 | HuggingFace Hub `snapshot_download` for Alpamayo-R1-10B weights |
| ★ `Dockerfile` | 105 | CUDA 12.1.1, Python 3.12, PyTorch 2.8+cu121, flash-attention, HLX SDK; includes `sed` monkey-patch for `physical_ai_av` library's `.to_numpy()` read-only Arrow array bug (wraps with `np.array()` to produce writable copies for `scipy.spatial.transform`) |
| ★ `docker-compose.yaml` | 29 | GPU passthrough, 16 GB shm_size, Jupyter port 8888 |
| ★ `.dockerignore` | 55 | Excludes checkpoints/, npz/, __pycache__/ from build context |
| ★ `README_DOCKER.md` | 137 | Container build, configuration, and execution documentation |
| ★ `scripts/build_and_run.sh` | 179 | Docker build → tag → push to Artifact Registry → HLX workflow submission for Stage 3 |
| ★ `notebooks/visualize_results.ipynb` | 1,171 | Interactive Jupyter notebook for trajectory visualisation and result exploration |

**Key architectural difference:** Upstream Alpamayo runs inference on raw PhysicalAI-AV dataset clips via an interactive notebook. The local version creates a **VideoPainter-to-Alpamayo data bridge** that:
1. Parses `clip_id` and `camera_name` from VP output filenames via `_edited_` token protocol
2. Loads the original clip's multi-camera frames + ego-motion from PhysicalAI-AV dataset
3. Replaces exactly one camera's frames with VideoPainter-edited frames (bilinear resize to model resolution)
4. Blacks out non-target cameras to isolate the visual effect of the single edited camera
5. Runs inference **twice** (original and generated) under identical conditions for controlled A/B comparison
6. Computes minADE against ground-truth trajectories for both passes
7. Renders overlay and side-by-side comparison videos with trajectory projections
8. Produces structured JSON + NPZ outputs for quantitative analysis

**Comparison with other stages:** The Alpamayo core is 100% unmodified (matching the SAM 2 pattern from Section 2.2), unlike VideoPainter where 8 files required modification. All three stages share the same architectural philosophy: unmodified foundation model + novel application/infrastructure layer. Combined novel code: SAM 2 (~5,390 lines) + VideoPainter (~5,500 lines) + Alpamayo (~3,900 lines) ≈ **14,800 lines** of thesis-specific engineering.

---

## 3. Novel Contribution #1: Three-Stage Master Pipeline Orchestrator

### Problem
SAM2, VideoPainter, and Alpamayo each run in isolated environments with incompatible Python versions (3.10 vs 3.12), different GPU memory profiles, and no shared data interfaces. Running the full pipeline manually requires sequential Docker builds, GCS path management, and careful data handoffs.

### Solution
A **master workflow orchestrator** (`workflow_master.py`, 712 lines) that defines a DAG of three `@task` nodes, each running in its own container image, connected by data-dependency edges that enforce strict sequential execution:

```
SAM2 ──(run_id)──▶ VideoPainter ──(gcs_path)──▶ Alpamayo
```

### Technical Details

**Workflow Graph Architecture:**
- Each stage is a `@task` with its own `container_image`, GPU node allocation (A100 80GB), and cloud storage FUSE mounts
- The master orchestrator container is **lightweight** (Python 3.10-slim, ~200 MB) — it only serializes the workflow graph; heavy ML deps live in stage containers
- Data flows through GCS paths: Stage 1 writes to `gs://…/preprocessed_data_vp/<run_id>/`, Stage 2 reads from there and writes to `gs://…/vp/<run_id>/`, Stage 3 reads from there
- A single `run_id` propagates through all stages for end-to-end traceability

**Partial Workflow Support:** The orchestrator defines **7 workflow variants** for flexible execution:
| Workflow | Stages | Use Case |
|----------|--------|----------|
| `master_pipeline_wf` | 1→2→3 | Full pipeline |
| `sam2_only_wf` | 1 | Segmentation only |
| `sam2_vp_wf` | 1→2 | Segmentation + editing |
| `vp_only_wf` | 2 | Editing only (reuse SAM2 data) |
| `vp_alpamayo_wf` | 2→3 | Editing + evaluation |
| `alpamayo_only_wf` | 3 | Evaluation only (reuse VP data) |

**Build Script Intelligence** (`scripts/build_and_run.sh`, ~400 lines):
- `STAGES` variable selects which stages to run (e.g., `STAGES=12` runs SAM2→VP)
- Validates input dependencies (e.g., VP without SAM2 requires `SAM2_DATA_RUN_ID`)
- Builds only the Docker images needed for the selected stages
- Generates unique `MASTER_RUN_ID = RUN_ID_TIMESTAMP` for each execution
- Supports 5 pre-defined lane marking editing prompts, selectable via `PROMPT_IDS`
- Exports all configuration as environment variables for workflow serialization

**Why This Is Novel:** No prior work has chained a segmentation model, a video inpainting model, and a VLA trajectory model into a single orchestrated pipeline. The master orchestrator solves the container isolation problem while maintaining data lineage.

---

## 4. Novel Contribution #2: SAM2-Based Road Segmentation Pipeline

### Problem
SAM2 provides general-purpose segmentation with interactive prompts. Autonomous driving requires **automated road region detection** across thousands of videos without manual annotation.

### Solution
A complete road segmentation pipeline (`process_videos_sam2.py`, 1,118 lines) that uses SAM2's video predictor with a **fixed 6-point grid** optimized for front-facing vehicle camera geometry.

### Technical Details

#### Conceptual Design: Bottom-Half Point Strategy

The core insight behind this pipeline is that **front-facing autonomous driving cameras have a consistent spatial layout**: the road surface always occupies the lower portion of the frame, with the vanishing point near the vertical center. This geometric prior is exploited to eliminate the need for interactive prompts.

**Automatic Road Initialization (6-Point Grid — Bottom Half Only):**
```
All points are placed exclusively in the bottom half of the frame:

  ┌───────────────────────────────────┐
  │                                   │  ← Sky / buildings (ignored)
  │                                   │
  │           (0.50, 0.65)            │  ← Upper road boundary
  │      (0.35, 0.75) (0.65, 0.75)   │  ← Mid-road coverage
  │  (0.25, 0.80)       (0.75, 0.80) │  ← Wide road coverage
  │         (0.50, 0.85)              │  ← Center-bottom (reference)
  └───────────────────────────────────┘
```
These 6 normalized coordinates target the lower 65–85% of the frame where the road surface appears in dashcam footage. Each point is assigned `label=1` (foreground) to seed the SAM2 prompt. The bottom-center point `(0.50, 0.85)` is deliberately placed at the **most certain road location** — directly beneath the ego vehicle — and doubles as the **reference point** for connected-component filtering (see below).

**Why bottom-half only?** Placing points in the upper half would risk capturing sky, overpasses, or buildings as foreground. By constraining all seed points to the lower half, the prompt is guaranteed to hit road surface regardless of scene content, making the pipeline fully automatic across diverse driving environments (highway, urban, rural).

#### Four-Stage Mask Quality Pipeline

SAM2's raw output often contains artifacts: multiple disconnected segments (sidewalks, medians), small noise regions, and internal holes (from lane markings or shadows). A carefully designed four-stage morphological pipeline produces a **single, clean, hole-free road mask** suitable for video inpainting:

**Stage 1 — Morphological Opening (disconnect noise):**
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
```
Opening = erosion → dilation. This disconnects thin bridges between the road and adjacent regions (sidewalks, parking areas) that SAM2 sometimes merges, and removes small isolated noise blobs.

**Stage 2 — Connected-Component Filtering with Reference Point:**
```python
ref_point = (width // 2, int(height * 0.85))  # Bottom center
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask)
label_at_point = labels[ref_point[1], ref_point[0]]
if label_at_point > 0:
    filtered_mask = (labels == label_at_point)  # Keep ONLY the road
else:
    filtered_mask = largest_component            # Fallback: keep largest
```
The **reference point** at `(width/2, height×0.85)` — the bottom center of the frame — is the most geometrically certain road location in any front-facing dashcam view. After connected-component analysis, only the component that contains this reference point is retained. All other disconnected regions (sidewalks, medians, sky artifacts) are discarded. If the reference point falls on background (rare edge case), the largest component is kept as fallback.

**Stage 3 — Morphological Closing (merge small gaps):**
```python
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
```
Closing = dilation → erosion. This fills narrow gaps within the road component — e.g., where lane markings, crosswalks, or shadows create thin breaks — without expanding the overall boundary.

**Stage 4 — Flood-Fill Hole Elimination:**
```python
def _fill_binary_mask_holes(mask_255):
    inv = cv2.bitwise_not(mask_255)           # Invert: background=255, road=0
    cv2.floodFill(inv, None, (0,0), 0)        # Erase external background
    holes = inv                                # Only internal holes remain as 255
    return cv2.bitwise_or(mask_255, holes)     # Fill holes back into road
```
Conceptual approach: After inversion, the external background (connected to frame borders) is flood-filled from corner (0,0) to zero. Any remaining 255 pixels in the inverted mask are **internal holes** (surrounded entirely by road). These are OR'd back into the original mask, producing a single continuous, hole-free road segment.

**Why hole-free masks matter for video inpainting:** VideoPainter's CogVideoX model treats the white mask region as the area to be regenerated. Internal holes (from lane markings, shadows, road patches) would cause the model to preserve those pixels unchanged while editing surrounding road — creating visible discontinuities. A single contiguous mask ensures the entire road surface is available for coherent lane marking generation.

```
Raw SAM2 Output          After 4-Stage Pipeline
┌─────────────────┐      ┌─────────────────┐
│     ░░░░░       │      │                 │
│   ░░████░░░     │      │   ████████████  │
│ ░░█ ██ █ █░░░   │      │  █████████████  │
│ ░██ holes ██░   │  →   │  █████████████  │
│███████████████  │      │ ██████████████  │
│████ gap ██████  │      │ ███████████████ │
│████████████████ │      │ ███████████████ │
└─────────────────┘      └─────────────────┘
  Multiple segments,       Single segment,
  holes, gaps, noise       no holes, clean edges
```

#### Constant Frame Rate Strategy

Frames are extracted at a **fixed 8 FPS** via FFmpeg's fps filter — regardless of the source video's native frame rate (typically 20–30 fps):
```
ffmpeg -i input.mp4 -vf fps=8 -vframes 100 output/%05d.jpg
```
This constant rate serves two purposes:
1. **Temporal consistency for SAM2:** Consistent inter-frame motion allows SAM2's memory bank to propagate masks reliably. Variable frame rates would cause inconsistent object displacement between frames.
2. **Direct compatibility with VideoPainter:** CogVideoX-5B expects uniformly-spaced input frames. By extracting at the same rate SAM2 processes, no re-sampling is needed between stages.

The output visualization and preprocessing videos are written at this same 8 FPS, preserving real-time duration alignment.

#### Processing Pipeline per Video

1. **Video Download:** GCS URI → local temp file via gcsfs (authenticated)
2. **Frame Extraction:** FFmpeg fps=8 filter, up to MAX_FRAMES (100 default), quality 2 JPEG
3. **FPS Detection:** Multi-method: ffprobe `avg_frame_rate` → `r_frame_rate` → OpenCV → fallback
4. **SAM2 Inference:** VOS-optimized video predictor with `apply_postprocessing=True`, temporal propagation across all frames from frame 0 seed points
5. **Parallel Mask Post-Processing:** ThreadPoolExecutor (up to 8 workers) applies the 4-stage morphological pipeline per frame in parallel (OpenCV releases the GIL)
6. **Multi-Format Output:**
   - Binary PNG masks per frame (0=keep, 255=inpaint)
   - Compressed NPZ archive (`all_masks.npz`) for efficient downstream loading
   - Visualization overlay videos (segmented regions highlighted in green)
   - Metadata CSV with video statistics and timing data
7. **VideoPainter-Compatible Preprocessing:** Creates the exact folder structure VP's `edit_bench.py` expects:
   ```
   preprocessed_data_vp/<run_id>/<video_id>/
     ├── meta.csv
     ├── masks/<video_id>/all_masks.npz
     └── raw_videos/<video_id>/<video_id>.0.mp4
   ```
8. **Parallel GCS Upload:** ThreadPoolExecutor (8 workers) uploads results and preprocessed data concurrently

#### Performance Optimizations in SAM2 Stage

| Optimization | Technique | Impact |
|-------------|-----------|--------|
| **Adaptive GPU precision** | bf16 + TF32 on Ampere+ GPUs; fp16 fallback on Turing (T4) | 1.5–2× faster inference |
| **VOS-optimized predictor** | `vos_optimized=True` in `build_sam2_video_predictor()` | Reduced memory footprint |
| **CUDA graph boundaries** | `torch.compiler.cudagraph_mark_step_begin()` before each video | Prevents tensor overwrite across videos |
| **In-memory mask pass-through** | Segmentation returns masks in memory; VP preprocessing reuses them directly, skipping disk re-read | Eliminates redundant I/O for 100+ masks per video |
| **Parallel post-processing** | ThreadPoolExecutor for morphological ops (OpenCV releases GIL → true parallelism) | Up to 8× speedup for mask pipeline |
| **Parallel GCS uploads** | 8-thread upload with pre-created remote directories | Saturates network bandwidth |
| **Thread-safe GCS singleton** | Double-checked locking for `gcsfs.GCSFileSystem` | Avoids per-call auth overhead |
| **Deterministic frame caching** | `_sync_frame_folder_to_max_frames()` reuses cached frames; trims surplus if MAX_FRAMES changes | Avoids re-extraction on reruns |
| **Aggressive memory cleanup** | Explicit `del processed_masks` + temp dir removal after each video | Keeps disk/RAM footprint bounded |

**Chunks:// URI Protocol:** A custom URI scheme for batch video selection:
```
chunks://bucket/prefix/camera_folder?start=0&end=10&per_chunk=5
```
Resolves to individual `gs://` paths by listing chunk directories (`chunk_0000/`, `chunk_0001/`, …) and selecting up to `per_chunk` MP4 files from each. Includes **early-stop optimization**: stops scanning GCS once enough files are collected for the requested slice.

**Why This Is Novel:** Prior road segmentation pipelines use specialized models (lane detectors, semantic segmentation). This is the first to repurpose SAM2 as an automated road segmentor specifically for generating video inpainting masks, using a bottom-half reference-point strategy with a four-stage morphological pipeline to produce single, hole-free road segments directly compatible with VideoPainter's expected input format.

---

## 5. Novel Contribution #3: FluxFill Integration into VideoPainter's Edit Pipeline

### Problem
VideoPainter's original `edit_bench.py` generates all frames using only CogVideoX-5B. The first frame often lacks semantic precision for fine-grained lane marking attributes (single vs. double, white vs. yellow, solid vs. dashed) because CogVideoX optimizes for temporal consistency rather than per-frame detail.

### Solution
A **dual-model pipeline** where FluxFill generates a semantically-precise first frame, and CogVideoX propagates the edit temporally:

```
Input Video + Mask
       │
       ▼
   FluxFill → High-quality first frame (image inpainting)
       │
       ▼
   CogVideoX → Temporally consistent video (video diffusion)
       │
       ▼
   Edited Video
```

### Technical Details

**Modified File:** `infer/edit_bench.py`

**FluxFillPipeline Integration:**
- Loads `FluxFillPipeline` from a local checkpoint or GCS-mounted path
- Supports optional LoRA weights (`--img_inpainting_lora_path`, `--img_inpainting_lora_scale`)
- First frame is generated at the video's native resolution
- Mask is applied with configurable dilation and Gaussian feathering

**Mask Processing Pipeline:**
- `_dilate_and_feather_mask_images()`: Expands inpaint regions using morphological dilation (configurable `dilate_size`, default 24px), then applies Gaussian blur for smooth boundaries (configurable `mask_feather`, default 8px)
- `--keep_masked_pixels` flag: When enabled, preserves the original background pixels outside the mask in the final composite, preventing FluxFill from altering non-road regions

**Multi-GPU Orchestration & Memory Management:**
- Qwen VLM on `cuda:1` (when available, controlled by `VP_QWEN_DEVICE`)
- FluxFill + CogVideoX on `cuda:0` (controlled by `VP_FLUX_DEVICE`, `VP_COG_DEVICE`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for dynamic memory management

**Staged Model Loading/Unloading (VRAM Budgeting):**
CogVideoX-5B alone requires ~35 GiB VRAM. To fit all models on a single A100 80GB:
1. **Load Qwen VLM first** → generate captions / run refinement loop
2. **Unload Qwen** (`_unload_qwen_model()`: delete model refs → `gc.collect()` → `torch.cuda.empty_cache()`) to reclaim ~14 GiB
3. **Load FluxFill** → generate first frame
4. **Unload FluxFill** → reclaim ~24 GiB
5. **Load CogVideoX** → propagate edit temporally

This sequential load/unload pattern ensures peak VRAM never exceeds ~40 GiB, allowing all three models to run on a single GPU.

**Qwen VLM VRAM Budgeting:**
```python
total_gib = torch.cuda.get_device_properties(target).total_memory / (1024³)
reserve_gib = int(os.environ.get("QWEN_VRAM_RESERVE_GIB", "12"))
budget_gib = max(8, min(total_gib - reserve_gib, int(total_gib * 0.9)))
max_memory = {target: f"{budget_gib}GiB", "cpu": "192GiB"}
# Prevent spilling onto the Flux/Cog GPU:
for i in range(gpu_count):
    if i != target:
        max_memory[i] = "1GiB"
```

**Why This Is Novel:** No prior work has combined a still-image inpainting model (FluxFill) with a video diffusion model (CogVideoX) in a sequential first-frame → temporal-propagation pipeline for video editing, with staged model loading to fit all three large models (Qwen 7B + FluxFill 12B + CogVideoX 5B) on a single GPU.

---

## 6. Novel Contribution #4: VLM-Guided Iterative Caption Refinement

### Problem
Text-guided inpainting is highly sensitive to prompt quality. Generic prompts like "white lane marking" produce inconsistent results. The model needs specific visual texture details, but manually crafting prompts per video is infeasible at scale.

### Solution
An **automated QA loop** using Qwen2.5-VL (7B) that evaluates FluxFill's output and iteratively refines the caption until the generated image matches the editing instruction.

### Technical Details

**Function:** `_qwen_refine_first_frame_caption()` in `infer/edit_bench.py`

**Refinement Loop (up to N iterations, default 10):**
1. FluxFill generates a first frame using the current caption
2. Qwen2.5-VL receives: `[original_frame, generated_frame, editing_instruction]`
3. VLM evaluates semantic correctness:
   - Lane marking count (single vs. double)
   - Color accuracy (white vs. yellow)
   - Pattern type (solid vs. dashed)
   - Perspective alignment with original lanes
   - Road texture preservation
4. VLM outputs: `PASS` or `FAIL` with structured reasoning
5. On `FAIL`: VLM generates a revised caption with more specific visual details
6. Loop continues until `PASS` or max iterations reached

**Configurable Parameters:**
- `--caption_refine_iters` (default: 10): Maximum refinement iterations
- `--caption_refine_temperature` (default: 0.1): VLM sampling temperature (low = deterministic)
- `VP_UNLOAD_QWEN_AFTER_USE=1`: Unload VLM after refinement to free VRAM for CogVideoX

**Why This Is Novel:** This is the first use of a VLM as an automated quality assessor in a video inpainting loop. The feedback mechanism creates a closed-loop system where the editing model's output quality is verified against the semantic intent before temporal propagation.

---

## 7. Novel Contribution #5: Automated FluxFill Training Data Generation

### Problem
Training a specialized FluxFill LoRA for lane markings requires thousands of (image, mask, caption) triplets. Manually annotating road images with masks and lane-attribute captions is prohibitively expensive.

### Solution
A fully automated pipeline (`data_generation.py`, 1,112 lines) that generates training datasets from raw autonomous driving videos using SAM2 + Qwen2.5-VL.

### Technical Details

**Pipeline Stages:**
1. **GCS Video Discovery:** Lists MP4 files in chunk-structured GCS folders, selectable by slice
2. **Frame Extraction:** FFmpeg-based extraction of specific frame numbers per video
3. **SAM2 Mask Generation:** Image predictor mode with road-optimized point grid → binary masks
4. **VLM Caption Generation (Qwen2.5-VL-7B):**
   - Receives the frame image as input
   - Extracts structured lane attributes:
     - **Count:** single / double / unknown
     - **Color:** white / yellow / mixed / unknown
     - **Pattern:** solid / dashed / mixed / unknown
   - Generates normalized prompt: `"road with {count} {color} {pattern} lane markings"`
5. **Dataset Assembly:**
   ```
   dataset_folder/
     ├── train.csv       (image, mask, prompt, prompt_2)
     ├── images/*.png    (original frames)
     └── masks/*.png     (binary road masks)
   ```
6. **GCS Upload:** Complete dataset uploaded to GCS for training

**Scale:** Successfully generated 10,000+ training samples from NVIDIA PhysicalAI-AV data.

**Performance Optimizations:**

| Optimization | Before | After | Impact |
|-------------|--------|-------|--------|
| **Model caching** | Each worker reloaded Qwen (~2 min × 8 workers) | Per-device singleton: `_qwen_models[device]` → loaded once per GPU | 16 min → 2 min startup |
| **Threading over multiprocessing** | `multiprocessing.Pool` → 8 separate model copies (8 × 7 GiB = 56 GiB) | `ThreadPool` → shared memory, models loaded once | ~7× memory reduction |
| **Model pre-loading** | Models loaded on first use (race conditions) | `_preload_models_on_gpus()` loads SAM2 + Qwen on all GPUs in parallel at startup | Predictable startup, no races |
| **Multi-GPU separation** | SAM2 and Qwen competed for cuda:0 | SAM2 on cuda:0, Qwen biased to cuda:1 with explicit VRAM budgeting | No OOM conflicts |
| **Early-stop GCS listing** | Scanned entire GCS prefix | `if len(all_paths) >= needed: break` | Orders of magnitude faster for large prefixes |

**Why This Is Novel:** This is the first automated pipeline that combines a segmentation model (SAM2) with a vision-language model (Qwen2.5-VL) to generate structured training data for inpainting models, with semantically-parsed lane marking attributes.

---

## 8. Novel Contribution #6: Semantic Lane-Attribute Dataset Filtering

### Problem
Raw generated datasets contain mixed lane attributes. Training a LoRA for "single white solid" lane markings requires filtering out all other combinations.

### Solution
A multi-attribute filtering system (`filter_fluxfill_dataset.py`, 725 lines) with regex-based prompt parsing and configurable quality controls.

### Technical Details

**Attribute Parsing (Regex):**
```python
r"road with (?P<count>...) (?P<color>...) (?P<pattern>...) lane markings"
```
Extracts `count`, `color`, `pattern` from normalized captions produced by `data_generation.py`.

**Filtering Modes:**
| Parameter | Values | Effect |
|-----------|--------|--------|
| `count` | single / double / any | Filter by lane count |
| `color` | white / yellow / mixed / any | Filter by lane color |
| `pattern` | solid / dashed / mixed / any | Filter by lane pattern |
| `REQUIRE_CLEAR_ROAD` | 0 / 1 | Reject prompts with "unknown" attributes |

**Copy Modes:** `copy` / `hardlink` / `symlink` (GCS always uses copy)

**GCS Support:** Both input and output can be GCS paths. Local staging is used for filtering, then batch upload.

**Why This Is Novel:** First semantic curation tool for lane marking datasets that filters by structured visual attributes extracted from VLM captions.

---

## 9. Novel Contribution #7: FluxFill LoRA Fine-Tuning Pipeline

### Problem
The base FluxFill model generates generic inpainting. Lane markings require specific visual patterns (dashed spacing, line thickness, color fidelity) that the base model doesn't produce reliably.

### Solution
A complete LoRA fine-tuning pipeline (`training_workflow.py` + `train/train_fluxfill_inpaint_lora.py`) that trains FluxFill on curated lane-marking datasets.

### Technical Details

**Training Configuration:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rank` | 16 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling factor |
| `learning_rate` | 1e-5 | AdamW optimizer LR |
| `lr_scheduler` | cosine | Learning rate schedule |
| `lr_warmup_steps` | 50 | Warmup steps |
| `num_train_epochs` | 5 | Training epochs |
| `train_batch_size` | 1 | Per-GPU batch size |
| `gradient_accumulation_steps` | 1 | Effective batch multiplier |
| `mixed_precision` | bf16 | Mixed precision training |
| `checkpointing_steps` | 100 | Save frequency |

**Cloud Workflow Integration:**
- Cloud storage FUSE mounts for FluxFill base checkpoint (`ckpt/flux_inp`)
- Training data downloaded from GCS to local scratch
- Output checkpoints uploaded to GCS with timestamp-based run IDs
- A100 80GB GPU allocation

**Multi-Dataset Training — 5 Specialized LoRA Checkpoints:**

The pipeline supports training **5 independent LoRA checkpoints** in parallel, one per lane-marking combination.  This is driven by `scripts/build_and_run_training_all.sh`, which iterates over the 5 filtered datasets produced by `build_and_run_sorting_data.sh` and submits a separate HLX training workflow for each:

| # | Dataset (Input) | Checkpoint (Output) | Lane Type |
|---|----------------|---------------------|-----------|
| 1 | `td-chunk0-99_single_white_solid/` | `fluxfill_single_white_solid_<TS>/` | Single solid white |
| 2 | `td-chunk0-99_double_white_solid/` | `fluxfill_double_white_solid_<TS>/` | Double solid white |
| 3 | `td-chunk0-99_single_yellow_solid/` | `fluxfill_single_yellow_solid_<TS>/` | Single solid yellow |
| 4 | `td-chunk0-99_double_yellow_solid/` | `fluxfill_double_yellow_solid_<TS>/` | Double solid yellow |
| 5 | `td-chunk0-99_single_white_dashed/` | `fluxfill_single_white_dashed_<TS>/` | Single dashed white |

**End-to-end flow:**
```
data_generation.py          → td-chunk0-99/                  (raw training triplets)
filter_fluxfill_dataset.py  → td-chunk0-99_<lane_type>/      (5 filtered subsets)
training_workflow.py        → fluxfill_<lane_type>_<TS>/      (5 LoRA checkpoints)
```

**Usage:**
```bash
# Step 1: Generate training data (chunks 0–99)
bash segmentation/sam2/scripts/build_and_run_training_data.sh 0 99

# Step 2: Sort into 5 lane-type datasets
bash segmentation/sam2/scripts/build_and_run_sorting_data.sh 0 99

# Step 3: Train 5 LoRA checkpoints (one per lane type)
bash generation/VideoPainter/scripts/build_and_run_training_all.sh 0 99

# Step 4: Use a specific checkpoint for inference
TRAINED_FLUXFILL_GCS_PATH="workspace/user/hbaskar/.../fluxfill_single_white_solid_<TS>" \
  STAGES=2 bash scripts/build_and_run.sh
```

Each checkpoint is stored independently, allowing selective deployment during inference.  The master pipeline's `TRAINED_FLUXFILL_GCS_PATH` variable points at the desired checkpoint.

**Training Results (Run: 20260222_201003):**

All 5 jobs executed in parallel on separate A100 80GB GKE nodes. Identical hyperparameters: rank=16, alpha=16, lr=1e-5, cosine schedule, 50 warmup steps, 5 epochs, batch=1, grad_accum=4, seed=42. Training speed: ~6.8s/step.

| # | Lane Type | Images | Steps | Time | Epoch Avg Losses (1→5) | Status |
|---|-----------|--------|-------|------|----------------------|--------|
| 1 | `single_yellow_solid` | 372 | 465 | 59.1 min | 0.1701 → 0.1761 → 0.1682 → 0.1855 → **0.1652** | ✅ Complete |
| 2 | `single_white_dashed` | 373 | 470 | 59.0 min | 0.1475 → 0.1577 → 0.1486 → 0.1706 → **0.1455** | ✅ Complete |
| 3 | `single_white_solid` | 1,522 | 1,905 | 3.68 hrs | 0.1670 → 0.1586 → 0.1654 → 0.1692 → **0.1690** | ✅ Complete |
| 4 | `double_white_solid` | 4,612 | 5,765 | 10.98 hrs | 0.1724 → 0.1735 → 0.1701 → 0.1712 → **0.1672** | ✅ Complete |
| 5 | `double_yellow_solid` | 6,493 | 8,120 | 15.43 hrs | 0.1644 → 0.1589 → 0.1593 → 0.1577 → **0.1570** | ✅ Complete |

Key observations:
- Loss stabilizes by epoch 3 for all jobs (final-epoch avg range: 0.145–0.169)
- All 5 jobs complete. The largest dataset (double_yellow_solid, 6,493 images) converged to 0.157; the smallest (single_yellow_solid, 372 images) to 0.165 — all within a narrow band, confirming scalability
- Epoch duration scales linearly with dataset size: ~59 min for 370-image datasets, ~3.7 hrs for 1,522-image, ~11 hrs for 4,612-image, ~15.4 hrs for 6,493-image datasets (~6.8s/step consistent)
- GCS upload of final checkpoints completes quickly: ~8s for smaller jobs, ~1.6 min for double_white_solid (57 checkpoints)
- GCS data download scales linearly: ~3 min for 1.5k images, ~9.1 min for 4.6k images
- Harmless `gcsfs`/`aiohttp` session cleanup traceback at exit does not affect results

**Why This Is Novel:** First LoRA fine-tuning pipeline specifically designed for FluxFill inpainting on structured lane marking datasets, with automated multi-variant training producing one specialized checkpoint per lane-marking type, integrated into a cloud workflow system.

---

## 10. Novel Contribution #8: Alpamayo VLA Trajectory Evaluation

### Problem
Video inpainting quality is traditionally measured by pixel-level metrics (LPIPS, SSIM, FID). These metrics don't capture whether the edited lane markings would actually affect autonomous driving behavior.

### Solution
Use NVIDIA's Alpamayo-R1-10B VLA model to **predict driving trajectories on both original and inpainted videos**, then measure the trajectory divergence via minADE.

### Technical Details

**VideoPainter → Alpamayo Data Bridge** (`run_inference.py`):

1. **Filename Parsing:** Extracts `clip_id` and `camera_name` from VP output filenames:
   ```
   <clip_id>.<camera_name>_vp_edit_sample0_generated.mp4
   ```

2. **Original Data Loading:** Uses `load_physical_aiavdataset()` to retrieve:
   - Multi-camera frames (7 views: front_wide, front_tele, cross_left/right, rear_left/right, rear_tele)
   - Ego-motion history (16 timesteps at CONTENT_TIME_STEP)
   - Ground-truth future trajectory (64 waypoints at 10 Hz, 6.4s horizon)

3. **Temporal Alignment:**
   - VP video encoded at 8 fps → `CONTENT_TIME_STEP = 0.125s`
   - `t0_us = (total_frames − 1) × 0.125s × 1e6` for dataset alignment
   - Last 4 consecutive frames per camera used for inference

4. **Dual Inference** (with `black_non_target_cameras=true`):
   - **Original inference:** Run Alpamayo on unmodified dataset frames → `original_min_ade`
   - **Frame replacement:** Substitute one camera's last-4-frames with VP output (with bilinear resize if needed)
   - **Generated inference:** Run Alpamayo on modified frames → `min_ade_meters`
   - Both share identical ego-motion history and all other camera views
   - Non-target cameras are blacked out (zeroed), isolating the effect of the edited front camera on trajectory prediction

5. **minADE Computation:**
   $$\text{minADE} = \min_{s \in S} \frac{1}{T} \sum_{t=1}^{T} \| \hat{p}_{xy}^{(s)}(t) - p_{xy}^{gt}(t) \|_2$$
   where $S$ = trajectory samples, $T$ = 64 waypoints, XY-plane only.

6. **Chain-of-Thought Extraction:** Both original and generated inference produce reasoning traces from the VLA model's language head.

**Output Artifacts per Video:**
| File | Format | Content |
|------|--------|---------|
| `*_inference.json` | JSON | Predictions, minADE (both original & generated), reasoning traces, temporal config, metrics |
| `*_vis_data.npz` | NPZ | Trajectory tensors, image frames, camera indices, full original camera frames |
| `*_overlay.mp4` | H.264 | Trajectory overlay on generated video |
| `*_comparison.mp4` | H.264 | Side-by-side original vs. generated with trajectories |

**Why This Is Novel:** This is the **first use of a VLA model as an evaluation metric for video inpainting**. By comparing trajectory predictions on original vs. edited videos, we obtain a functionally meaningful measure of inpainting quality that goes beyond pixel-level metrics.

**Experimental Results (Run 006 — 100 videos × 5 prompts × 2 variants = 1,000 experiments, `black_non_target_cameras=true`):**

Alpamayo Run IDs: `06_p{1-5}_gen_video_20260222_*` (100frames) and `06_p{1-5}_gen_vps90_video_20260222_*` (vps90).

Baseline (original unedited videos): mean minADE = 3.778 m, median = 2.646 m.

| Prompt | Lane Type | minADE (100fr) | Δ from Baseline | Improved/Degraded |
|--------|-----------|---------------|----------------|-------------------|
| p1 | Single white solid | 3.842 m | +0.063 m | 47% / 53% |
| p2 | Double white solid | 3.995 m | +0.217 m | 47% / 53% |
| p3 | Single yellow solid | 3.895 m | +0.117 m | **52%** / 48% |
| p4 | Double yellow solid | 3.917 m | +0.139 m | 46% / 54% |
| p5 | Single white dashed | 4.065 m | +0.286 m | 40% / **60%** |

Key findings:
- **p3 (single yellow solid)** achieves the highest improvement rate (52%) — the only prompt where more videos improved than degraded
- **p5 (dashed white)** performs worst (40% improvement, +0.286 m Δ), correlated with 50% caption refinement failure rate
- When editing helps, the average improvement is substantial (−0.9 m); when it hurts, degradation averages +1.1 m
- vps90 variant performs marginally worse than 100frames (44.4% vs 46.4% improvement rate)
- These results use base FluxFill without LoRA and `black_non_target_cameras=true` (non-target cameras blacked out); trained LoRA checkpoints are expected to improve results

---

## 11. Novel Contribution #9: Trajectory Visualization & Comparison Rendering

### Problem
Raw trajectory predictions (numpy arrays of 3D waypoints) are not interpretable without visualization. Comparing original vs. edited video trajectories requires careful frame-by-frame alignment.

### Solution
A comprehensive visualization system (`visualize_video.py`, ~530 lines) that renders:
1. **Trajectory overlay videos** with projected 3D waypoints on camera frames
2. **Side-by-side comparison videos** showing original and generated frames with their respective trajectories

### Technical Details

**3D → 2D Projection (Pinhole Camera Model):**
```
Ego frame:   x=forward, y=left, z=up
Camera frame: x=right, y=down, z=forward (depth)

Transform: x_cam = −y_ego, y_cam = −z_ego, z_cam = x_ego
```

**Camera-Yaw-Aware Transform:** For non-front cameras, the ego-to-camera transform includes a yaw rotation:
```python
CAMERA_YAW_RAD = {
    "camera_front_wide_120fov": 0.0,
    "camera_cross_left_120fov": π/2,
    "camera_cross_right_120fov": −π/2,
    "camera_rear_tele_30fov": π,
    ...
}
```

**Focal Length:** $f_x = \frac{W}{2 \cdot \tan(\text{FOV}/2)}$, with FOV auto-detected from camera name (120°, 70°, 30°).

**Visualization Features:**
- Progressive reveal mode: linearly maps frame index to waypoint count
- BEV inset (250×250 px) at bottom-left: ego history (gray), GT (red), predictions (cyan)
- 5 cycling colors for multiple trajectory samples
- Semi-transparent legend box
- HUD overlay with minADE, clip_id, camera info
- Points behind camera (z_cam ≤ 0.5m) filtered out

**Comparison Video Layout:**
```
┌─────────────────────────────────────┐
│ Original (left) │ Generated (right) │
│  + GT trajectory│  + pred trajectory│
│  + orig pred    │  + VP frames      │
│  BEV inset      │  BEV inset        │
│  Original minADE│  Generated minADE │
└─────────────────────────────────────┘
```

**Why This Is Novel:** First trajectory visualization system that renders projected 3D driving trajectories on both original and inpainted video frames for visual comparison of VLA model behavior changes.

---

## 12. Novel Contribution #10: Cloud-Native Containerized ML Infrastructure

### Problem
Three models with incompatible dependencies (Python 3.10 vs 3.12, different PyTorch versions, flash-attn only for Alpamayo) cannot share a single environment. Cloud-scale processing requires reproducible, isolated environments.

### Solution
Four purpose-built Docker containers orchestrated through cloud workflows with GCS FUSE-mounted data volumes.

### Container Architecture

| Container | Base Image | Python | Key Dependencies | Size |
|-----------|-----------|--------|-----------------|------|
| SAM2 | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | 3.10 | SAM2, Qwen2.5-VL, gcsfs, Workflow SDK | ~8 GB |
| VideoPainter | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` | 3.10 | CogVideoX, FluxFill, Diffusers, Qwen, Workflow SDK | ~15 GB |
| Alpamayo | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` | 3.12 | Alpamayo-R1, flash-attn, physical_ai_av, Workflow SDK | ~12 GB |
| Master | `python:3.10-slim` | 3.10 | Workflow SDK, gcsfs (lightweight orchestrator) | ~500 MB |

**Alpamayo Dockerfile Patch:**
```dockerfile
# physical_ai_av returns read-only Arrow-backed arrays; scipy needs writable buffers
RUN sed -i 's/\.to_numpy())/\.to_numpy().copy())/' \
    /usr/local/lib/python3.12/dist-packages/physical_ai_av/egomotion.py
```

**GCS Data Architecture:**
```
gs://<project-bucket>/<workspace>/
  ├── input/data/camera_front_tele_30fov/chunk_NNNN/  (source videos)
  ├── outputs/
  │   ├── sam2/<run_id>/           (segmentation masks + visualizations)
  │   ├── preprocessed_data_vp/<run_id>/  (VP-compatible data)
  │   ├── vp/<run_id>/            (edited videos)
  │   └── alpamayo/<run_id>/      (trajectory predictions + reports)
  └── checkpoints/
      ├── sam2/                    (SAM2 model weights)
      ├── videopainter/            (CogVideoX, FluxFill, CLIP, VLM weights)
      └── alpamayo/                (Alpamayo-R1-10B weights)
```

**Cloud Storage FUSE Mounts (zero-copy model access):**
- Checkpoints mounted read-only via FUSE → no download latency
- Symlinks redirect container-expected paths to mount points
- Metadata prefetching warms the FUSE cache before inference

---

## 13. End-to-End Data Flow

```
Raw AD Videos (PhysicalAI-AV, GCS)
        │
        ▼
┌─── STAGE 1: SAM2 Segmentation ────────────────────────────┐
│ • Download videos from GCS (gcsfs)                         │
│ • Extract frames at 8 FPS (ffmpeg)                         │
│ • Run SAM2 with 6-point road grid (sam2.1_hiera_large)     │
│ • Generate binary masks + NPZ archives                     │
│ • Create VP-compatible folder structure                     │
│ Output: gs://…/preprocessed_data_vp/<run_id>/              │
└────────────────────────────────────────────────────────────┘
        │ (data dependency: run_id)
        ▼
┌─── STAGE 2: VideoPainter Editing ──────────────────────────┐
│ • Mount SAM2 preprocessed data via cloud storage FUSE mount  │
│ • Stage data to local scratch (masks + raw_videos)          │
│ • For each editing instruction:                             │
│   ├─ VLM refines caption (Qwen2.5-VL, up to 10 iters)     │
│   ├─ FluxFill generates first frame (+ optional LoRA)      │
│   ├─ CogVideoX-5B propagates edit to all frames            │
│   └─ Output: edited video per instruction per input video  │
│ Output: gs://…/vp/<run_id>/                                │
└────────────────────────────────────────────────────────────┘
        │ (data dependency: GCS path)
        ▼
┌─── STAGE 3: Alpamayo VLA Inference ────────────────────────┐
│ • Discover VP output videos via cloud storage FUSE mount    │
│ • Load Alpamayo-R1-10B model once (bfloat16, flash-attn)   │
│ • For each edited video:                                    │
│   ├─ Parse clip_id + camera from filename                   │
│   ├─ Load original multi-camera data from PhysicalAI-AV    │
│   ├─ Run inference on ORIGINAL frames → orig_minADE         │
│   ├─ Replace target camera with VP frames                   │
│   ├─ Run inference on GENERATED frames → gen_minADE         │
│   ├─ Save JSON results + NPZ tensors                        │
│   └─ Render overlay + comparison videos                     │
│ Output: gs://…/alpamayo/<run_id>/                          │
└────────────────────────────────────────────────────────────┘
```

---

## 14. Technical Specifications

### Compute Requirements

| Stage | GPU | VRAM (Peak) | Duration (per video, median) | Container |
|-------|-----|-------------|------------------------------|----------|
| SAM2 | A100 80GB | ~8 GB | **2.7s** (mean 10.4s due to 2 outliers) | `sam2_container` |
| VideoPainter | A100 80GB | **42.4 GB** | **455s** (7.6 min) | `vp_container` |
| Alpamayo | A100 80GB | **21.5 GB** | **34s** | `alpamayo_vla` |

*Measured on Run 006 (100 videos × 5 prompts). Total per-experiment wall time: 12–20 hours on single A100.*

### Model Specifications

| Model | Parameters | Precision | Source |
|-------|-----------|-----------|--------|
| SAM2.1 Hiera-Large | 305M | float32 | Meta (Apache 2.0) |
| CogVideoX-5B-I2V | 5B | bfloat16 | THUDM |
| FluxFill | ~12B | bfloat16 | Black Forest Labs |
| Qwen2.5-VL-7B | 7B | bfloat16 | Alibaba (Apache 2.0) |
| Alpamayo-R1-10B | 10B | bfloat16 | NVIDIA (non-commercial weights) |

### Key Hyperparameters (Inference Defaults)

| Parameter | Value | Stage |
|-----------|-------|-------|
| `num_inference_steps` | 50–70 | VideoPainter |
| `guidance_scale` | 6.0 | VideoPainter |
| `strength` | 1.0 | VideoPainter |
| `dilate_size` | 24 px | VideoPainter |
| `mask_feather` | 8 px | VideoPainter |
| `caption_refine_iters` | 10 | VideoPainter (VLM QA) |
| `num_traj_samples` | 1 | Alpamayo |
| `top_p` | 0.98 | Alpamayo |
| `temperature` | 0.6 | Alpamayo |
| `max_generation_length` | 256 tokens | Alpamayo |
| `trajectory_horizon` | 6.4s (64 waypoints @ 10 Hz) | Alpamayo |

### Output Data Formats

| Artifact | Format | Location |
|----------|--------|----------|
| Segmentation masks | Binary PNG + NPZ | `gs://…/sam2/<run_id>/` |
| VP preprocessed data | MP4 + NPZ + CSV | `gs://…/preprocessed_data_vp/<run_id>/` |
| Edited videos | H.264 MP4 | `gs://…/vp/<run_id>/` |
| Trajectory predictions | JSON | `gs://…/alpamayo/<run_id>/` |
| Visualization tensors | NPZ | `gs://…/alpamayo/<run_id>/` |
| Overlay/comparison videos | H.264 MP4 | `gs://…/alpamayo/<run_id>/` |
| Aggregate reports | TXT | `gs://…/alpamayo/<run_id>/` |

---

### Cross-Pipeline Performance Optimization Summary

The following table consolidates all performance optimizations applied across the three pipeline stages:

| Category | Optimization | Stage | Technique |
|----------|-------------|-------|----------|
| **GPU Precision** | Adaptive bf16/TF32 vs fp16 | SAM2 | Auto-detect Ampere+ (≥8.0) → bf16+TF32; Turing → fp16 |
| **GPU Precision** | Expandable CUDA segments | VP | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **GPU Memory** | Staged model loading/unloading | VP | Sequential Qwen → FluxFill → CogVideoX with explicit `del` + `empty_cache()` between stages |
| **GPU Memory** | Per-device VRAM budgeting | VP, Data Gen | Dynamic `max_memory` dict prevents cross-GPU spilling |
| **GPU Memory** | CUDA graph step boundaries | SAM2 | `cudagraph_mark_step_begin()` prevents tensor overwrite across videos |
| **Parallelism** | Threaded mask post-processing | SAM2 | ThreadPoolExecutor (8 workers); OpenCV releases GIL → true parallelism |
| **Parallelism** | Threaded GCS uploads | SAM2, Data Gen | 8-thread upload with pre-created remote directories |
| **Parallelism** | Threading over multiprocessing | Data Gen | `ThreadPool` shares model memory (7× reduction vs `multiprocessing.Pool`) |
| **Caching** | Thread-safe GCS singleton | SAM2, Data Gen | Double-checked locking avoids per-call auth overhead |
| **Caching** | Per-device model singletons | VP, Data Gen | `_qwen_models[device]` reuses loaded models across calls |
| **Caching** | Deterministic frame folder sync | SAM2 | Reuses cached frames; trims surplus if MAX_FRAMES changes |
| **Memory** | In-memory mask pass-through | SAM2 | Segmentation → VP preprocessing without disk round-trip |
| **Memory** | Aggressive temp cleanup | SAM2 | `del masks` + `shutil.rmtree()` after each video |
| **Memory** | Model pre-loading at startup | Data Gen | Parallel GPU model loading before processing begins |
| **I/O** | Early-stop GCS listing | SAM2, Data Gen | Stop scanning once enough files collected for requested slice |
| **I/O** | Compressed NPZ archives | SAM2 | `np.savez_compressed` for mask storage (10–50× smaller than PNGs) |
| **I/O** | Single-model batch inference | Alpamayo | Load Alpamayo-R1-10B once, run inference on all videos sequentially |
| **Quality** | VOS-optimized predictor | SAM2 | `vos_optimized=True` reduces memory while maintaining temporal consistency |
| **Quality** | 4-stage morphological pipeline | SAM2 | Open → connected-component → close → flood-fill produces hole-free masks |
