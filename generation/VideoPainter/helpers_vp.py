"""Helper functions and data classes for the VideoPainter HLX workflow.

All non-HLX utility code (file I/O, GCS uploads, video evaluation, metrics
helpers, run-report generation, and the edit-bench driver) lives here so that
workflow_vp.py contains only the HLX @task / @workflow definitions.
"""

import csv
import itertools
import logging
import os
import platform
import re
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import gcsfs
import numpy as np
import torch

from config_vp import (
	BASE_WORKDIR,
	DEFAULT_CKPT_DIR,
	SCRATCH_DATA_BASE,
	VP_DATA_FUSE_MOUNT_ROOT,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT / PATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_instruction_list(instructions: str | None) -> list[str]:
	"""Parse a list of editing instructions from a single CLI string.

	Supported formats:
	- "instr1 || instr2 || instr3" (recommended)
	- newline-separated (useful in configs)
	"""
	if not instructions:
		return []
	s = str(instructions).strip()
	if not s:
		return []
	parts = s.split("||") if "||" in s else s.splitlines()
	return [p.strip().strip('"').strip() for p in parts if p.strip().strip('"').strip()]


def _sanitize_folder_component(text: str, *, max_len: int = 60) -> str:
	"""Make a safe folder name component for GCS/local paths."""
	base = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip()).strip("_") or "instruction"
	return base[:max_len].rstrip("_") if len(base) > max_len else base


def _lane_spec_to_instruction(c: str, col: str, pat: str) -> str:
	"""Convert lane specification to editing instruction."""
	c2 = (c or "").strip().lower()
	col2 = (col or "").strip().lower()
	pat2 = (pat or "").strip().lower()
	return f"lane {c2} {col2} {pat2}".strip() if (c2 and col2 and pat2) else ""


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSED-DATA STAGING
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_preprocessed_video_paths(*, data_run_id: str, data_video_id: str) -> tuple[str, str]:
	"""Get paths to meta.csv and raw_videos for a specific run_id + video ID."""
	meta = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id, "meta.csv")
	raw_root = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id, "raw_videos")
	return meta, raw_root


def _video_base_name_from_meta(*, meta_csv_path: str, inpainting_sample_id: int) -> str:
	"""Extract video base name from meta.csv for the given sample ID."""
	if inpainting_sample_id < 0:
		raise ValueError("inpainting_sample_id must be >= 0")
	with open(meta_csv_path, newline="") as f:
		row = next(itertools.islice(csv.DictReader(f), inpainting_sample_id, inpainting_sample_id + 1), None)
		if row:
			path = (row.get("path") or "").strip()
			if not path:
				raise KeyError(
					f"meta.csv row {inpainting_sample_id} is missing required 'path' column/value"
				)
			return path.split(".")[0]
	raise IndexError(
		f"meta.csv does not have row index {inpainting_sample_id} (file={meta_csv_path})"
	)


def _stage_preprocessed_inputs(*, data_run_id: str, data_video_id: str, inpainting_sample_id: int) -> tuple[str, str]:
	"""Stage meta.csv, masks, and video files to local scratch for processing."""
	meta_src, raw_root = _resolve_preprocessed_video_paths(data_run_id=data_run_id, data_video_id=data_video_id)
	if not os.path.exists(meta_src):
		raise FileNotFoundError(
			f"Missing meta.csv for run_id={data_run_id} video_id={data_video_id}: {meta_src}"
		)

	stage_dir = os.path.join(SCRATCH_DATA_BASE, data_video_id)
	Path(stage_dir).mkdir(parents=True, exist_ok=True)
	meta_dst = os.path.join(stage_dir, "meta.csv")
	shutil.copy2(meta_src, meta_dst)

	# IMPORTANT: the name `edit_bench.py` uses for mask lookup comes from the meta.csv row
	# (`video_base_name = meta_data['path'].split('.')[0]`). Do not assume it equals
	# `data_video_id`.
	video_base_name = _video_base_name_from_meta(
		meta_csv_path=meta_dst,
		inpainting_sample_id=inpainting_sample_id,
	)

	# Prefer a 'masks/' layout if it already exists in the mounted data, else fall back
	# to our preprocessing 'mask_root/' layout.
	mounted_video_dir = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, data_video_id)
	mounted_masks_preferred = os.path.join(mounted_video_dir, "masks", data_video_id, "all_masks.npz")
	mounted_masks_legacy = os.path.join(mounted_video_dir, "mask_root", data_video_id, "all_masks.npz")

	if os.path.exists(mounted_masks_preferred):
		masks_src = mounted_masks_preferred
	elif os.path.exists(mounted_masks_legacy):
		masks_src = mounted_masks_legacy
	else:
		raise FileNotFoundError(
			f"Missing all_masks.npz for video_id={data_video_id}. "
			f"Looked for: {mounted_masks_preferred} and {mounted_masks_legacy}"
		)

	# Create the path edit_bench.py looks for: <data_dir>/masks/<video_base_name>/all_masks.npz
	staged_masks_dst = os.path.join(stage_dir, "masks", video_base_name, "all_masks.npz")
	_copy_file(masks_src, staged_masks_dst)

	# Also create the legacy repo-layout location that edit_bench.py may fall back to:
	#   ../data/video_inpainting/videovo/<video_id>/all_masks.npz
	# When the working directory is /workspace/VideoPainter, that resolves to:
	#   /workspace/data/video_inpainting/videovo/<video_id>/all_masks.npz
	legacy_masks_dst = os.path.join(
		"/workspace",
		"data",
		"video_inpainting",
		"videovo",
		video_base_name,
		"all_masks.npz",
	)
	_copy_file(masks_src, legacy_masks_dst)

	# Sanity check: edit_bench.py decides between preferred and legacy using os.path.exists,
	# so ensure both exist as real files.
	if not os.path.exists(staged_masks_dst):
		raise RuntimeError(f"Failed to stage masks to preferred path: {staged_masks_dst}")
	if not os.path.exists(legacy_masks_dst):
		raise RuntimeError(f"Failed to stage masks to legacy path: {legacy_masks_dst}")
	logger.info(
		"Staged masks for video_id=%s (video_base_name=%s): preferred=%s legacy=%s (src=%s)",
		data_video_id,
		video_base_name,
		staged_masks_dst,
		legacy_masks_dst,
		masks_src,
	)


	# Stage the raw video file into the exact layout edit_bench.py expects.
	# In edit_bench.py (for '.0.mp4' rows):
	#   video_path = os.path.join(image_or_video_path, video_base_name[:-3], f"{video_base_name}.0.mp4")
	# So we make image_or_video_path point at our staged 'raw_videos' root.
	staged_video_root = os.path.join(stage_dir, "raw_videos")
	expected_filename = f"{video_base_name}.0.mp4"
	video_src = _select_video_file_under_root(
		raw_root=raw_root,
		video_base_name=video_base_name,
		data_video_id=data_video_id,
		expected_filename=expected_filename,
	)
	staged_video_dir = os.path.join(staged_video_root, video_base_name[:-3])
	Path(staged_video_dir).mkdir(parents=True, exist_ok=True)
	staged_video_dst = os.path.join(staged_video_dir, expected_filename)
	ensure_symlink(video_src, staged_video_dst)
	if not os.path.exists(staged_video_dst):
		raise RuntimeError(f"Failed to stage video to expected path: {staged_video_dst}")
	logger.info(
		"Staged video for video_id=%s (video_base_name=%s): %s -> %s",
		data_video_id,
		video_base_name,
		video_src,
		staged_video_dst,
	)

	return meta_dst, staged_video_root


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE LISTING / SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _list_files_under_root(*, raw_root: str, suffix: str, max_depth: int = 6, max_results: int = 200) -> list[str]:
	"""List files ending with suffix under raw_root (bounded depth)."""
	raw_root_abs = os.path.abspath(raw_root)
	results: list[str] = []
	if not os.path.isdir(raw_root_abs):
		return results
	for root, dirs, files in os.walk(raw_root_abs):
		rel = os.path.relpath(root, raw_root_abs)
		depth = 0 if rel == "." else rel.count(os.sep) + 1
		if depth > max_depth:
			dirs[:] = []
			continue
		for fn in files:
			if fn.endswith(suffix):
				results.append(os.path.join(root, fn))
				if len(results) >= max_results:
					return results
	return results


def _select_video_file_under_root(
	*,
	raw_root: str,
	video_base_name: str,
	data_video_id: str,
	expected_filename: str,
	max_depth: int = 6,
) -> str:
	"""Find the video file matching the expected filename under raw_root."""
	# 1) Exact fast-paths
	exact = os.path.join(raw_root, video_base_name[:-3], expected_filename)
	if os.path.exists(exact):
		return exact
	exact2 = os.path.join(raw_root, expected_filename)
	if os.path.exists(exact2):
		return exact2

	# 2) Scan for candidates
	candidates = _list_files_under_root(raw_root=raw_root, suffix=".mp4", max_depth=max_depth)
	if not candidates:
		raise FileNotFoundError(
			f"No .mp4 files found under raw_root={os.path.abspath(raw_root)} (max_depth={max_depth}). "
			f"Expected something like '{expected_filename}'."
		)
	if len(candidates) == 1:
		return candidates[0]

	# 3) Score candidates
	def score(p: str) -> tuple[int, int]:
		fn = os.path.basename(p)
		# Higher is better.
		s = 0
		if fn == expected_filename:
			s += 100
		elif fn == f"{video_base_name}.mp4":
			s += 90
		elif video_base_name in fn:
			s += 70
		elif data_video_id in fn:
			s += 60
		if fn.endswith(".0.mp4"):
			s += 10
		# Prefer shallower paths
		depth_penalty = p.count(os.sep)
		return (s, -depth_penalty)

	scored = sorted(((score(p), p) for p in candidates), reverse=True)
	best_score, best_path = scored[0]

	# If multiple share the same top score, it's ambiguous; show a short list.
	ties = [p for (sc, p) in scored if sc == best_score]
	if len(ties) > 1:
		preview = "\n".join(ties[:20])
		raise FileNotFoundError(
			f"Ambiguous video selection under raw_root={os.path.abspath(raw_root)}\n"
			f"Expected filename={expected_filename}\n"
			f"Multiple mp4 candidates match equally well. Top candidates (first 20):\n{preview}"
		)

	logger.info(
		"Selected video source for %s (video_base_name=%s): %s (score=%s)",
		data_video_id,
		video_base_name,
		best_path,
		best_score,
	)
	return best_path



def _list_preprocessed_video_ids(*, data_run_id: str) -> list[str]:
	"""List candidate video IDs for a given run_id under the mounted dataset."""
	try:
		entries = os.listdir(os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id))
	except FileNotFoundError:
		return []
	video_ids = []
	for name in sorted(entries):
		p = os.path.join(VP_DATA_FUSE_MOUNT_ROOT, data_run_id, name)
		if os.path.isdir(p) and (os.path.exists(os.path.join(p, "meta.csv")) or os.path.exists(os.path.join(p, "raw_videos"))):
			video_ids.append(name)
	return video_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  FILESYSTEM HELPERS (symlinks, copies, writable dirs)
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_symlink(src: str, dest: str) -> None:
	"""Create a symlink from dest -> src if not already present."""
	dest_parent = Path(dest).parent
	dest_parent.mkdir(parents=True, exist_ok=True)
	if os.path.islink(dest):
		if os.readlink(dest) == src:
			return
		os.unlink(dest)
	elif os.path.exists(dest):
		# Replace existing directory (empty or non-empty) with symlink
		if os.path.isdir(dest):
			try:
				shutil.rmtree(dest)
				logger.info("Removed existing directory at %s to create symlink.", dest)
			except OSError as e:
				logger.warning("Failed to remove directory at %s: %s", dest, e)
				return
		else:
			logger.info("Path %s already exists and is not a symlink; leaving as-is.", dest)
			return
	os.symlink(src, dest)
	logger.info("Created symlink %s -> %s", dest, src)


def ensure_writable_dir(path: str) -> None:
	"""Ensure `path` is a real, writable directory (not a symlink).

	We need this when we want to create additional symlinks *inside* the directory.
	If `path` is a symlink to a read-only FUSE mount, creating new entries under it
	will fail.
	"""
	if os.path.islink(path):
		os.unlink(path)
	if os.path.exists(path):
		if os.path.isdir(path):
			return
		raise RuntimeError(f"Path exists and is not a directory: {path}")
	Path(path).mkdir(parents=True, exist_ok=True)



def _copy_file(src: str, dest: str) -> None:
	"""Copy a file to dest, replacing symlinks/empty dirs if needed.

	We use copies (not symlinks) for masks because `edit_bench.py` uses `os.path.exists`
	to pick a path; with FUSE-backed symlinks that can still be flaky.
	"""
	dest_parent = Path(dest).parent
	dest_parent.mkdir(parents=True, exist_ok=True)

	# If a symlink exists at destination, replace it.
	if os.path.islink(dest):
		os.unlink(dest)
	elif os.path.exists(dest):
		# If an empty directory exists at destination, replace it.
		if os.path.isdir(dest):
			try:
				if not any(Path(dest).iterdir()):
					shutil.rmtree(dest)
				else:
					logger.info("Path %s exists as non-empty directory; leaving as-is.", dest)
					return
			except OSError:
				logger.info("Path %s exists and cannot be replaced safely; leaving as-is.", dest)
				return
		else:
			# File already exists; keep it.
			return

	shutil.copy2(src, dest)
	logger.info("Copied file %s -> %s", src, dest)


# ═══════════════════════════════════════════════════════════════════════════════
#  GCS UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def upload_directory_to_gcs(local_dir: str, gcs_prefix: str) -> None:
	"""Recursively upload a local directory to a GCS prefix using gcsfs."""

	fs = gcsfs.GCSFileSystem(token="google_default")
	base = Path(local_dir)
	for path in base.rglob("*"):
		if path.is_dir():
			continue
		rel = path.relative_to(base).as_posix()
		remote = f"{gcs_prefix.rstrip('/')}/{rel}"
		remote_parent = os.path.dirname(remote)
		if remote_parent:
			fs.makedirs(remote_parent, exist_ok=True)
		fs.put(path.as_posix(), remote)


def upload_file_to_gcs(local_path: str, gcs_path: str) -> None:
	"""Upload a single file to a full gs:// path using gcsfs."""
	fs = gcsfs.GCSFileSystem(token="google_default")
	remote_parent = os.path.dirname(gcs_path)
	if remote_parent:
		fs.makedirs(remote_parent, exist_ok=True)
	fs.put(local_path, gcs_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO / MASK LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
	"""Load video frames from a video file."""
	cap = cv2.VideoCapture(video_path)
	frames = []
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(frame_rgb)
	
	cap.release()
	return frames, fps


def _frames_to_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
	"""Convert list of numpy frames to tensor [T, C, H, W]."""
	frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
	frames_tensor = frames_tensor.permute(0, 3, 1, 2)
	return frames_tensor.to(device)


def _load_mask_from_npz(mask_path: str, num_frames: int, frame_height: int, frame_width: int) -> np.ndarray:
	"""Load mask frames from npz file."""
	if os.path.exists(mask_path):
		data = np.load(mask_path)
		if 'masks' in data:
			masks = data['masks'][:num_frames]
			if masks.shape[1:3] != (frame_height, frame_width):
				resized_masks = []
				for mask in masks:
					resized = cv2.resize(mask, (frame_width, frame_height))
					resized_masks.append(resized)
				return np.stack(resized_masks)
			return masks
	return np.ones((num_frames, frame_height, frame_width), dtype=np.uint8) * 255


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_video(
	*,
	original_path: str,
	generated_path: str,
	mask_path: str,
	caption: str,
	output_dir: str,
	video_id: str,
	metrics_calculator=None,
) -> dict:
	"""Evaluate generated video against original using numerical metrics.

	If *metrics_calculator* is provided it will be reused (avoids reloading
	CLIP / LPIPS weights for every video).  When ``None`` a fresh instance is
	created internally (backwards-compatible fallback).
	"""
	try:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		logger.info("[EVAL][%s] Loading videos for evaluation", video_id)
		
		# Import metrics calculator — guard against missing optional deps
		# (e.g. qwen-vl-utils, clip) that metrics.py may pull in at module level.
		try:
			sys.path.insert(0, os.path.join(BASE_WORKDIR, "evaluate"))
			from metrics import MetricsCalculator
		except ImportError as imp_err:
			logger.warning("[EVAL][%s] Cannot import MetricsCalculator (%s); skipping evaluation", video_id, imp_err)
			return {'error': f"MetricsCalculator import failed: {imp_err}"}
		
		original_frames, original_fps = _load_video_frames(original_path)
		generated_frames, _ = _load_video_frames(generated_path)
		
		min_frames = min(len(original_frames), len(generated_frames))
		original_frames = original_frames[:min_frames]
		generated_frames = generated_frames[:min_frames]
		
		orig_h, orig_w = original_frames[0].shape[:2]
		gen_h, gen_w = generated_frames[0].shape[:2]
		
		if (orig_h != gen_h) or (orig_w != gen_w):
			generated_frames = [cv2.resize(f, (orig_w, orig_h)) for f in generated_frames]
		
		masks = _load_mask_from_npz(mask_path, min_frames, orig_h, orig_w)
		mask_ratio = (masks > 127).sum() / masks.size
		
		original_tensor = _frames_to_tensor(original_frames, device)
		generated_tensor = _frames_to_tensor(generated_frames, device)
		mask_tensor = torch.from_numpy(masks).float().to(device) / 255.0
		mask_tensor = mask_tensor.unsqueeze(1)
		
		# Reuse the caller-provided calculator or create a fresh one.
		if metrics_calculator is None:
			logger.info("[EVAL][%s] Creating new MetricsCalculator (no shared instance provided)", video_id)
			metrics_calculator = MetricsCalculator(device)
		
		# Full-frame metrics (single-pass, all on GPU)
		logger.info("[EVAL][%s] Calculating full-frame metrics", video_id)
		full_metrics = metrics_calculator.compute_all_metrics(
			original_tensor, generated_tensor, caption=caption or None,
		)
		
		# Masked region metrics (single-pass, all on GPU)
		logger.info("[EVAL][%s] Calculating masked-region metrics", video_id)
		mask_binary = (mask_tensor > 0.5).float()
		masked_metrics = metrics_calculator.compute_all_metrics(
			original_tensor, generated_tensor, mask=mask_binary, caption=caption or None,
		)
		
		# Save results
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_path = os.path.join(output_dir, f"eval_{video_id}.txt")
		
		with open(output_path, 'w') as f:
			f.write("="*80 + "\n")
			f.write("VIDEO EVALUATION SUMMARY\n")
			f.write("="*80 + "\n\n")
			f.write(f"Video ID: {video_id}\n")
			f.write(f"Evaluation Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
			f.write(f"Original Video:  {original_path}\n")
			f.write(f"Generated Video: {generated_path}\n")
			f.write(f"Mask:            {mask_path}\n\n")
			f.write(f"Frames:          {min_frames}\n")
			f.write(f"Resolution:      {orig_w} x {orig_h}\n")
			f.write(f"FPS:             {original_fps:.2f}\n")
			f.write(f"Mask Coverage:   {mask_ratio*100:.2f}%\n\n")
			f.write("="*80 + "\n")
			f.write("FULL-FRAME METRICS\n")
			f.write("="*80 + "\n")
			f.write(f"  PSNR:                 {full_metrics['PSNR']:>10.4f} dB\n")
			f.write(f"  SSIM:                 {full_metrics['SSIM']:>10.4f}\n")
			f.write(f"  LPIPS:                {full_metrics['LPIPS']:>10.4f}\n")
			f.write(f"  MSE:                  {full_metrics['MSE']:>10.6f}\n")
			f.write(f"  MAE:                  {full_metrics['MAE']:>10.6f}\n")
			f.write(f"  Temporal Consistency: {full_metrics['Temporal_Consistency']:>10.4f}\n")
			if caption:
				f.write(f"  CLIP Score:           {full_metrics['CLIP_Score']:>10.4f}\n")
			f.write("\n" + "="*80 + "\n")
			f.write("MASKED-REGION METRICS\n")
			f.write("="*80 + "\n")
			f.write(f"  PSNR:                 {masked_metrics['PSNR']:>10.4f} dB\n")
			f.write(f"  SSIM:                 {masked_metrics['SSIM']:>10.4f}\n")
			f.write(f"  LPIPS:                {masked_metrics['LPIPS']:>10.4f}\n")
			f.write(f"  MSE:                  {masked_metrics['MSE']:>10.6f}\n")
			f.write(f"  MAE:                  {masked_metrics['MAE']:>10.6f}\n")
			f.write(f"  Temporal Consistency: {masked_metrics['Temporal_Consistency']:>10.4f}\n")
			if caption:
				f.write(f"  CLIP Score:           {masked_metrics['CLIP_Score']:>10.4f}\n")
			f.write("\n" + "="*80 + "\n")
			
			quality_score = 0
			if full_metrics['PSNR'] > 40:
				quality_score += 1
			if full_metrics['SSIM'] > 0.95:
				quality_score += 1
			if full_metrics['LPIPS'] < 0.1:
				quality_score += 1
			
			f.write("QUALITY ASSESSMENT: ")
			if quality_score >= 2:
				f.write("EXCELLENT\n")
			elif quality_score == 1:
				f.write("GOOD\n")
			else:
				f.write("MODERATE\n")
			f.write("="*80 + "\n")
		
		logger.info("[EVAL][%s] Evaluation complete: PSNR=%.2f SSIM=%.4f LPIPS=%.4f", 
				   video_id, full_metrics['PSNR'], full_metrics['SSIM'], full_metrics['LPIPS'])
		
		return {
			'output_path': output_path,
			'full_metrics': full_metrics,
			'masked_metrics': masked_metrics,
			'mask_coverage': float(mask_ratio),
		}
		
	except Exception as e:
		logger.error("[EVAL][%s] Evaluation failed: %s", video_id, e, exc_info=True)
		return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCE-FILE UPLOAD (reproducibility)
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_run_source_files(*, gcs_save_path: str) -> list[str]:
	"""Upload key source files for reproducibility.

	Uploads into: <gcs_save_path>/sources/
	"""
	uploaded: list[str] = []
	sources_prefix = os.path.join(gcs_save_path, "sources")

	# These paths are inside the container image.
	candidates: list[tuple[str, str]] = [
		(os.path.join(BASE_WORKDIR, "workflow.py"), "workflow.py"),
		(os.path.join(BASE_WORKDIR, "infer", "edit_bench.py"), "infer/edit_bench.py"),
		# The user-facing launcher script is typically a .sh; accept a .py name too.
		(os.path.join(BASE_WORKDIR, "scripts", "build_and_run.sh"), "scripts/build_and_run.sh"),
		(os.path.join(BASE_WORKDIR, "scripts", "build_and_run.py"), "scripts/build_and_run.py"),
	]

	for local_path, rel_name in candidates:
		if not os.path.exists(local_path):
			logger.info("Source file not found (skipping): %s", local_path)
			continue
		remote_path = os.path.join(sources_prefix, rel_name)
		try:
			upload_file_to_gcs(local_path, remote_path)
			uploaded.append(remote_path)
		except Exception as e:
			logger.info("Failed to upload source file %s -> %s (non-fatal): %s", local_path, remote_path, e)

	if uploaded:
		logger.info("Uploaded %d source file(s) under: %s", len(uploaded), sources_prefix)
	return uploaded


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN METRICS DATA CLASS + REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VPVideoMetrics:
	video_id: str
	output_name: str
	generation_s: float
	phase1_s: float  # Qwen + FluxFill (first-frame generation)
	phase2_s: float  # CogVideoX (video generation)
	upload_s: float
	device: str
	gpu_name: str
	gpu_compute_capability: str
	peak_gpu_mem_allocated_mb: float
	peak_gpu_mem_reserved_mb: float
	rss_mb: float


def _reset_torch_cuda_peaks() -> None:
	try:
		import torch  # type: ignore
		if torch.cuda.is_available():
			torch.cuda.reset_peak_memory_stats(0)
	except Exception:
		return


def _get_torch_cuda_metrics() -> tuple[str, str, str, float, float]:
	"""Return (device, gpu_name, compute_capability, peak_alloc_mb, peak_reserved_mb)."""
	try:
		import torch  # type: ignore
	except Exception:
		return ("unknown", "unknown", "unknown", 0.0, 0.0)

	if not torch.cuda.is_available():
		return ("cpu", "cpu", "n/a", 0.0, 0.0)

	try:
		props = torch.cuda.get_device_properties(0)
		cc = f"{props.major}.{props.minor}"
		name = torch.cuda.get_device_name(0)
		peak_alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
		peak_reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
		return ("cuda", name, cc, float(peak_alloc), float(peak_reserved))
	except Exception:
		return ("cuda", "unknown", "unknown", 0.0, 0.0)


def _get_rss_mb() -> float:
	"""Best-effort RSS (resident set size) in MB."""
	try:
		import psutil  # type: ignore
		p = psutil.Process(os.getpid())
		return float(p.memory_info().rss) / (1024 * 1024)
	except Exception:
		pass
	try:
		# ru_maxrss is KB on Linux.
		rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		return float(rss_kb) / 1024.0
	except Exception:
		return 0.0


def _write_videopainter_run_report(
	*,
	run_id: str,
	report_path: str,
	per_video: list[VPVideoMetrics],
	input_data_run_id: Optional[str] = None,
) -> None:
	lines: list[str] = []
	lines.append(f"run_id: {run_id}")
	if input_data_run_id is not None and input_data_run_id != run_id:
		lines.append(f"input_data_run_id: {input_data_run_id}")
	lines.append(f"timestamp_utc: {datetime.utcnow().isoformat()}Z")
	lines.append(f"platform: {platform.platform()}")
	lines.append("")
	lines.append("per_video:")
	for m in per_video:
		lines.append(f"- video_id: {m.video_id}")
		lines.append(f"  output_name: {m.output_name}")
		lines.append(f"  generation_s: {m.generation_s:.3f}")
		lines.append(f"  phase1_s: {m.phase1_s:.3f}")
		lines.append(f"  phase2_s: {m.phase2_s:.3f}")
		lines.append(f"  upload_s: {m.upload_s:.3f}")
		lines.append(f"  device: {m.device}")
		lines.append(f"  gpu_name: {m.gpu_name}")
		lines.append(f"  gpu_compute_capability: {m.gpu_compute_capability}")
		lines.append(f"  peak_gpu_mem_allocated_mb: {m.peak_gpu_mem_allocated_mb:.1f}")
		lines.append(f"  peak_gpu_mem_reserved_mb: {m.peak_gpu_mem_reserved_mb:.1f}")
		lines.append(f"  rss_mb: {m.rss_mb:.1f}")

	Path(os.path.dirname(report_path)).mkdir(parents=True, exist_ok=True)
	Path(report_path).write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  EDIT-BENCH DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_edit_bench(
	*,
	output_path: str,
	inpainting_mask_meta: str,
	image_or_video_path: str,
	prompt: str,
	model_path: str,
	inpainting_branch: str,
	img_inpainting_model: str,
	img_inpainting_lora_path: str,
	img_inpainting_lora_scale: float,
	num_inference_steps: int,
	guidance_scale: float,
	num_videos_per_prompt: int,
	dtype: str,
	inpainting_sample_id: int,
	inpainting_frames: int,
	down_sample_fps: int,
	overlap_frames: int,
	prev_clip_weight: float,
	strength: float,
	video_editing_instruction: str,
	llm_model: str,
	dilate_size: int,
	mask_feather: int,
	caption_refine_iters: int,
	caption_refine_temperature: float,
	keep_masked_pixels: bool,
	seed: int,
	preloaded_models: dict | None = None,
	run_phase: str = "all",
) -> None:
	"""Run VideoPainter generation via direct in-process call.

	All heavy models (CogVideoX, FluxFill, Qwen) are pre-loaded and reused
	across calls via *preloaded_models*.

	*run_phase* controls two-phase execution:
	  - ``"first_frame"`` — Qwen + FluxFill only, save results to disk
	  - ``"video"``       — CogVideoX only (reads Phase-1 outputs from disk)
	"""
	cog_device = (os.environ.get("VP_COG_DEVICE") or "").strip()
	flux_device = (os.environ.get("VP_FLUX_DEVICE") or "").strip()
	qwen_device = (os.environ.get("VP_QWEN_DEVICE") or "").strip()

	dtype_torch = torch.bfloat16 if dtype == "bfloat16" else torch.float16

	sys.path.insert(0, os.path.join(BASE_WORKDIR, "infer"))
	from edit_bench import generate_video  # type: ignore[import-untyped]

	generate_video(
		prompt=prompt,
		model_path=model_path,
		output_path=output_path,
		image_or_video_path=image_or_video_path,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		num_videos_per_prompt=num_videos_per_prompt,
		dtype=dtype_torch,
		generate_type="i2v_inpainting",
		seed=seed,
		inpainting_mask_meta=inpainting_mask_meta,
		inpainting_sample_id=inpainting_sample_id,
		inpainting_branch=inpainting_branch,
		inpainting_frames=inpainting_frames,
		down_sample_fps=down_sample_fps,
		overlap_frames=overlap_frames,
		prev_clip_weight=prev_clip_weight,
		strength=float(strength),
		img_inpainting_model=img_inpainting_model,
		img_inpainting_lora_path=img_inpainting_lora_path,
		img_inpainting_lora_scale=float(img_inpainting_lora_scale),
		video_editing_instruction=video_editing_instruction,
		llm_model=llm_model,
		qwen_device=qwen_device or None,
		cog_device=cog_device or "cuda",
		flux_device=flux_device or "cuda",
		dilate_size=dilate_size,
		mask_feather=mask_feather,
		caption_refine_iters=int(caption_refine_iters or 0),
		caption_refine_temperature=float(caption_refine_temperature or 0.2),
		keep_masked_pixels=keep_masked_pixels,
		first_frame_gt=True,
		replace_gt=True,
		mask_add=True,
		preloaded_models=preloaded_models,
		run_phase=run_phase,
	)


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def _upload_outputs(
	*,
	output_path: str,
	remote_output_path: str,
) -> list[str]:
	"""Upload main and optional sidecar outputs; return uploaded gs:// paths."""
	uploaded: list[str] = []
	upload_file_to_gcs(output_path, remote_output_path)
	uploaded.append(remote_output_path)

	# Upload common sidecar artifacts produced by infer/edit_bench.py (when enabled).
	# These are very helpful for debugging prompt/mask issues.
	sidecars = [
		output_path.replace(".mp4", ".json"),
		output_path + "_flux_i_img.png",
		output_path + "_flux_i_mask.png",
		output_path + "_flux_o_img.png",
		output_path + "_gt_o_img.png",
	]

	# Optional iterative-refinement artifacts (only when caption_refine_iters > 0).
	# Stored under a dedicated per-video folder: <output_basename>_caption_refine/
	try:
		from glob import glob
		base_no_ext = os.path.splitext(os.path.basename(output_path))[0]
		refine_dir = os.path.join(os.path.dirname(output_path), f"{base_no_ext}_caption_refine")
		for p in sorted(glob(os.path.join(refine_dir, "*"))):
			sidecars.append(p)
	except Exception:
		pass
	for local_path in sidecars:
		if not os.path.exists(local_path):
			continue

		# Preserve folder structure for refinement artifacts.
		if local_path.endswith(".json") and os.path.basename(local_path) == os.path.basename(output_path).replace(".mp4", ".json"):
			remote_path = remote_output_path.replace(".mp4", ".json")
		else:
			remote_dir = os.path.dirname(remote_output_path)
			refine_dir_basename = f"{os.path.splitext(os.path.basename(output_path))[0]}_caption_refine"
			if os.path.basename(os.path.dirname(local_path)) == refine_dir_basename:
				remote_path = os.path.join(remote_dir, refine_dir_basename, os.path.basename(local_path))
			elif local_path.startswith(output_path):
				suffix = local_path[len(output_path):]
				remote_path = remote_output_path + suffix
			else:
				remote_path = os.path.join(remote_dir, os.path.basename(local_path))

		upload_file_to_gcs(local_path, remote_path)
		uploaded.append(remote_path)

	generated_only = output_path.replace(".mp4", "_generated.mp4")
	if os.path.exists(generated_only):
		remote_generated = remote_output_path.replace(".mp4", "_generated.mp4")
		upload_file_to_gcs(generated_only, remote_generated)
		uploaded.append(remote_generated)
	return uploaded
