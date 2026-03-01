"""SAM2 workflow helper functions."""
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


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
                import shutil
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


def _resolve_chunk_uri(chunk_uri: str) -> List[str]:
    """Resolve a chunks:// URI into a list of gs:// video file paths.

    Format: chunks://<bucket>/<prefix>?start=N&end=M&per_chunk=K

    The base path (bucket/prefix) should point to a directory containing
    chunk_NNNN/ subfolders, each holding .mp4 files.  This function lists
    the files in each requested chunk and returns up to *per_chunk* files
    from each.

    Returns:
        List of gs:// URIs for individual video files.
    """
    from urllib.parse import urlparse, parse_qs
    import gcsfs

    parsed = urlparse(chunk_uri)
    # netloc + path gives us the GCS bucket/prefix
    base_path = parsed.netloc + parsed.path  # e.g. bucket/prefix/camera_folder
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
            # Filter to .mp4 files and take up to per_chunk
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
