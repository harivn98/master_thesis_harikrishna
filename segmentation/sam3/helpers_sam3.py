"""SAM3 workflow helper functions."""
import logging
from typing import List

logger = logging.getLogger(__name__)


def _resolve_chunk_uri(chunk_uri: str) -> List[str]:
    """Resolve a chunks:// URI into a list of gs:// video file paths."""
    from urllib.parse import urlparse, parse_qs
    import gcsfs

    parsed = urlparse(chunk_uri)
    base_path = parsed.netloc + parsed.path
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
