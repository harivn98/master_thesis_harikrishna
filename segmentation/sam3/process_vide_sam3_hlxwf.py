"""
SAM 3 Video Processing Script for HLX Workflow
Wrapper script that accepts parameters and calls process_videos_sam3.py
"""
import os
import sys
import argparse
from pathlib import Path

# Add sam3 to path
BASE_WORKDIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_WORKDIR))

# Import the main processing script
exec_globals = {"__name__": "__imported__", "__file__": str(BASE_WORKDIR / "process_videos_sam3.py")}
with open(BASE_WORKDIR / "process_videos_sam3.py", 'r') as f:
    exec(f.read(), exec_globals)

# Extract the function we need
process_all_videos = exec_globals["process_all_videos"]


def main():
    parser = argparse.ArgumentParser(description="SAM3 Processing for HLX Workflow")
    parser.add_argument("--video-uris", nargs="+", required=True, help="List of video URIs/paths to process")
    parser.add_argument("--output-bucket", required=True, help="GCS output bucket")
    parser.add_argument("--preprocessed-bucket", required=True, help="GCS preprocessed bucket")
    parser.add_argument("--upload-gcp", action="store_true", help="Upload to GCP")
    parser.add_argument("--upload-local", action="store_true", help="Keep local copies")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames per video")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--text-prompt", type=str, default="road surface",
                        help="Text prompt for SAM3 segmentation")

    args = parser.parse_args()

    # Override configuration in the imported function's global namespace
    process_all_videos.__globals__["UPLOAD_TO_GCP"] = args.upload_gcp
    process_all_videos.__globals__["UPLOAD_TO_LOCAL"] = args.upload_local
    process_all_videos.__globals__["GCP_OUTPUT_BUCKET"] = args.output_bucket
    process_all_videos.__globals__["GCP_PREPROCESSED_BUCKET"] = args.preprocessed_bucket
    process_all_videos.__globals__["MAX_FRAMES"] = args.max_frames
    process_all_videos.__globals__["TIMESTAMP"] = args.run_id
    process_all_videos.__globals__["SAM3_TEXT_PROMPT"] = args.text_prompt

    # Set up directories
    BASE_DATA_DIR = Path("/tmp/sam3_data")
    OUTPUT_DIR = BASE_DATA_DIR / f"output_{args.run_id}"
    FRAMES_DIR = BASE_DATA_DIR / f"frames_{args.run_id}"

    process_all_videos.__globals__["BASE_DATA_DIR"] = BASE_DATA_DIR
    process_all_videos.__globals__["OUTPUT_DIR"] = OUTPUT_DIR
    process_all_videos.__globals__["FRAMES_DIR"] = FRAMES_DIR

    print(f"Starting SAM3 processing for run_id: {args.run_id}")
    print(f"Processing {len(args.video_uris)} videos")
    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Output bucket: {args.output_bucket}")
    print(f"Preprocessed bucket: {args.preprocessed_bucket}")
    print()

    # Run the processing
    process_all_videos(args.video_uris)


if __name__ == "__main__":
    main()
