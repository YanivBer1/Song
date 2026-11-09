#!/usr/bin/env python3
"""
Run the deenasun/ai-sl-api Space locally, with no external secrets.

  python run_local.py generate "Hi, my name is John" -o outputs/john.mp4
  python run_local.py serve            # Gradio UI on http://0.0.0.0:7860

Requires gensim-data/GoogleNews-vectors-negative300.bin.gz (see README).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "upstream"))
os.chdir(HERE)  # the Space loads gensim-data/ relative to the CWD

# app.py refuses to import without R2 credentials, and its document glosser
# requires an Anthropic key at construction time. Neither is used on the
# text-input path (the R2 upload is patched out below), so placeholders suffice.
for var in ("R2_ASL_VIDEOS_URL", "R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ.setdefault(var, "https://unused.invalid")
os.environ.setdefault("ANTHROPIC_API_KEY", "unused-for-text-input")

import app  # noqa: E402  (upstream Space code)
import local_backend  # noqa: E402

local_backend.install(app)


def to_h264(src, dst):
    """OpenCV's pip wheel writes mp4v; re-encode to browser-playable H.264."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
        check=True,
    )


def generate(text, out_path):
    result, video_path = app.predict(text, None)
    result = {k: v for k, v in result.items() if k != "video_as_base_64"}
    if not video_path:
        print(json.dumps(result, indent=2))
        raise SystemExit("No video produced")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if len(result.get("videos", [])) > 1:
        to_h264(video_path, out_path)  # stitched skeleton render
    else:
        shutil.copy(video_path, out_path)  # single original clip
    result["output_file"] = out_path
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="text -> ASL video file")
    g.add_argument("text")
    g.add_argument("-o", "--output", default="outputs/asl.mp4")
    s = sub.add_parser("serve", help="launch the Space's Gradio UI")
    s.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    if args.cmd == "generate":
        generate(args.text, args.output)
    else:
        app.create_interface().launch(server_name="0.0.0.0", server_port=args.port,
                                      share=False)


if __name__ == "__main__":
    main()
