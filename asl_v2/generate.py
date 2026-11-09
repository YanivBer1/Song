#!/usr/bin/env python3
"""
v2 text -> ASL video.

  python -m asl_v2.generate "Hi, my name is John" -o outputs/v2/john.mp4
  python -m asl_v2.generate "..." --renderer human -o outputs/v2/john_realistic.mp4

Sign lookup is the same as the Space (word2vec nearest neighbour over the
Space's sign dictionary); everything after that is replaced:
Holistic keypoints -> asl_v2.motion (rest hands, normalisation, gap filling,
eased transitions, smoothing) -> renderer:
  2d        illustrated 2D person (seconds)
  human     realistic MakeHuman character in Blender/Cycles (~7 s/frame on CPU)
  mannequin procedural Blender skin-modifier figure (experiment)
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "upstream"))


def lookup(words):
    os.chdir(HERE)  # upstream Vectorizer loads gensim-data/ relative to CWD
    from vectorizer import Vectorizer
    from local_backend import LocalVectorIndex

    vec = Vectorizer()
    index = LocalVectorIndex(vec.kv)
    matches = []
    for w in words:
        emb = vec.encode(w)
        if emb is None:
            print(f"skip '{w}': not in vocabulary")
            continue
        best = index.match(emb)[0]
        matches.append(dict(query=w, sign=best["word"], url=best["video_url"],
                            similarity=round(best["similarity"], 3)))
    return matches


def tokens(text):
    return [t for t in (re.sub(r"[^\w\s]", "", tok).lower().strip() for tok in text.split()) if t]


def tracked_keypoints(word, url=None):
    """v3 tracking (crop-model hands with identity checks) in the keypoints() layout."""
    from .track import track
    t = track(word, url)
    return dict(pose=t["pose"], face=t["face"], lh=t["img_L"], rh=t["img_R"], size=t["size"], fps=t["fps"],
                world_lh=t["world_L"], world_rh=t["world_R"])


def build_motion(matches, tracking="v3"):
    from .extract import keypoints
    from .motion import process_clip, stitch
    get = tracked_keypoints if tracking == "v3" else keypoints
    return stitch([process_clip(get(m["sign"], m["url"])) for m in matches])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="outputs/v2/asl.mp4")
    ap.add_argument("--renderer", choices=["2d", "human", "mannequin"], default="2d")
    ap.add_argument("--signs", help="comma-separated sign words; skips the word2vec lookup")
    ap.add_argument("--samples", type=int, default=16, help="Cycles samples (Blender renderers)")
    ap.add_argument("--res", default="960x540", help="Blender render resolution")
    args = ap.parse_args()
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.signs:
        from .extract import VIDEO_BASE
        matches = [dict(query=s, sign=s, url=f"{VIDEO_BASE}/{s}.mp4", similarity=None)
                   for s in args.signs.split(",")]
    else:
        matches = lookup(tokens(args.text))
    print(json.dumps(matches, indent=2))
    seq = build_motion(matches)
    print(f"motion: {len(seq)} frames ({len(seq) / 30:.1f}s)")

    if args.renderer == "2d":
        from .render2d import write_video
        write_video(seq, out)
    else:
        motion_file = os.path.splitext(out)[0] + ".motion.npy"
        np.save(motion_file, seq.astype(np.float32))
        blender_py = os.path.join(HERE, ".venv-blender", "bin", "python")
        script = "render_human.py" if args.renderer == "human" else "render3d.py"
        subprocess.run([blender_py, os.path.join(HERE, "asl_v2", script), motion_file, out,
                        "--samples", str(args.samples), "--res", args.res], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
