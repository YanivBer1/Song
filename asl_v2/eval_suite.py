"""
Consistency suite: many random phrases (different word orders and combinations)
built through the full pipeline and scored with asl_v2/evaluate.py.

  .venv/bin/python -m asl_v2.eval_suite --n 24 --tag v3 [--words hi,my,...]
Writes outputs/eval/<tag>/{phrases.json, results.json, summary.md}.
"""

import argparse
import json
import os
import random
import subprocess

import numpy as np

from .extract import VIDEO_BASE
from .generate import build_motion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "cache", "tracks"))
               if f.endswith(".npz")) if os.path.isdir(os.path.join(ROOT, "cache", "tracks")) else []
TARGETS = ["signing_handshape_pck", "hand_shape_pck", "handshape_pck_size_norm", "hand_placement_pck", "wrist_pck", "elbow_pck", "joint_limits_ok",
           "no_body_penetration", "no_hand_penetration", "smooth_frames", "coverage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--words", default=",".join(WORDS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--assembly", default="phrase", choices=["phrase", "stitch"])
    a = ap.parse_args()
    words = a.words.split(",")
    rng = random.Random(a.seed)
    phrases = [["hi", "my", "name", "is", "john"], ["he", "win", "seven", "billion", "dollars"]]
    while len(phrases) < a.n:
        k = rng.randint(2, 6)
        phrases.append([rng.choice(words) for _ in range(k)])
    out = os.path.join(ROOT, "outputs", "eval", a.tag)
    os.makedirs(out, exist_ok=True)
    from .phrase import junction_continuity, junction_no_freeze
    paths, junction = [], {}
    for i, ph in enumerate(phrases):
        seq, info = build_motion([dict(sign=w, url=f"{VIDEO_BASE}/{w}.mp4") for w in ph],
                                 assembly=a.assembly, return_info=True)
        p = os.path.join(out, f"p{i:02d}.npy")
        np.save(p, seq.astype(np.float32))
        paths.append(p)
        if "segments" in info:
            junction[f"p{i:02d}.npy"] = (junction_continuity(seq, info["segments"]),
                                         junction_no_freeze(seq, info["segments"], static=info.get("static")))
    json.dump({f"p{i:02d}.npy": ph for i, ph in enumerate(phrases)}, open(os.path.join(out, "phrases.json"), "w"), indent=1)
    subprocess.run([os.path.join(ROOT, ".venv-blender", "bin", "python"), os.path.join(ROOT, "asl_v2", "evaluate.py"),
                    *paths, "--json", os.path.join(out, "results.json")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = json.load(open(os.path.join(out, "results.json")))
    for k, (jc, fl) in junction.items():
        res[k]["junction_continuity"] = jc
        res[k]["junction_no_freeze"] = fl
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    lines = [f"# Consistency suite `{a.tag}`: {len(phrases)} phrases, "
             f"{sum(r['frames'] for r in res.values())} frames", "",
             "| metric | mean | worst phrase | phrases ≥ 95% |", "|---|---|---|---|"]
    for m in TARGETS + (["junction_continuity", "junction_no_freeze"] if junction else []):
        v = np.array([r[m] for r in res.values() if r.get(m) is not None])
        lines.append(f"| {m} | {v.mean():.1f} | {v.min():.1f} | {(v >= 95).sum()}/{len(v)} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
