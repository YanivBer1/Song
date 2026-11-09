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
WORDS = "hi my name is john he win pm seven billion dollars".split()
TARGETS = ["hand_shape_pck", "handshape_pck_size_norm", "wrist_pck", "elbow_pck", "joint_limits_ok",
           "no_body_penetration", "no_hand_penetration", "smooth_frames", "coverage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--words", default=",".join(WORDS))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    words = a.words.split(",")
    rng = random.Random(a.seed)
    phrases = [["hi", "my", "name", "is", "john"], ["he", "win", "seven", "billion", "dollars"]]
    while len(phrases) < a.n:
        k = rng.randint(2, 6)
        phrases.append([rng.choice(words) for _ in range(k)])
    out = os.path.join(ROOT, "outputs", "eval", a.tag)
    os.makedirs(out, exist_ok=True)
    paths = []
    for i, ph in enumerate(phrases):
        seq = build_motion([dict(sign=w, url=f"{VIDEO_BASE}/{w}.mp4") for w in ph])
        p = os.path.join(out, f"p{i:02d}.npy")
        np.save(p, seq.astype(np.float32))
        paths.append(p)
    json.dump({f"p{i:02d}.npy": ph for i, ph in enumerate(phrases)}, open(os.path.join(out, "phrases.json"), "w"), indent=1)
    subprocess.run([os.path.join(ROOT, ".venv-blender", "bin", "python"), os.path.join(ROOT, "asl_v2", "evaluate.py"),
                    *paths, "--json", os.path.join(out, "results.json")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = json.load(open(os.path.join(out, "results.json")))
    lines = [f"# Consistency suite `{a.tag}`: {len(phrases)} phrases, "
             f"{sum(r['frames'] for r in res.values())} frames", "",
             "| metric | mean | worst phrase | phrases ≥ 95% |", "|---|---|---|---|"]
    for m in TARGETS:
        v = np.array([r[m] for r in res.values() if r.get(m) is not None])
        lines.append(f"| {m} | {v.mean():.1f} | {v.min():.1f} | {(v >= 95).sum()}/{len(v)} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
