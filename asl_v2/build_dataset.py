"""
Track a sample of the sign dictionary (frame-by-frame, asl_v2.track) to build a
training/evaluation set for hand-pose priors and transition statistics.

  python -m asl_v2.build_dataset --n 300 --shard 0 --shards 2
"""

import argparse
import os
import random
import sys
import time

import pandas as pd

from .track import TRACK_DIR, track

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream", "videos_rows.csv")
CORE = "hi my name is john he win pm seven billion dollars".split()


def sample_words(n, seed=0):
    df = pd.read_csv(CSV)
    words = [w for w in df["word"].astype(str) if w.isalpha() and len(w) > 1]
    random.Random(seed).shuffle(words)
    return CORE + [w for w in words if w not in CORE][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    words = sample_words(a.n)[a.shard::a.shards]
    t0 = time.time()
    for i, w in enumerate(words):
        if os.path.exists(os.path.join(TRACK_DIR, f"{w}.npz")):
            continue
        try:
            track(w)
        except Exception as e:  # missing/corrupt clip: skip
            print(f"skip {w}: {e}", file=sys.stderr, flush=True)
        if i % 10 == 0:
            print(f"shard {a.shard}: {i + 1}/{len(words)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"shard {a.shard} done", flush=True)


if __name__ == "__main__":
    main()
