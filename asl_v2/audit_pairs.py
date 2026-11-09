"""
All-orders transition audit: every ordered pair of tracked signs (A -> B) is
assembled exactly as inside a sentence (asl_v2.phrase) and the A->B junction is
scored. Writes outputs/audit/pairs.json + pairs_summary.md.

  .venv/bin/python -m asl_v2.audit_pairs [--words a,b,c]

Per junction:
  freeze        hands frozen (both wrists < 1.4 mm/frame) for >= 4 frames around it
                (skipped if A or B is a static sign, whose brief hold is its content)
  vel_jump      acceleration at the hand-off > 1.5 x the 95th pct inside real strokes
  loop          wrist path length > 2.2 x straight distance (and distance > 0.15 sw)
  shape_pop     max per-frame change of the wrist-relative hand configuration
                > 2.5 x the 95th pct of that change inside real strokes
  body_pass     a wrist passes behind the torso front plane (z > -0.25 sw) while
                in front of the torso (|x| < 0.55, 0.05 < y < 1.6)
  frames        transition length
"""

import argparse
import json
import os
import time

import numpy as np

from .generate import tracked_keypoints
from .motion import LH, RH, process_clip
from .phrase import assemble, junction_no_freeze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "audit")


def _shape_change(seq):
    out = np.zeros(len(seq))
    for off in (LH, RH):
        h = seq[:, off:off + 21] - seq[:, off:off + 1]
        d = np.zeros(len(seq))
        d[1:] = np.sqrt(((h[1:] - h[:-1]) ** 2).sum(2).mean(1))
        out = np.maximum(out, d)
    return out


def audit(words):
    items = {}
    t0 = time.time()
    for w in words:
        try:
            items[w] = process_clip(tracked_keypoints(w), return_meta=True, trim=False)
        except Exception as e:          # noqa: BLE001
            print("skip", w, e)
    words = [w for w in words if w in items]
    print(f"{len(words)} signs prepared in {time.time() - t0:.0f}s", flush=True)

    # stroke-internal reference statistics (what real signing looks like)
    ref_acc, ref_shape = [], []
    for w in words:
        seq, info = assemble([items[w]])
        for kind, a, b, _ in info["segments"]:
            if kind == "stroke" and b - a > 6:
                s = seq[a + 2:b - 2]
                wr = np.stack([s[:, LH], s[:, RH]], 1)
                ref_acc.append(np.linalg.norm(wr[2:] - 2 * wr[1:-1] + wr[:-2], axis=2).max(1))
                ref_shape.append(_shape_change(s)[1:])
    acc_thr = max(1.5 * np.percentile(np.concatenate(ref_acc), 95), 0.004)
    shape_thr = 2.5 * np.percentile(np.concatenate(ref_shape), 95)

    results = {}
    for a in words:
        for b in words:
            seq, info = assemble([items[a], items[b]])
            segs = info["segments"]
            tr = [s for s in segs if s[0] == "transition"]
            _, ta, tb, _ = tr[1]                       # the A -> B junction
            wr = np.stack([seq[:, LH], seq[:, RH]], 1)
            acc = np.zeros(len(seq))
            acc[1:-1] = np.linalg.norm(wr[2:] - 2 * wr[1:-1] + wr[:-2], axis=2).max(1)
            vel_jump = bool(max(acc[max(ta - 1, 1):ta + 2].max(), acc[tb - 1:min(tb + 2, len(seq) - 1)].max()) > acc_thr)
            loop = False
            for h in (0, 1):
                path = wr[ta - 1:tb + 1, h]
                L = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                D = np.linalg.norm(path[-1] - path[0])
                loop |= bool(D > 0.15 and L > 2.2 * D)
            shape_pop = bool(_shape_change(seq)[ta:tb + 1].max() > shape_thr)
            p = wr[ta:tb + 1].reshape(-1, 3)
            inside = (np.abs(p[:, 0]) < 0.55) & (p[:, 1] > 0.05) & (p[:, 1] < 1.6)
            body_pass = bool(np.any(inside & (p[:, 2] > -0.25)))
            freeze = junction_no_freeze(seq, segs, static=info["static"]) < 100
            results[f"{a}>{b}"] = dict(freeze=bool(freeze), vel_jump=vel_jump, loop=loop, shape_pop=shape_pop,
                                       body_pass=body_pass, frames=int(tb - ta))
    return results, dict(acc_thr=acc_thr, shape_thr=shape_thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words")
    a = ap.parse_args()
    words = a.words.split(",") if a.words else sorted(
        os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "cache", "tracks")) if f.endswith(".npz"))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res, thr = audit(words)
    json.dump(dict(thresholds=thr, pairs=res), open(os.path.join(OUT, "pairs.json"), "w"))
    n = len(res)
    keys = ["freeze", "vel_jump", "loop", "shape_pop", "body_pass"]
    clean = sum(not any(r[k] for k in keys) for r in res.values())
    lines = [f"# All-orders transition audit: {n} ordered sign pairs ({time.time() - t0:.0f}s)", "",
             f"**Clean junctions (no defect of any kind): {clean}/{n} = {100 * clean / n:.2f}%**", "",
             "| defect | pairs | % |", "|---|---|---|"]
    for k in keys:
        c = sum(r[k] for r in res.values())
        lines.append(f"| {k} | {c} | {100 * c / n:.2f} |")
    fr = np.array([r["frames"] for r in res.values()])
    lines += ["", f"Transition length: median {np.median(fr):.0f} frames ({np.median(fr) / 30 * 1000:.0f} ms), "
              f"5-95% {np.percentile(fr, 5):.0f}-{np.percentile(fr, 95):.0f} frames", ""]
    for k in keys:
        bad = {}
        for pair, r in res.items():
            if r[k]:
                for w in pair.split(">"):
                    bad[w] = bad.get(w, 0) + 1
        top = sorted(bad.items(), key=lambda x: -x[1])[:10]
        lines.append(f"- most involved in `{k}`: " + ", ".join(f"{w} ({c})" for w, c in top))
    open(os.path.join(OUT, "pairs_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
