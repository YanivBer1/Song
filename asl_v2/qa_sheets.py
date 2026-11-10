"""
Visual QA sheets: every tracked sign, source video frame next to the avatar at the same
moment and in the same framing (calibrated orthographic view), zoomed on the hands.

  .venv/bin/python -m asl_v2.qa_sheets prepare [--words a,b,c] [--per 6]
  .venv-blender/bin/python asl_v2/render_human.py x x --batch outputs/audit/qa/jobs.json \
        --res 640x360 --samples 8 --audit-view
  .venv/bin/python -m asl_v2.qa_sheets compose          # -> outputs/audit/qa/sheets/<word>.png

Each sign is built as a one-word phrase through the full pipeline (tracking fusion,
stroke segmentation, hand fit), so the sheets show exactly what a sentence would use.
"""

import argparse
import json
import os

import cv2
import numpy as np

from .extract import VIDEO_BASE, clip_path
from .motion import LH, RH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA = os.path.join(ROOT, "outputs", "audit", "qa")
W, H = 640, 360


def _words():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "cache", "tracks")) if f.endswith(".npz"))


def prepare(words, per):
    from .generate import build_motion, save_motion, tracked_keypoints
    from .motion import process_clip
    os.makedirs(QA, exist_ok=True)
    jobs, meta = [], {}
    for i, w in enumerate(words):
        seq, info = build_motion([dict(sign=w, url=f"{VIDEO_BASE}/{w}.mp4")], return_info=True)
        path = os.path.join(QA, f"{w}.npy")
        save_motion(path, seq, info)
        a, b = [(a, b) for k, a, b, _ in info["segments"] if k == "stroke"][0]
        frames = sorted({int(x) for x in np.linspace(a, b - 1, per).round()})
        # map each output frame to the closest source frame of the clip
        clip, _ = process_clip(tracked_keypoints(w, solver=False), return_meta=True, trim=False)

        def key(x):
            return np.r_[(x[RH:RH + 21] - x[RH])[:, :2].ravel(), (x[LH:LH + 21] - x[LH])[:, :2].ravel(),
                         0.5 * x[RH, :2], 0.5 * x[LH, :2]]
        src = [int(np.argmin([np.linalg.norm(key(seq[f]) - key(c)) for c in clip])) for f in frames]
        jobs.append(dict(motion=path, frames=frames, prefix=os.path.join(QA, f"{w}_av")))
        meta[w] = dict(frames=frames, src=src)
        if i % 10 == 0:
            print(f"{i + 1}/{len(words)} prepared", flush=True)
    json.dump(jobs, open(os.path.join(QA, "jobs.json"), "w"))
    json.dump(meta, open(os.path.join(QA, "meta.json"), "w"))
    print("prepared", len(jobs), "signs")


def compose(zoom=True):
    from .track import track
    meta = json.load(open(os.path.join(QA, "meta.json")))
    out = os.path.join(QA, "sheets")
    os.makedirs(out, exist_ok=True)
    n = 0
    for w, m in meta.items():
        t = track(w)
        cap = cv2.VideoCapture(clip_path(w))
        src = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            src.append(f)
        tiles = []
        for fr, j in zip(m["frames"], m["src"]):
            av_path = os.path.join(QA, f"{w}_av_{fr:04d}.png")
            if not os.path.exists(av_path) or j >= len(src):
                continue
            av = cv2.imread(av_path)
            p = t["pose"][j]
            mid = (p[11, :2] + p[12, :2]) / 2
            sw = np.linalg.norm(p[11, :2] - p[12, :2])
            hh = 2.16 * sw
            ww = hh * 16 / 9
            cy = mid[1] + 0.30 * sw
            M = np.array([[W / ww, 0, W / 2 - W / ww * mid[0]], [0, H / hh, H / 2 - H / hh * cy]])
            s_img = cv2.warpAffine(src[j], M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            pts = np.concatenate([t["img_R"][j][:, :2], t["img_L"][j][:, :2]])
            pts = pts[~np.isnan(pts[:, 0])]
            if zoom and len(pts):
                q = pts @ M[:, :2].T + M[:, 2]
                c = (q.min(0) + q.max(0)) / 2
                half = int(np.clip(max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) / 2 + 45, 90, 180))
                x0 = int(np.clip(c[0] - half, 0, W - 2 * half))
                y0 = int(np.clip(c[1] - half, 0, H - 2 * half))
                s_img = cv2.resize(s_img[y0:y0 + 2 * half, x0:x0 + 2 * half], (300, 300), interpolation=cv2.INTER_CUBIC)
                av = cv2.resize(av[y0:y0 + 2 * half, x0:x0 + 2 * half], (300, 300), interpolation=cv2.INTER_CUBIC)
            cv2.putText(s_img, f"{w.upper()} source f{j}", (6, 18), 0, 0.5, (30, 30, 220), 1)
            cv2.putText(av, f"avatar f{fr}", (6, 18), 0, 0.5, (30, 30, 220), 1)
            tiles.append(np.hstack([s_img, av]))
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(os.path.join(out, f"{w}.png"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
            n += 1
    print(n, "sheets in", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "compose"])
    ap.add_argument("--words")
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.words.split(",") if a.words else _words(), a.per)
    else:
        compose()
