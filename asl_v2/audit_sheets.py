"""
Compose per-word visual QA sheets: source frame | avatar frame at 4 points of each
sign's stroke, both cropped to the same chest+hands framing.

  .venv/bin/python -m asl_v2.audit_sheets      # -> outputs/audit/sheets/<word>.png
"""

import json
import os

import cv2
import numpy as np

from .extract import clip_path
from .track import track

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "outputs", "audit", "words")
OUT = os.path.join(ROOT, "outputs", "audit", "sheets")


def frames_of(word):
    cap = cv2.VideoCapture(clip_path(word))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main(size=(480, 270)):
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(WD, "meta.json")))
    W, H = size
    n = 0
    for word, m in meta.items():
        t = track(word)
        src = frames_of(word)
        rows = []
        for i in m["frames"]:
            av_path = os.path.join(WD, f"{word}_av_{i:04d}.png")
            if not os.path.exists(av_path) or i >= len(src):
                continue
            p = t["pose"][i]
            mid = (p[11, :2] + p[12, :2]) / 2
            sw = np.linalg.norm(p[11, :2] - p[12, :2])
            hh = 2.16 * sw
            ww = hh * 16 / 9
            cy = mid[1] + 0.30 * sw
            M = np.array([[W / ww, 0, W / 2 - W / ww * mid[0]], [0, H / hh, H / 2 - H / hh * cy]])
            s_img = cv2.warpAffine(src[i], M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
            a_img = cv2.resize(cv2.imread(av_path), (W, H))
            cv2.putText(s_img, f"{word.upper()} source f{i}", (8, 22), 0, 0.6, (30, 30, 200), 2)
            cv2.putText(a_img, f"avatar f{i}", (8, 22), 0, 0.6, (30, 30, 200), 2)
            rows.append(np.hstack([s_img, a_img]))
        if rows:
            cv2.imwrite(os.path.join(OUT, f"{word}.png"), np.vstack(rows))
            n += 1
    print(f"{n} sheets in {OUT}")


if __name__ == "__main__":
    main()
