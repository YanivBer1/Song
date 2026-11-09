"""
Real-footage output: the recorded human signers themselves, stitched into one
sentence video.

Each dictionary clip is affinely aligned (scale + translate) so the signer's
shoulders land in the same place at the same width, signs are sped up
slightly, cut at the closest-matching frames between neighbouring signs, and
joined with short cross-dissolves. Output is cropped around the signer.

  python -m asl_v2.realvideo hi,my,name,is,john -o outputs/v2/john_real.mp4
"""

import argparse
import subprocess

import cv2
import numpy as np

from .extract import clip_path, keypoints

OUT_W, OUT_H = 960, 540
SHOULDER_W = 165          # target shoulder width in output pixels
CENTER = (480, 255)       # target shoulder midpoint in output pixels
SPEED = 1.2
DISSOLVE = 5              # frames
WINDOW = 9                # frames searched at each boundary for the best cut


def _frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def _align_matrix(kp):
    pose = kp["pose"]
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    mid = np.median((ls + rs) / 2, axis=0)
    width = np.median(np.linalg.norm(ls - rs, axis=1))
    s = SHOULDER_W / width
    return np.array([[s, 0, CENTER[0] - s * mid[0]], [0, s, CENTER[1] - s * mid[1]]]), s, mid


def _hand_track(kp, M):
    """Per-frame aligned positions of both wrists (hand wrist if detected, else pose wrist)."""
    pose = kp["pose"]
    out = []
    for side, hand_key, pw in ((0, "lh", 15), (1, "rh", 16)):
        h = kp[hand_key][:, 0, :2]
        p = np.where(np.isnan(h), pose[:, pw, :2], h)
        out.append(p @ M[:, :2].T + M[:, 2])
    return np.stack(out, 1)          # (T, 2, 2)


def build(signs, out_path):
    clips = []
    for s in signs:
        kp = keypoints(s)
        frames = _frames(clip_path(s))
        n = len(kp["pose"])
        frames = frames[:n] if len(frames) >= n else frames
        M, _, _ = _align_matrix(kp)
        track = _hand_track(kp, M)[:len(frames)]
        idx = np.round(np.arange(0, len(frames), SPEED)).astype(int)
        idx = idx[idx < len(frames)]
        clips.append(dict(frames=[frames[i] for i in idx], track=track[idx], M=M))

    # closest-frame cuts between neighbouring signs
    for a, b in zip(clips[:-1], clips[1:]):
        best = (np.inf, len(a["frames"]) - 1, 0)
        ta, tb = a["track"], b["track"]
        for i in range(max(len(ta) - WINDOW, 0), len(ta)):
            for j in range(min(WINDOW, len(tb))):
                cost = np.linalg.norm(ta[i] - tb[j], axis=1).max() + 2.0 * ((len(ta) - 1 - i) + j)
                if cost < best[0]:
                    best = (cost, i, j)
        _, i, j = best
        a["frames"], a["track"] = a["frames"][:i + 1], a["track"][:i + 1]
        b["frames"], b["track"] = b["frames"][j:], b["track"][j:]

    def warp(frame, M):
        frame = frame.copy()
        h, w = frame.shape[:2]
        frame[: int(0.10 * h), int(0.76 * w):] = 255     # source watermark (top-right) on white bg
        return cv2.warpAffine(frame, M, (OUT_W, OUT_H), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "17",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    prev_tail = None
    for c in clips:
        warped = [warp(f, c["M"]) for f in c["frames"]]
        if prev_tail is not None:
            k = min(DISSOLVE, len(warped) - 1, len(prev_tail))
            for t in range(k):
                w = (t + 1) / (k + 1)
                w = 0.5 - 0.5 * np.cos(np.pi * w)
                proc.stdin.write(cv2.addWeighted(prev_tail[t], 1 - w, warped[t], w, 0).tobytes())
            warped = warped[k:]
        body = warped[:-DISSOLVE] if c is not clips[-1] and len(warped) > 2 * DISSOLVE else warped
        for f in body:
            proc.stdin.write(f.tobytes())
        prev_tail = warped[len(body):] if len(body) < len(warped) else None
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signs", help="comma-separated sign words")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    build(a.signs.split(","), a.output)


if __name__ == "__main__":
    main()
