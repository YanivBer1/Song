"""
Second, independent hand detector (RTMPose-m hand, Apache-2.0, via rtmlib/onnxruntime)
and per-frame fusion with the MediaPipe track.

Why: within one held handshape MediaPipe sometimes flips between a correct reading
and a wrong one (e.g. a fist facing the camera read as an open hand upside down).
Two detectors rarely fail on the same frame in the same way. For every frame we keep
the reading that is consistent over time (dynamic programming over the two
candidates); where they agree the two are averaged.

  augment(word)          -> adds rtm_L/rtm_R (T,21,2) and rtmc_L/rtmc_R (T,) to the cached track
  fused(track)           -> dict(img_L, img_R (T,21,3), noz_L, noz_R (T,) bool)
                            noz = 2D came from RTMPose only; MediaPipe's depth must not be used
"""

import os

import cv2
import numpy as np

from .extract import clip_path
from .track import POSE_HAND, TRACK_DIR, track

MODEL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
         "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from rtmlib import RTMPose
        _MODEL = RTMPose(MODEL, model_input_size=(256, 256), backend="onnxruntime", device="cpu")
    return _MODEL


def augment(word):
    path = os.path.join(TRACK_DIR, f"{word}.npz")
    t = dict(np.load(path)) if os.path.exists(path) else track(word)
    if "rtm_R" in t:
        return t
    model = _model()
    cap = cv2.VideoCapture(clip_path(word))
    T = len(t["pose"])
    out = {f"rtm_{s}": np.full((T, 21, 2), np.nan, np.float32) for s in "LR"}
    out.update({f"rtmc_{s}": np.zeros(T, np.float32) for s in "LR"})
    last = {s: None for s in "LR"}
    i = 0
    while i < T:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        pose = t["pose"][i]
        for s in "LR":
            mp_h = t[f"img_{s}"][i]
            wr, pk, ix, th, el = POSE_HAND[s]
            if not np.isnan(mp_h[0, 0]):
                c = (mp_h[:, :2].min(0) + mp_h[:, :2].max(0)) / 2
                size = max(np.ptp(mp_h[:, 0]), np.ptp(mp_h[:, 1])) * 1.6 + 10
            elif pose[wr, 3] > 0.5 and pose[wr, 1] < 0.98 * h:
                if last[s] is not None:
                    c, size = last[s]
                    c = c + (pose[wr, :2] - last_wr[s])
                else:
                    fore = pose[wr, :2] - pose[el, :2]
                    c = pose[[wr, pk, ix, th], :2].mean(0) + 0.25 * fore
                    size = max(np.linalg.norm(fore) * 1.3, 44)
            else:
                last[s] = None
                continue
            bb = [c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2]
            k, sc = model(frame, [bb])
            out[f"rtm_{s}"][i] = k[0]
            out[f"rtmc_{s}"][i] = float(np.mean(sc[0]))
            if out[f"rtmc_{s}"][i] > 0.4:
                kk = k[0]
                last[s] = ((kk.min(0) + kk.max(0)) / 2, max(np.ptp(kk[:, 0]), np.ptp(kk[:, 1])) * 1.6 + 10)
        last_wr = {s: pose[POSE_HAND[s][0], :2].copy() for s in "LR"}
        i += 1
    cap.release()
    t.update(out)
    np.savez_compressed(path, **t)
    return t


def _shape(p):
    """Translation/scale-normalised 2D hand shape."""
    q = p - p.mean(0)
    return q / (np.sqrt((q ** 2).sum(1).mean()) + 1e-9)


def fuse_side(mp, rtm, rtm_conf, other_mp=None, min_conf=0.30, agree=0.30, w_un=0.15):
    """mp (T,21,3) with NaN rows, rtm (T,21,2). Returns (img (T,21,3), noz (T,), choice (T,))
    choice: 0 none, 1 MediaPipe, 2 RTMPose, 3 both (averaged)."""
    T = len(mp)
    has_mp = ~np.isnan(mp[:, 0, 0])
    has_rt = (~np.isnan(rtm[:, 0, 0])) & (rtm_conf > min_conf)
    if other_mp is not None:                       # RTMPose box may have locked on to the other hand
        for t in np.flatnonzero(has_rt & ~np.isnan(other_mp[:, 0, 0])):
            span = max(np.ptp(rtm[t, :, 0]), np.ptp(rtm[t, :, 1]), 1e-6)
            d_other = np.linalg.norm(other_mp[t, :, :2] - rtm[t], axis=1).mean()
            d_own = np.linalg.norm(mp[t, :, :2] - rtm[t], axis=1).mean() if has_mp[t] else np.inf
            if d_other < 0.25 * span and d_other < d_own:
                has_rt[t] = False
    cand = [mp[:, :, :2], rtm]
    avail = np.stack([has_mp, has_rt], 1)
    INF = 1e9
    cost = np.full((T, 2), INF)
    back = np.zeros((T, 2), int)
    prev_t = None
    un = np.stack([np.zeros(T), w_un * (1.0 - rtm_conf)], 1)        # slight preference by confidence
    for t in range(T):
        if not avail[t].any():
            continue
        for k in (0, 1):
            if not avail[t, k]:
                continue
            if prev_t is None:
                cost[t, k] = un[t, k]
                continue
            best = INF
            for j in (0, 1):
                if cost[prev_t, j] >= INF:
                    continue
                gap = t - prev_t
                d = np.linalg.norm(_shape(cand[k][t]) - _shape(cand[j][prev_t]), axis=1).mean() / np.sqrt(gap)
                c = cost[prev_t, j] + d
                if c < best:
                    best, back[t, k] = c, j
            cost[t, k] = best + un[t, k]
        prev_t = t
    choice = np.zeros(T, int)
    ts = [t for t in range(T) if avail[t].any()]
    if ts:
        k = int(np.argmin(cost[ts[-1]]))
        for a in range(len(ts) - 1, -1, -1):
            choice[ts[a]] = k + 1
            k = back[ts[a], k]
    img = np.full((T, 21, 3), np.nan, np.float32)
    noz = np.zeros(T, bool)
    for t in ts:
        both_ok = avail[t].all() and np.linalg.norm(_shape(cand[0][t]) - _shape(cand[1][t]), axis=1).mean() < agree
        if both_ok:
            img[t, :, :2] = (mp[t, :, :2] + rtm[t]) / 2
            img[t, :, 2] = mp[t, :, 2]
            choice[t] = 3
        elif choice[t] == 1:
            img[t] = mp[t]
        else:
            img[t, :, :2] = rtm[t]
            img[t, :, 2] = 0.0
            noz[t] = True
    return img, noz, choice


def fused(t):
    out = {}
    for s, o in (("L", "R"), ("R", "L")):
        img, noz, ch = fuse_side(t[f"img_{s}"], t[f"rtm_{s}"], t[f"rtmc_{s}"], other_mp=t[f"img_{o}"])
        out[f"img_{s}"], out[f"noz_{s}"], out[f"choice_{s}"] = img, noz, ch
    return out


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        t = augment(w)
        f = fused(t)
        for s in "LR":
            c = f[f"choice_{s}"]
            print(f"{w:10s} {s}: none {int((c == 0).sum()):3d}  MediaPipe {int((c == 1).sum()):3d}  "
                  f"RTMPose {int((c == 2).sum()):3d}  both agree {int((c == 3).sum()):3d}")
