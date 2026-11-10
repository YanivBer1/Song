"""
3D hand tracking with WiLoR (ViT-H hand mesh recovery; research licence, CC-BY-NC-ND +
MANO) on top of the cached 2D tracks. Runs in its own environment:

  .venv-wilor/bin/python asl_v2/track_wilor.py john small help ...   -> cache/wilor/<word>.npz

Per frame: RTMW whole-body (rtmlib) on the person box associates each hand with its arm
and gives the hand box; WiLoR then reconstructs the hand in 3D from that crop.
Saved per side S in {L, R}:
  wl_S    (T,21,3)  MediaPipe-style landmarks: x, y px in the frame, z px relative to the
                    wrist (negative = towards the camera)  - usable depth
  wl3d_S  (T,21,3)  metres, wrist-relative, camera axes       wlR_S (T,3,3) wrist orientation
  wlok_S  (T,)      WiLoR's projection agrees with an independent 2D detector (gate)
  rtmw_S  (T,21,2), rtmwc_S (T,)   whole-body hand keypoints and score
No GitHub code is used: the backbone is the published ONNX export and the refinement head
and MANO layer are re-implemented in wilor_full.py / mano_np.py.
"""

import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CLIPS, TRACKS, OUT = (os.path.join(ROOT, "cache", d) for d in ("clips", "tracks", "wilor"))
URL_RTMW = ("https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip")
HAND_IDX = {"L": slice(91, 112), "R": slice(112, 133)}
WRIST = {"L": 15, "R": 16}


def shape_norm(p):
    q = p[:, :2] - p[:, :2].mean(0)
    return q / (np.sqrt((q ** 2).sum(1).mean()) + 1e-9)


def shape_dist(a, b):
    return float(np.linalg.norm(shape_norm(a) - shape_norm(b), axis=1).mean())


def run(word, step=1, threads=2):
    from rtmlib import RTMPose
    from wilor_full import WiLoR
    dst = os.path.join(OUT, f"{word}.npz")
    if os.path.exists(dst):
        return
    os.makedirs(OUT, exist_ok=True)
    t = dict(np.load(os.path.join(TRACKS, f"{word}.npz")))
    cap = cv2.VideoCapture(os.path.join(CLIPS, f"{word}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    T = min(len(frames), len(t["pose"]))
    h, w = frames[0].shape[:2]
    body = RTMPose(URL_RTMW, model_input_size=(288, 384), backend="onnxruntime", device="cpu")
    wil = WiLoR(threads=threads)
    out = {}
    for s in "LR":
        out[f"wl_{s}"] = np.full((T, 21, 3), np.nan, np.float32)
        out[f"wl3d_{s}"] = np.full((T, 21, 3), np.nan, np.float32)
        out[f"wlR_{s}"] = np.full((T, 3, 3), np.nan, np.float32)
        out[f"wlok_{s}"] = np.zeros(T, bool)
        out[f"wld_{s}"] = np.full(T, np.nan, np.float32)
        out[f"rtmw_{s}"] = np.full((T, 21, 2), np.nan, np.float32)
        out[f"rtmwc_{s}"] = np.zeros(T, np.float32)
    t0 = time.time()
    n = 0
    for i in range(0, T, step):
        p = t["pose"][i]
        v = p[:, 3] > 0.3
        xy = p[v, :2] if v.sum() > 4 else p[:, :2]
        x1, y1 = xy.min(0)
        x2, y2 = xy.max(0)
        mx, my = 0.12 * (x2 - x1), 0.12 * (y2 - y1)
        k, sc = body(frames[i], [[max(0, x1 - mx), max(0, y1 - 1.5 * my), min(w, x2 + mx), min(h, y2 + my)]])
        for s in "LR":
            hk, hs = k[0][HAND_IDX[s]].astype(np.float32), float(sc[0][HAND_IDX[s]].mean())
            out[f"rtmw_{s}"][i], out[f"rtmwc_{s}"][i] = hk, hs
            cands = []                                   # independent 2D readings of this hand
            if hs > 0.3:
                cands.append(hk)
            for key in ("img_", "rtm_"):
                q = t.get(key + s)
                if q is not None and not np.isnan(q[i][0, 0]) and (key == "img_" or t["rtmc_" + s][i] > 0.3):
                    cands.append(q[i][:, :2])
            present = bool(cands) and (t["src_" + s][i] > 0 or hs > 0.45
                                       or (p[WRIST[s], 3] > 0.5 and p[WRIST[s], 1] < 0.97 * h))
            if not present:
                continue
            ref = cands[0]
            c = (ref.min(0) + ref.max(0)) / 2
            o = wil(frames[i], [c[0], c[1], max(float(np.ptp(ref, 0).max()) * 1.25, 24.0)], is_right=(s == "R"))
            d = min(shape_dist(o["kp2d"], q) for q in cands)
            out[f"wl_{s}"][i], out[f"wl3d_{s}"][i], out[f"wlR_{s}"][i] = o["img"], o["kp3d"], o["R"]
            out[f"wld_{s}"][i], out[f"wlok_{s}"][i] = d, d < 0.5
            n += 1
    out["step"] = np.array(step)
    np.savez_compressed(dst, **out)
    print(f"{word}: {T} frames, {n} hand-frames, {time.time() - t0:.0f}s, "
          f"gate ok L {int(out['wlok_L'].sum())} R {int(out['wlok_R'].sum())}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    step = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--step=")), 1))
    for word in args:
        try:
            run(word, step)
        except Exception as e:  # noqa: BLE001
            print("skip", word, repr(e), flush=True)
