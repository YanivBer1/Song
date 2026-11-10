"""
Avatar-space hand fitting: solve the avatar's OWN hand skeleton (hand rotation +
finger joint angles, within human range of motion) so that its joints project onto
the signer's tracked 2D landmarks.

Why: the image-plane (2D) positions from tracking are reliable, the depth is not.
Aiming bones at noisy 3D points gave loose fists and flipped palms. Here the 2D
evidence drives the solution and anatomy resolves the depth ambiguity: fingers only
curl towards the palm, so a fist seen from any side has exactly one valid pose.

  fit_sequence(seq, segments=None) -> dict(R (T,2,3,3), F (T,2,15,3,3), rms (T,2), valid (T,))
side index 0 = 'l', 1 = 'r'; R = hand bone orientation in armature (Blender) space,
F = local rotation of each finger bone (thumb, index, middle, ring, pinky x 01,02,03).
Pure numpy/scipy (runs outside Blender); the rig comes from models/rig_hand.json.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
LH, RH = 33, 54
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
KNUCK = [5, 9, 13, 17]
IDX = np.array([2, 3, 4] + list(range(5, 21)))
W2D = np.ones(len(IDX))
W2D[np.isin(IDX, [4, 8, 12, 16, 20])] = 1.3
WZ, WT_ANG, WT_ROT, WREG, WWRIST = 0.30, 0.03, 0.06, 0.004, 0.7

# bounds (rad), relative to the avatar's relaxed rest pose
B_FLEX1, B_ABD, B_FLEX2, B_FLEX3 = (-0.35, 1.5), (-0.4, 0.4), (-0.05, 1.9), (-0.05, 1.45)
B_TH1, B_TH2, B_THX, B_TH3 = (-0.7, 0.7), (-0.3, 1.15), (-0.3, 0.3), (-0.35, 1.4)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _exp(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if np.pi - th < 1e-4:                       # near 180 deg: use the symmetric part
        A = (R + np.eye(3)) / 2
        ax = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(ax))
        ax = A[:, i] / (ax[i] + 1e-12)
        return ax / np.linalg.norm(ax) * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _frame(y, n):
    y = y / (np.linalg.norm(y) + 1e-12)
    n = n - y * np.dot(n, y)
    n = n / (np.linalg.norm(n) + 1e-12)
    return np.stack([np.cross(y, n), y, n], 1)


class Hand:
    """Rest skeleton + forward kinematics of one avatar hand."""

    def __init__(self, side, rig, cal):
        s = self.side = side
        hand = rig[f"hand_{s}"]
        self.R_rest = np.array(hand["rest"])
        h0 = np.array(hand["head"])
        self.bones = []
        for fn in FINGERS:
            for k in ("01", "02", "03"):
                b = rig[f"{fn}_{k}_{s}"]
                par = rig[b["parent"]]
                self.bones.append(dict(
                    name=f"{fn}_{k}", Mrel=np.array(par["rest"]).T @ np.array(b["rest"]),
                    orel=np.array(par["rest"]).T @ (np.array(b["head"]) - np.array(par["head"])),
                    length=float(np.linalg.norm(np.array(b["tail"]) - np.array(b["head"])))))
        rest = self.fk(self.R_rest, [np.eye(3)] * 15)
        n = np.cross(rest[5] - rest[0], rest[17] - rest[0])
        n /= np.linalg.norm(n)
        self.palm_local = self.R_rest.T @ n
        curl = np.mean([np.dot(rest[b + 3] - rest[b], n) for b in (5, 9, 13, 17)])
        palmar = n * np.sign(curl)                       # relaxed fingers curl towards the palm
        self.sign = np.ones(15)                          # flexion sign per bone (X axis; thumb 02/03: Z)
        for i, b in enumerate(self.bones):
            L = [np.eye(3)] * 15
            thumb = b["name"].startswith("thumb")
            L[i] = (_rz if thumb and i > 0 else _rx)(0.3)
            moved = self.fk(self.R_rest, L)
            tip = (i // 3) * 4 + 4
            if thumb:
                d0 = np.linalg.norm(rest[tip] - rest[17])
                self.sign[i] = 1.0 if np.linalg.norm(moved[tip] - rest[17]) < d0 else -1.0
            else:
                self.sign[i] = 1.0 if np.dot(moved[tip] - rest[tip], palmar) > 0 else -1.0
        self.finger_len = sum(self.bones[i]["length"] for i in range(3, 15))
        self.hand_len = float(np.linalg.norm(rest[9] - rest[0]) + sum(self.bones[i]["length"] for i in (6, 7, 8)))
        # MediaPipe's wrist landmark sits a little distal of the bone's wrist joint (calibrated)
        bh = cal["bone_heads_canon"]
        hd, md = np.array(bh[f"hand_{s}"]), np.array(bh[f"middle_01_{s}"])
        u = (md - hd) / np.linalg.norm(md - hd)
        mpw = cal["hands"].get(s.upper())
        self.wrist_off = float(np.dot(np.array(mpw[0]) - hd, u)) * cal["mp_shoulder_width_m"] if mpw else 0.0
        self.mid_dir_local = self.R_rest.T @ ((rest[9] - rest[0]) / np.linalg.norm(rest[9] - rest[0]))
        lo, hi = [-3.3] * 3, [3.3] * 3
        for _ in range(4):
            lo += [B_FLEX1[0], B_ABD[0], B_FLEX2[0], B_FLEX3[0]]
            hi += [B_FLEX1[1], B_ABD[1], B_FLEX2[1], B_FLEX3[1]]
        lo += [B_TH1[0]] * 3 + [B_TH2[0], B_THX[0], B_TH3[0]] + [-0.25]
        hi += [B_TH1[1]] * 3 + [B_TH2[1], B_THX[1], B_TH3[1]] + [0.25]   # log scale: proportions differ
        self.lo, self.hi = np.array(lo), np.array(hi)

    def fk(self, R_h, L):
        J = np.zeros((21, 3))
        posed, head = R_h, np.zeros(3)
        for f in range(5):
            pp, ph = R_h, np.zeros(3)
            for k in range(3):
                b = self.bones[f * 3 + k]
                h = ph + pp @ b["orel"]
                p = pp @ b["Mrel"] @ L[f * 3 + k]
                J[f * 4 + 1 + k] = h
                pp, ph = p, h
            J[f * 4 + 4] = ph + pp[:, 1] * b["length"]
        return J

    def locals_from(self, x):
        """x[3:] -> 15 local rotation matrices (order thumb, index, middle, ring, pinky)."""
        s = self.sign
        L = [None] * 15
        for f in range(4):                               # index..pinky
            a = x[3 + 4 * f: 7 + 4 * f]
            i = 3 + 3 * f
            L[i] = _rz(a[1]) @ _rx(s[i] * a[0])
            L[i + 1] = _rx(s[i + 1] * a[2])
            L[i + 2] = _rx(s[i + 2] * a[3])
        t = x[19:25]
        L[0] = _exp(t[0:3])
        L[1] = _rz(s[1] * t[3]) @ _rx(t[4])
        L[2] = _rz(s[2] * t[5])
        return L

    def solve(self, T, R_init, x_prev, R_prev):
        """T: (21,3) target landmarks in Blender axes (any scale, any origin)."""
        data_len = sum(np.linalg.norm(T[b + k + 1] - T[b + k]) for b in (5, 9, 13, 17) for k in range(3))
        k = self.finger_len / (data_len + 1e-9)
        Tn = k * (T - T[KNUCK].mean(0)) / self.hand_len

        def res(x):
            R_h = R_init @ _exp(x[:3])
            A = self.fk(R_h, self.locals_from(x))
            An = np.exp(x[25]) * (A - A[KNUCK].mean(0)) / self.hand_len
            d = An[IDX] - Tn[IDX]
            wr = np.exp(x[25]) * (A[0] + self.wrist_off * (R_h @ self.mid_dir_local) - A[KNUCK].mean(0)) / self.hand_len - Tn[0]
            out = [(W2D[:, None] * d[:, [0, 2]]).ravel(), WZ * d[:, 1], WWRIST * wr[[0, 2]], WREG * x[3:25]]
            if x_prev is not None:
                out += [WT_ANG * (x[3:25] - x_prev[3:25]), WT_ROT * _log(R_prev.T @ R_h), 0.3 * (x[25:] - x_prev[25:])]
            return np.concatenate(out)

        x0 = np.zeros(26) if x_prev is None else np.r_[0, 0, 0, x_prev[3:]]
        x0 = np.clip(x0, self.lo + 1e-6, self.hi - 1e-6)
        sol = least_squares(res, x0, bounds=(self.lo, self.hi), method="trf", max_nfev=30, xtol=1e-4, ftol=1e-5)
        R_h = R_init @ _exp(sol.x[:3])
        A = self.fk(R_h, self.locals_from(sol.x))
        d = (np.exp(sol.x[25]) * (A - A[KNUCK].mean(0)) / self.hand_len - Tn)[IDX][:, [0, 2]]
        return sol.x, R_h, float(sol.cost), float(np.sqrt((d ** 2).sum(1).mean()))

    def init_rot(self, T):
        d = T[9] - T[0]
        n = np.cross(T[5] - T[0], T[17] - T[0])
        return _frame(d, n) @ _frame(np.array([0, 1.0, 0]), self.palm_local).T


_RIG = None


def _load():
    global _RIG
    if _RIG is None:
        rig = json.load(open(os.path.join(HERE, "models", "rig_hand.json")))
        cal = json.load(open(os.path.join(HERE, "models", "avatar_calib.json")))
        _RIG = {s: Hand(s, rig, cal) for s in "lr"}
    return _RIG


def _to_bl(p):
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def _fit_side(args):
    side, targets, mask = args
    hand = _load()[side]
    T = len(targets)
    X = np.full((T, 26), np.nan)
    R = np.full((T, 3, 3), np.nan)
    rms = np.full(T, np.nan)
    x_prev = R_prev = None
    for t in range(T):
        if not mask[t]:
            x_prev = R_prev = None                       # new block after a transition
            continue
        tg = targets[t]
        cands = []
        mirror = tg * np.array([1.0, -1.0, 1.0])
        if R_prev is not None:
            # The depth SIGN of a tracked hand is unreliable frame to frame (a hand and its
            # depth mirror look the same in the image). Fit both readings from the previous
            # pose; the temporal term then keeps the one that continues the motion.
            cands.append(hand.solve(tg, R_prev, x_prev, R_prev))
            cands.append(hand.solve(mirror, R_prev, x_prev, R_prev))
        if R_prev is None or min(c[3] for c in cands) > 0.07:   # (re)initialise from the data
            for init in (hand.init_rot(tg), hand.init_rot(mirror)):
                cands.append(hand.solve(tg, init, x_prev, R_prev))
        x, Rh, _, r = min(cands, key=lambda c: c[2])
        x = np.r_[0, 0, 0, x[3:]]
        X[t], R[t], rms[t] = x, Rh, r
        x_prev, R_prev = x, Rh
    return side, X, R, rms


def _smooth_blocks(mask, fn):
    t, T = 0, len(mask)
    while t < T:
        if not mask[t]:
            t += 1
            continue
        u = t
        while u < T and mask[u]:
            u += 1
        fn(t, u)
        t = u


def fit_sequence(seq, segments=None, workers=2, sigma=0.5):
    seq = np.asarray(seq, float)
    T = len(seq)
    mask = np.ones(T, bool)
    if segments:
        for kind, a, b, _ in segments:
            if kind == "transition":
                mask[a:b] = False
    jobs = [(s, _to_bl(seq[:, off:off + 21] - seq[:, off:off + 1]), mask) for s, off in (("l", LH), ("r", RH))]
    if workers > 1:
        with ProcessPoolExecutor(2) as ex:
            results = list(ex.map(_fit_side, jobs))
    else:
        results = [_fit_side(j) for j in jobs]
    hands = _load()
    Rout = np.zeros((T, 2, 3, 3))
    Fout = np.zeros((T, 2, 15, 3, 3))
    rms = np.zeros((T, 2))
    for side, X, R, r in results:
        i = "lr".index(side)

        def smooth(a, b, X=X, R=R):
            if b - a < 3:
                return
            X[a:b, 3:] = gaussian_filter1d(X[a:b, 3:], sigma, axis=0, mode="nearest")
            # rotations: smooth relative rotation vectors about the block's running reference
            for _ in range(1):
                Rs = R[a:b].copy()
                for t in range(a, b):
                    lo, hi = max(a, t - 2), min(b, t + 3)
                    w = np.exp(-0.5 * ((np.arange(lo, hi) - t) / sigma) ** 2)
                    v = sum(wi * _log(Rs[t - a].T @ Rs[j - a]) for wi, j in zip(w, range(lo, hi))) / w.sum()
                    R[t] = Rs[t - a] @ _exp(v)

        _smooth_blocks(mask, smooth)
        for t in range(T):
            if mask[t]:
                Rout[t, i] = R[t]
                Fout[t, i] = np.array(hands[side].locals_from(X[t]))
        rms[:, i] = np.nan_to_num(r)
    return dict(R=Rout, F=Fout, rms=rms, valid=mask)


def fit_file(motion_path, segments=None):
    seq = np.load(motion_path)
    base = motion_path[:-4]
    if segments is None and os.path.exists(base + ".segments.json"):
        segments = json.load(open(base + ".segments.json"))
    out = fit_sequence(seq, segments)
    np.savez_compressed(base + ".fit.npz", **out)
    return out


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    o = fit_file(sys.argv[1])
    v = o["valid"]
    print(f"{v.sum()} frames fitted in {time.time() - t0:.1f}s; 2D rms (fraction of hand length): "
          f"L {o['rms'][v, 0].mean():.3f} R {o['rms'][v, 1].mean():.3f}; p95 R {np.percentile(o['rms'][v, 1], 95):.3f}")
