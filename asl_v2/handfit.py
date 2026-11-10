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
# DIP-PIP coupling on: finger-joint error against the 3D reference 17.5 -> 16.3 deg; the
# fingers-together prior gave nothing on top and stays off
W_DIP, W_ABD = float(os.environ.get("ASL_WDIP", 0.05)), float(os.environ.get("ASL_WABD", 0.0))
WZ3D = float(os.environ.get("ASL_WZ3D", 0.8))             # depth weight on frames with a 3D-model reading

EURO_FC, EURO_BETA = 1.5, 1.0                              # input de-jitter (One-Euro)
PASSES, SIG_TRAJ, PULL, SIG_FINAL = 1, 2.0, 10.0, 0.6     # trajectory smoother (see _fit_side)

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
        self.palmar_local = self.R_rest.T @ palmar
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

    def solve(self, T, R_init, x_prev, R_prev, ts=1.0, arm=None, z3d=False):
        """T: (21,3) target landmarks in Blender axes (any scale, any origin)."""
        data_len = sum(np.linalg.norm(T[b + k + 1] - T[b + k]) for b in (5, 9, 13, 17) for k in range(3))
        k = self.finger_len / (data_len + 1e-9)
        Tn = k * (T - T[KNUCK].mean(0)) / self.hand_len
        # a (nearly) flat depth profile means "no depth evidence" for this frame
        wz = WZ * float(np.clip((np.ptp(Tn[:, 1]) - 0.05) / 0.10, 0.0, 1.0))
        if z3d:                       # depth from a 3D hand-mesh model: trust it
            wz = WZ3D

        pri = _prior()

        def res(x):
            R_h = R_init @ _exp(x[:3])
            A = self.fk(R_h, self.locals_from(x))
            An = np.exp(x[25]) * (A - A[KNUCK].mean(0)) / self.hand_len
            d = An[IDX] - Tn[IDX]
            wr = np.exp(x[25]) * (A[0] + self.wrist_off * (R_h @ self.mid_dir_local) - A[KNUCK].mean(0)) / self.hand_len - Tn[0]
            out = [(W2D[:, None] * d[:, [0, 2]]).ravel(), wz * d[:, 1], WWRIST * wr[[0, 2]]]
            if arm is not None:
                # Wrist / forearm range of motion: the hand must be reachable from the forearm.
                #   swing  (wrist flexion / deviation): hand axis within ~80 deg of the forearm
                #   twist  (pronation / supination): palm within ~100 deg of facing the body midline
                v, medial = arm
                hax = R_h @ self.mid_dir_local
                swing = np.arccos(np.clip(hax @ v, -1, 1))
                pal = R_h @ self.palmar_local
                pp = pal - v * (pal @ v)
                npp = np.linalg.norm(pp)
                twist = np.arccos(np.clip(pp @ medial / (npp + 1e-9), -1, 1)) if npp > 0.2 else 0.0
                out.append(np.array([W_WRIST * max(0.0, swing - 1.4), W_WRIST * max(0.0, twist - 1.75),
                                     W_COMFORT * twist, 0.6 * W_COMFORT * swing]))   # prefer a relaxed wrist
            if W_DIP or W_ABD:
                # anatomy: the last finger joint bends with the middle one (DIP ~ 2/3 PIP), and
                # fingers stay together unless the image says they are spread
                fa = x[3:19].reshape(4, 4)
                out += [W_DIP * (fa[:, 3] - 0.67 * fa[:, 2]), W_ABD * fa[:, 1]]
            if pri is None:
                out.append(WREG * x[3:25])
            else:
                # learned handshape prior: stay near the subspace of finger poses signers use
                dv = x[3:25] - pri["mean"]
                z = dv @ pri["comps"].T
                out += [WPRI_OFF * (dv - z @ pri["comps"]), WPRI_IN * z / np.sqrt(pri["lam"])]
            if x_prev is not None:
                out += [ts * WT_ANG * (x[3:25] - x_prev[3:25]), ts * WT_ROT * _log(R_prev.T @ R_h),
                        0.3 * (x[25:] - x_prev[25:])]
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


W_WRIST, W_COMFORT = 0.6, 0.035


def solve_arm(S, E_obs, T_obs, a, b):
    """Two-bone arm from 2D shoulder/elbow/wrist (canonical units) with depth recovered from
    the bone lengths (same construction as the retargeter). Returns elbow, wrist (3D)."""
    S = np.array([S[0], S[1], 0.0])
    dxy1 = np.linalg.norm(E_obs[:2] - S[:2])
    E0 = np.array([E_obs[0], E_obs[1], -0.5 * np.sqrt(max(a * a - dxy1 * dxy1, 0.0))])
    dxy2 = np.linalg.norm(T_obs[:2] - E_obs[:2])
    z_t = E0[2] - np.sqrt(max(b * b - dxy2 * dxy2, 0.0))
    x, y = T_obs[0], T_obs[1]
    if abs(x) < 0.62 and -0.25 < y < 1.75:
        z_t = min(z_t, -0.40)
    elif abs(x) < 0.55 and y <= -0.25:
        z_t = min(z_t, -0.50)
    Tt = np.array([x, y, z_t])
    u = Tt - S
    D = np.clip(np.linalg.norm(u), abs(a - b) + 1e-3, a + b - 1e-3)
    u /= np.linalg.norm(u)
    Tt = S + u * D
    cos_a = np.clip((a * a + D * D - b * b) / (2 * a * D), -1, 1)
    pole = E0 - S
    vv = pole - u * np.dot(pole, u)
    if np.linalg.norm(vv) < 1e-6:
        vv = np.array([0.0, 1.0, 0.0]) - u * u[1]
    vv /= np.linalg.norm(vv)
    return S + a * (cos_a * u + np.sqrt(1 - cos_a ** 2) * vv), Tt


def arm_frames(seq, side, cal):
    """Per frame: forearm direction and the direction the palm faces at neutral forearm
    rotation (towards the body midline), both in Blender axes."""
    bh = cal["bone_heads_canon"]
    S = np.array(bh[f"upperarm_{side}"])
    a = np.linalg.norm(np.array(bh[f"lowerarm_{side}"]) - S)
    b = np.linalg.norm(np.array(bh[f"hand_{side}"]) - np.array(bh[f"lowerarm_{side}"]))
    el, off = (13, LH) if side == "l" else (14, RH)
    out = []
    for f in seq:
        E, Tt = solve_arm(S, f[el], f[off], a, b)
        u = _to_bl(E - np.array([S[0], S[1], 0.0]))
        v = _to_bl(Tt - E)
        u /= np.linalg.norm(u) + 1e-9
        v /= np.linalg.norm(v) + 1e-9
        n = np.cross(u, v)
        if np.linalg.norm(n) < 0.15:                      # straight arm: elbow axis undefined
            n = np.cross(v, np.array([0.0, -1.0, 0.0]))    # use the horizontal axis instead
            n = n if (n[0] < 0) else -n
        n /= np.linalg.norm(n) + 1e-9
        medial = n if side == "l" else -n
        medial = medial - v * (medial @ v)
        out.append((v, medial / (np.linalg.norm(medial) + 1e-9)))
    return out


_RIG = None
_PRI = False
WPRI_OFF, WPRI_IN = 0.06, 0.004


def _prior():
    """Hand-pose prior trained by asl_v2.train_prior (off by default: an A/B test on 5 phrases showed no gain; enable with ASL_PRIOR=1)."""
    global _PRI
    if _PRI is False:
        path = os.path.join(HERE, "models", "avatar_hand_prior.npz")
        _PRI = dict(np.load(path)) if os.path.exists(path) and os.environ.get("ASL_PRIOR", "0") == "1" else None
    return _PRI


def _load():
    global _RIG
    if _RIG is None:
        rig = json.load(open(os.path.join(HERE, "models", "rig_hand.json")))
        cal = json.load(open(os.path.join(HERE, "models", "avatar_calib.json")))
        _RIG = {s: Hand(s, rig, cal) for s in "lr"}
    return _RIG


def _to_bl(p):
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def one_euro(x, fps=30.0, min_cutoff=1.5, beta=1.0, d_cutoff=1.0):
    """One-Euro filter (Casiez et al. 2012) run forwards and backwards and averaged (zero
    lag): strong smoothing while the signal is nearly still (tracker jitter), light smoothing
    when it really moves (a handshape change). x: (T, D)."""
    def run(x):
        def alpha(fc):
            tau = 1.0 / (2 * np.pi * fc)
            return 1.0 / (1.0 + tau * fps)
        y = np.zeros_like(x)
        y[0] = x[0]
        dx = np.zeros(x.shape[1])
        for t in range(1, len(x)):
            d = (x[t] - y[t - 1]) * fps
            dx = dx + alpha(d_cutoff) * (d - dx)
            fc = min_cutoff + beta * np.linalg.norm(dx)
            y[t] = y[t - 1] + alpha(fc) * (x[t] - y[t - 1])
        return y
    return 0.5 * (run(x) + run(x[::-1])[::-1])


def _fit_side(args):
    side, targets, mask = args[:3]
    arms = args[3] if len(args) > 3 else None
    z3d = (args[4] > 0.5) if len(args) > 4 else np.zeros(len(targets), bool)
    hand = _load()[side]
    T = len(targets)
    # de-jitter the tracked hand shape (hand-length units) block by block before fitting
    ch = [0, 9, 10, 11, 12]
    l0 = np.median([sum(np.linalg.norm(tg[b] - tg[a]) for a, b in zip(ch[:-1], ch[1:])) for tg in targets]) + 1e-9
    targets = targets.copy()

    def dejitter(a, b):
        if b - a >= 4:
            targets[a:b] = one_euro((targets[a:b] / l0).reshape(b - a, -1), min_cutoff=EURO_FC, beta=EURO_BETA
                                    ).reshape(b - a, 21, 3) * l0
    _smooth_blocks(mask, dejitter)
    X = np.full((T, 26), np.nan)
    R = np.full((T, 3, 3), np.nan)
    rms = np.full(T, np.nan)
    # Reliability from temporal stability: where the tracked hand shape keeps jumping from
    # frame to frame for a sustained stretch (hands pointing at the camera, heavy occlusion)
    # the landmarks are noise; the pose is then held close to the previous frame (temporal
    # weight up to 12x). Real fast signing (a letter change) is unstable for 2-3 frames only,
    # so the 60th percentile over a 9-frame window ignores it.
    chain = [0, 9, 10, 11, 12]
    l3 = np.median([sum(np.linalg.norm(tg[b] - tg[a]) for a, b in zip(chain[:-1], chain[1:])) for tg in targets])
    step = np.zeros(T)
    step[1:] = np.linalg.norm((targets[1:] - targets[:-1])[:, :, [0, 2]], axis=2).max(1) / (l3 + 1e-9)
    inst = np.array([np.percentile(step[max(0, t - 4):t + 5], 60) for t in range(T)])
    tscale = 1.0 + 11.0 * np.clip((inst - 0.16) / 0.10, 0.0, 1.0)
    used_mirror = np.zeros(T, bool)
    x_prev = R_prev = None
    for t in range(T):
        if not mask[t]:
            x_prev = R_prev = None                       # new block after a transition
            continue
        tg = targets[t]
        arm = arms[t] if arms is not None else None
        cands = []
        mirror = tg * np.array([1.0, -1.0, 1.0])
        if z3d[t]:
            # depth is known (3D hand model): no mirror ambiguity; start from the previous pose
            # and from the data's own orientation, keep the better fit
            if R_prev is not None:
                cands.append(hand.solve(tg, R_prev, x_prev, R_prev, tscale[t], arm, True))
            cands.append(hand.solve(tg, hand.init_rot(tg), x_prev, R_prev, tscale[t], arm, True))
            ci = int(np.argmin([c[2] for c in cands]))
            x, Rh, _, r = cands[ci]
        else:
            if R_prev is not None:
                # The depth SIGN of a tracked hand is unreliable frame to frame (a hand and its
                # depth mirror look the same in the image). Fit both readings from the previous
                # pose; the temporal term then keeps the one that continues the motion.
                cands.append(hand.solve(tg, R_prev, x_prev, R_prev, tscale[t], arm))
                cands.append(hand.solve(mirror, R_prev, x_prev, R_prev, tscale[t], arm))
            if R_prev is None or (min(c[3] for c in cands) > 0.07 and tscale[t] < 2):   # (re)initialise from the data
                for init in (hand.init_rot(tg), hand.init_rot(mirror)):
                    cands.append(hand.solve(tg, init, x_prev, R_prev, tscale[t], arm))
            ci = int(np.argmin([c[2] for c in cands]))
            x, Rh, _, r = cands[ci]
            used_mirror[t] = (R_prev is not None and ci == 1) or (len(cands) in (2, 4) and ci == len(cands) - 1
                                                                  and R_prev is None)
        x = np.r_[0, 0, 0, x[3:]]
        X[t], R[t], rms[t] = x, Rh, r
        x_prev, R_prev = x, Rh

    # ---- pass 2: smoother. Low-pass the whole trajectory (zero phase), then re-solve every
    # frame against the image with a pull towards that smooth trajectory (past AND future),
    # so frame-to-frame solver noise is removed while real handshape changes, which the
    # image insists on, are kept.
    def lowpass(a, b, sig):
        if b - a < 3:
            return
        X[a:b, 3:] = gaussian_filter1d(X[a:b, 3:], sig, axis=0, mode="nearest")
        Rs = R[a:b].copy()
        half = int(3 * sig + 0.5)
        for t in range(a, b):
            lo, hi = max(a, t - half), min(b, t + half + 1)
            w = np.exp(-0.5 * ((np.arange(lo, hi) - t) / sig) ** 2)
            v = sum(wi * _log(Rs[t - a].T @ Rs[j - a]) for wi, j in zip(w, range(lo, hi))) / w.sum()
            R[t] = Rs[t - a] @ _exp(v)

    # How fast the handshape really changes in the image (low-passed 2D shape speed): the pull
    # towards the smooth trajectory is strong while a handshape is held and relaxed during a
    # real handshape change (fingerspelling), so letters stay crisp and holds stay still.
    act = np.zeros(T)      # (speed-adaptive relaxation disabled: 2D speed cannot tell jitter from change)
    for _ in range(PASSES):
        _smooth_blocks(mask, lambda a, b: lowpass(a, b, SIG_TRAJ))
        for t in np.flatnonzero(mask):
            tg = targets[t] * (np.array([1.0, -1.0, 1.0]) if used_mirror[t] else 1.0)
            x, Rh, _, r = hand.solve(tg, R[t], X[t], R[t], PULL * (1 - 0.85 * act[t]) * tscale[t],
                                     arms[t] if arms is not None else None, bool(z3d[t]))
            X[t], R[t], rms[t] = np.r_[0, 0, 0, x[3:]], Rh, r
    Xp, Rp = X.copy(), R.copy()
    _smooth_blocks(mask, lambda a, b: lowpass(a, b, SIG_FINAL))
    for t in np.flatnonzero(mask):                      # keep fast real changes un-blurred
        X[t] = (1 - act[t]) * X[t] + act[t] * Xp[t]
        R[t] = Rp[t] @ _exp((1 - act[t]) * _log(Rp[t].T @ R[t]))
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
    cal = json.load(open(os.path.join(HERE, "models", "avatar_calib.json")))
    jobs = [(s, _to_bl(seq[:, off:off + 21] - seq[:, off:off + 1]), mask, arm_frames(seq, s, cal), seq[:, slot, 0])
            for s, off, slot in (("l", LH, 17), ("r", RH, 18))]
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
