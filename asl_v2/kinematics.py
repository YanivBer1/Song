"""
Anatomical hand kinematics for the ASL avatar pipeline.

MediaPipe landmarks are 21 independent points per frame: bone lengths wobble,
joints bend sideways or backwards, and occluded frames are simply missing.
This module replaces them with a *skeleton*: per-signer constant bone lengths
and 21 joint angles that stay inside published ranges of motion, fitted to the
landmarks, smoothed / gap-filled in ANGLE space, and finally de-penetrated.

Conventions
-----------
Landmarks  0 wrist; thumb 1 CMC, 2 MCP, 3 IP, 4 tip; index 5-8; middle 9-12;
           ring 13-16; pinky 17-20.
Camera     x right, y down, z away from the camera (same as track.world_*).
Hand-local frame (canonical RIGHT hand, wrist at the origin, metres):
           +y distal (wrist -> middle MCP), +x radial (towards the thumb side),
           +z palmar (out of the palm).  A LEFT hand is the mirror image:
           local x is negated (x = ulnar), y distal, z palmar.
orient     (3,3) proper rotation, hand-local -> camera axes:
           joints_cam = joints_local @ orient.T
angles     (21,) radians, see ANGLE_NAMES.  Layout: 5 digits x 4 + palm arch
             thumb : CMC flexion, CMC abduction, MCP flexion, IP flexion
             finger: MCP flexion, MCP abduction, PIP flexion, DIP flexion
             [20]  : palm arch (flexion of the 5th metacarpal; 4th gets half)
           flexion > 0 curls towards the palm; finger "abduction" > 0 is a
           radial deviation (towards the thumb) for every finger; thumb CMC
           abduction > 0 moves the thumb away from the palm/index (palmar-
           radial abduction).
bones      dict(lengths (5,3) phalanx lengths per digit, palm (5,3) local rest
           positions of landmarks [1,5,9,13,17], side 'L'/'R', hand_size).
"""

import os

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.optimize import least_squares

D = 21                                   # number of joint angles
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
ANGLE_NAMES = (
    ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp_flex", "thumb_ip_flex"]
    + [f"{f}_{j}" for f in DIGITS[1:] for j in ("mcp_flex", "mcp_abd", "pip_flex", "dip_flex")]
    + ["palm_arch"])
CHAIN = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
PALM_IDX = CHAIN[:, 0]                   # 1, 5, 9, 13, 17
BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)] + [
    (int(c[k]), int(c[k + 1])) for c in CHAIN for k in range(3)]
_MIRROR = np.array([-1.0, 1.0, 1.0])
_d = np.radians

# --------------------------------------------------------------------------
# Joint limits (degrees).  Sources: AAOS "Joint Motion: Method of Measuring
# and Recording"; Kapandji, Physiology of the Joints vol. 1; Cooney et al.
# 1981 (thumb CMC); Lin, Wu & Huang 2000 (hand-model constraints).
# --------------------------------------------------------------------------
LIMITS_DEG = {
    # Thumb CMC (trapeziometacarpal saddle joint).  Cooney: flexion-extension
    # arc ~53 deg, abduction-adduction arc ~42 deg.  The angles here are
    # measured from the calibrated neutral frame THUMB_REST of the *landmark*
    # chain 1->2 (MediaPipe's "CMC" point sits proximal of the true joint and
    # the apparent arc is a little larger), so the arcs are 75 / 75 deg.
    "thumb_cmc_flex": (-35.0, 40.0),
    "thumb_cmc_abd": (-40.0, 35.0),
    # Thumb MCP: flexion 50-60 deg typical (very variable), hyperextension
    # 10-15 deg common.
    "thumb_mcp_flex": (-15.0, 65.0),
    # Thumb IP: flexion ~80 deg, hyperextension 20-25 deg common ("hitch-hiker").
    "thumb_ip_flex": (-25.0, 85.0),
    # Finger MCP flexion: 90 deg flexion (AAOS), 20-30 deg active hyperextension.
    "mcp_flex": (-30.0, 90.0),
    # PIP: pure hinge, 0..100-110 deg, no hyperextension.
    "pip_flex": (0.0, 110.0),
    # DIP: pure hinge, 0..80-90 deg, ~10 deg hyperextension allowed.
    "dip_flex": (-10.0, 90.0),
    # MCP abduction, expressed as radial(+)/ulnar(-) deviation from the rest
    # direction FINGER_SPLAY.  About +/-20 deg overall; the index abducts
    # radially and the pinky ulnarly further, the middle finger least.
    "index_mcp_abd": (-15.0, 25.0),
    "middle_mcp_abd": (-15.0, 15.0),
    "ring_mcp_abd": (-20.0, 15.0),
    "pinky_mcp_abd": (-30.0, 15.0),
    # Palm arch: 5th carpometacarpal flexion ~20-25 deg, 4th about half
    # (2nd/3rd CMC are rigid).  Measured from the signer's median palm shape,
    # hence the small negative allowance.
    "palm_arch": (-10.0, 20.0),
}


def _limits():
    lo, hi = np.zeros(D), np.zeros(D)
    for i, n in enumerate(ANGLE_NAMES):
        key = n if n in LIMITS_DEG else n.split("_", 1)[1]
        lo[i], hi[i] = LIMITS_DEG[key]
    return _d(lo), _d(hi)


LIMITS_LO, LIMITS_HI = _limits()
DIP_PIP_RATIO = 2.0 / 3.0                # soft tendon coupling DIP ~ 2/3 PIP (Rijpkema & Girard 1991)
ARCH_FACTOR = np.array([0.0, 0.0, 0.0, 0.5, 1.0])   # share of the palm-arch angle per digit
ARCH_PIVOT = 0.25                        # metacarpal base = 25% of the way wrist -> MCP

# Rest direction of each finger in the palm plane (degrees of radial splay from
# +y).  Calibrated on the cached clips (median of extended fingers).
FINGER_SPLAY_DEG = {"index": -3.0, "middle": -5.0, "ring": -5.0, "pinky": -15.0}


def _frame_from(y, z_hint):
    y = np.asarray(y, float) / np.linalg.norm(y)
    z = np.asarray(z_hint, float)
    z = z - y * (z @ y)
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)       # columns x, y, z


# Thumb neutral frame in palm coordinates (columns: thumb x, y, z).
# y = neutral direction of the thumb metacarpal (radial + distal + palmar);
# z = the direction thumb flexion moves the tip (ulnar + palmar, i.e. across
# the palm).  Both flexion axis (x) and abduction axis (z) are therefore
# oblique to the palm plane, as in the real saddle joint.
THUMB_REST = _frame_from((0.255, 0.795, 0.494), (-0.83, 0.42, -0.25))


def _digit_frames():
    S = np.zeros((5, 3, 3))
    S[0] = THUMB_REST
    for k, f in enumerate(DIGITS[1:], 1):
        a = _d(FINGER_SPLAY_DEG[f])
        S[k] = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return S


DIGIT_FRAMES = _digit_frames()

# Generic adult right hand (metres; median proportions of the cached signers),
# used when a clip has no metric landmarks.
DEFAULT_BONES = {
    "lengths": 0.095 * np.array([[0.328, 0.367, 0.316], [0.279, 0.231, 0.249], [0.347, 0.284, 0.281],
                                 [0.304, 0.263, 0.263], [0.243, 0.247, 0.198]]),
    "palm": 0.095 * np.array([[0.202, 0.297, 0.124], [0.191, 1.006, 0.046], [0.0, 1.0, 0.0],
                              [-0.179, 0.950, 0.023], [-0.330, 0.796, 0.046]]),
    "side": "R", "hand_size": 0.095,
}


# --------------------------------------------------------------------------
# Small rotation helpers
# --------------------------------------------------------------------------
def _exp(w):
    """Rodrigues: rotation vectors (...,3) -> matrices (...,3,3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -w[..., 2], w[..., 1]
    K[..., 1, 0], K[..., 1, 2] = w[..., 2], -w[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -w[..., 1], w[..., 0]
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6, np.sin(ths) / ths)
    b = np.where(small, 0.5 - th ** 2 / 24, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + a * K + b * (K @ K)


def _mat_to_quat(R):
    """(...,3,3) -> (...,4) [w,x,y,z]; robust branch-free (largest component)."""
    R = np.asarray(R, float)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    q = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    q = np.sqrt(np.maximum(q, 0)) / 2
    k = np.argmax(q, -1)
    out = np.zeros_like(q)
    it = np.ndindex(k.shape)
    for i in it:
        r, kk, w = R[i], k[i], q[i][k[i]]
        if kk == 0:
            out[i] = [w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)]
        elif kk == 1:
            out[i] = [(r[2, 1] - r[1, 2]) / (4 * w), w, (r[0, 1] + r[1, 0]) / (4 * w), (r[0, 2] + r[2, 0]) / (4 * w)]
        elif kk == 2:
            out[i] = [(r[0, 2] - r[2, 0]) / (4 * w), (r[0, 1] + r[1, 0]) / (4 * w), w, (r[1, 2] + r[2, 1]) / (4 * w)]
        else:
            out[i] = [(r[1, 0] - r[0, 1]) / (4 * w), (r[0, 2] + r[2, 0]) / (4 * w), (r[1, 2] + r[2, 1]) / (4 * w), w]
    return out


def _quat_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _kabsch(A, B, w=None):
    """Rotation R (and translation) minimising |A @ R.T + t - B| for (...,n,3)."""
    w = np.ones(A.shape[-2]) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ca = (A * w[:, None]).sum(-2, keepdims=True)
    cb = (B * w[:, None]).sum(-2, keepdims=True)
    H = np.swapaxes((A - ca) * w[:, None], -1, -2) @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.swapaxes(-1, -2) @ U.swapaxes(-1, -2)))
    Dm = np.zeros(H.shape)
    Dm[..., 0, 0] = Dm[..., 1, 1] = 1
    Dm[..., 2, 2] = d
    R = Vt.swapaxes(-1, -2) @ Dm @ U.swapaxes(-1, -2)
    t = cb[..., 0, :] - np.einsum("...ij,...j->...i", R, ca[..., 0, :])
    return R, t


def _rotx(v, c, s):
    """Rotate vectors (...,3) about the local x axis by angles given as cos/sin."""
    return np.stack([v[..., 0] + 0 * c, v[..., 1] * c - v[..., 2] * s, v[..., 1] * s + v[..., 2] * c], -1)


# --------------------------------------------------------------------------
# Forward kinematics
# --------------------------------------------------------------------------
def _fk_local(ang, bones):
    """Canonical right hand: angles (N,21) -> hand-local joints (N,21,3)."""
    ang = np.atleast_2d(ang)
    N = len(ang)
    a = ang[:, :20].reshape(N, 5, 4)
    th = ang[:, 20:21] * ARCH_FACTOR                            # (N,5) metacarpal flexion
    c, s = np.cos(th), np.sin(th)
    palm, L = bones["palm"], bones["lengths"]
    pivot = palm * ARCH_PIVOT
    pivot[0] = palm[0]
    base = pivot + _rotx((palm - pivot)[None], c, s)            # (N,5,3)
    S0 = _rotx(DIGIT_FRAMES[None, :, :, 0], c, s)
    S1 = _rotx(DIGIT_FRAMES[None, :, :, 1], c, s)
    S2 = _rotx(DIGIT_FRAMES[None, :, :, 2], c, s)
    sa, ca = np.sin(a[..., 1:2]), np.cos(a[..., 1:2])
    by = S0 * sa + S1 * ca                                      # abducted pointing direction
    c1 = a[..., 0:1]
    c2 = c1 + a[..., 2:3]
    c3 = c2 + a[..., 3:4]
    J = np.zeros((N, 21, 3))
    p = base
    J[:, CHAIN[:, 0]] = p
    for k, ck in enumerate((c1, c2, c3)):
        p = p + L[None, :, k:k + 1] * (by * np.cos(ck) + S2 * np.sin(ck))
        J[:, CHAIN[:, k + 1]] = p
    return J


def forward(angles, orient=None, bones=None):
    """Forward kinematics.

    angles (21,) or (T,21) radians; orient (3,3) or (T,3,3) hand-local -> camera
    (None = identity, i.e. hand-local output); bones from estimate_bones().
    Returns joints (21,3) or (T,21,3) in metres, wrist at the origin, rotated
    by `orient`.  Handedness comes from bones['side'].
    """
    bones = DEFAULT_BONES if bones is None else bones
    ang = np.asarray(angles, float)
    J = _fk_local(ang.reshape(-1, D), bones)
    if bones.get("side", "R") == "L":
        J = J * _MIRROR
    if orient is not None:
        J = np.einsum("...ij,...kj->...ki", np.asarray(orient, float).reshape(-1, 3, 3), J)
    return J[0] if ang.ndim == 1 else J


# --------------------------------------------------------------------------
# Bones + analytic inverse kinematics
# --------------------------------------------------------------------------
def _rough_frame(P):
    """Palm frame from landmarks 0, 5, 9, 17 of a right hand: (T,3,3) columns x,y,z."""
    y = P[:, 9] - P[:, 0]
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    x = P[:, 5] - P[:, 17]
    x -= y * (x * y).sum(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return np.stack([x, y, np.cross(x, y)], axis=2)


def detect_side(world):
    """Guess handedness from finger curl (fingers and thumb sit on the palmar side)."""
    P = np.asarray(world, float)
    P = P[np.isfinite(P).all((1, 2))]
    if len(P) == 0:
        return "R"
    F = _rough_frame(P.copy())
    tips = (P[:, [4, 8, 12, 16, 20]] - P[:, [2, 5, 9, 13, 17]])
    curl = np.einsum("tki,ti->tk", tips, F[:, :, 2])
    return "R" if np.median(curl.mean(1)) >= 0 else "L"


def estimate_bones(world, src=None, side="R", min_frames=5):
    """Per-signer constant skeleton: medians over confident (src>=2) metric frames."""
    W = np.asarray(world, float)
    ok = np.isfinite(W).all((1, 2))
    if src is not None:
        ok &= np.asarray(src) >= 2
    if ok.sum() < min_frames:
        b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULT_BONES.items()}
        b["side"] = side
        return b
    P = W[ok] - W[ok][:, :1]
    if side == "L":
        P = P * _MIRROR
    lengths = np.median(np.linalg.norm(P[:, CHAIN[:, 1:]] - P[:, CHAIN[:, :-1]], axis=-1), axis=0)
    F = _rough_frame(P.copy())
    local = np.einsum("tki,tij->tkj", P, F)
    palm = np.median(local[:, PALM_IDX], axis=0)
    for _ in range(2):                                           # generalised Procrustes refinement
        tmpl = np.vstack([np.zeros(3), palm[1:]])
        R, t = _kabsch(tmpl[None], P[:, [0, 5, 9, 13, 17]], w=[1, 1, 1, 0.5, 0.3])
        local = np.einsum("tki,tij->tkj", P - t[:, None], R)
        palm = np.median(local[:, PALM_IDX], axis=0) - np.median(local[:, 0], axis=0)
    # canonicalise: middle MCP on +y, index/pinky MCPs in the z = 0 plane
    y = palm[2] / np.linalg.norm(palm[2])
    x = palm[1] - palm[4]
    x -= y * (x @ y)
    x /= np.linalg.norm(x)
    C = np.stack([x, y, np.cross(x, y)], axis=1)
    palm = palm @ C
    return {"lengths": lengths, "palm": palm, "side": side, "hand_size": float(np.linalg.norm(palm[2]))}


def _ik_right(P, bones):
    """Analytic IK of right-hand landmarks (T,21,3) -> raw angles (T,21), R (T,3,3), t (T,3)."""
    palm = bones["palm"]
    tmpl = np.vstack([np.zeros(3), palm[1:]])
    R, t = _kabsch(tmpl[None], P[:, [0, 5, 9, 13, 17]], w=[1, 1, 1, 0.5, 0.3])
    loc = np.einsum("tki,tij->tkj", P - t[:, None], R)
    T = len(P)
    ang = np.zeros((T, D))
    # palm arch from the palmar drop of the pinky / ring knuckles
    lever = (1 - ARCH_PIVOT) * np.abs(palm[[3, 4], 1])
    dz = loc[:, [13, 17], 2] - palm[[3, 4], 2]
    arch = (dz[:, 0] * 0.5 / lever[0] + dz[:, 1] / lever[1]) / 1.25
    ang[:, 20] = arch
    th = arch[:, None] * ARCH_FACTOR
    c, s = np.cos(th), np.sin(th)
    S0 = _rotx(DIGIT_FRAMES[None, :, :, 0], c, s)
    S1 = _rotx(DIGIT_FRAMES[None, :, :, 1], c, s)
    S2 = _rotx(DIGIT_FRAMES[None, :, :, 2], c, s)
    seg = loc[:, CHAIN[:, 1:]] - loc[:, CHAIN[:, :-1]]            # (T,5,3,3)
    v1, v2, v3 = seg[:, :, 0], seg[:, :, 1], seg[:, :, 2]
    u = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-12)
    ux, uy, uz = (u * S0).sum(-1), (u * S1).sum(-1), (u * S2).sum(-1)
    flip = (uy < 0) & (np.abs(ux) < np.abs(uy))                  # bent past 90 deg
    abd = np.where(flip, np.arctan2(-ux, -uy), np.arctan2(ux, uy))
    abd = np.where(np.hypot(ux, uy) < 0.05, 0.0, abd)
    fl = np.arcsin(np.clip(uz, -1, 1))
    fl = np.where(flip, np.sign(uz + 1e-12) * np.pi - fl, fl)
    axis = S0 * np.cos(abd)[..., None] - S1 * np.sin(abd)[..., None]   # flexion axis after abduction
    def hinge(a, b):
        return np.arctan2((np.cross(a, b) * axis).sum(-1), (a * b).sum(-1))
    A = ang[:, :20].reshape(T, 5, 4)
    A[..., 0], A[..., 1], A[..., 2], A[..., 3] = fl, abd, hinge(v1, v2), hinge(v2, v3)
    return ang, R, t


def inverse_kinematics(joints, bones=None, side=None):
    """Raw (unclamped, unfiltered) joint angles of landmark sets.

    joints (T,21,3) or (21,3) in any metric/rigid frame; returns angles
    (T,21) and orient (T,3,3).  Rows containing NaN give NaN.
    """
    P = np.asarray(joints, float)
    single = P.ndim == 2
    P = P.reshape(-1, 21, 3)
    side = side or (bones or {}).get("side", "R")
    bones = bones or estimate_bones(P, None, side)
    ok = np.isfinite(P).all((1, 2))
    ang = np.full((len(P), D), np.nan)
    R = np.full((len(P), 3, 3), np.nan)
    if ok.any():
        Q = P[ok] * (_MIRROR if side == "L" else 1.0)
        ang[ok], Rr, _ = _ik_right(Q, bones)
        R[ok] = Rr * (np.outer(_MIRROR, _MIRROR) if side == "L" else 1.0)
    return (ang[0], R[0]) if single else (ang, R)


def outside_limits(angles, tol_deg=0.5):
    """Boolean (T,21): which joint angles violate the range of motion."""
    a = np.asarray(angles, float)
    tol = _d(tol_deg)
    return (a < LIMITS_LO - tol) | (a > LIMITS_HI + tol)


# --------------------------------------------------------------------------
# Angle prior (PCA + diagonal Gaussian)
# --------------------------------------------------------------------------
class AnglePrior:
    """Gaussian prior over joint angles in PCA space.

    residuals(a) are whitened PCA coordinates plus the whitened off-subspace
    remainder, so 0.5*|residuals|^2 is the negative log-likelihood (up to a
    constant) and they can be appended to a least-squares problem directly.
    """

    def __init__(self, mean, components, std, resid_std):
        self.mean = np.asarray(mean, float)
        self.components = np.asarray(components, float)          # (k, D) orthonormal rows
        self.std = np.asarray(std, float)
        self.resid_std = float(resid_std)
        k = len(self.components)
        null = np.eye(D) - self.components.T @ self.components
        self.matrix = np.vstack([self.components / self.std[:, None], null / self.resid_std])
        self.k = k

    def residuals(self, angles):
        return (np.asarray(angles, float) - self.mean) @ self.matrix.T

    def jac(self, angles=None):
        return self.matrix

    def neg_log_lik(self, angles):
        r = self.residuals(angles)
        return 0.5 * (r * r).sum(-1)

    def grad(self, angles):
        return self.residuals(angles) @ self.matrix

    def to_dict(self):
        return dict(mean=self.mean, components=self.components, std=self.std, resid_std=self.resid_std)

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["components"], d["std"], float(d["resid_std"]))


def train_angle_prior(angle_arrays, n_components=10, min_std_deg=3.0):
    """PCA + diagonal Gaussian prior from a list of (T,21) angle arrays (NaN rows ignored)."""
    X = np.concatenate([np.asarray(a, float).reshape(-1, D) for a in angle_arrays])
    X = X[np.isfinite(X).all(1)]
    mean = X.mean(0)
    _, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = int(min(n_components, len(sv), D - 1))
    var = sv ** 2 / max(len(X) - 1, 1)
    floor = _d(min_std_deg)
    std = np.maximum(np.sqrt(var[:k]), floor)
    resid = max(np.sqrt(var[k:].mean()) if len(var) > k else 0.0, floor)
    return AnglePrior(mean, Vt[:k], std, resid)


# --------------------------------------------------------------------------
# Per-frame fit
# --------------------------------------------------------------------------
SRC_WEIGHT = np.array([0.0, 0.3, 0.8, 1.0])   # none / Holistic-only / crop / crop+Holistic agree
OCCL_WEIGHT = 0.5                             # hand-over-hand frames are trusted half as much
WORLD_W = np.array([0.3, 0.3, 0.5])           # metric landmarks: weight of x, y, depth (image xy weighs 1)
LS_W = 0.2                                    # image-scale freedom (log) in metric frames
SMOOTH_SIGMA = 1.5                            # frames; Gaussian taper of the 7-frame Savitzky-Golay window
TUKEY_C = 5.0                                 # glitch rejection cut-off (robust sigmas)
RESCUE_RMS = 0.085                            # 2D misfit (hand sizes) that triggers an image-only refit
_NP = 7 + D                                   # rotvec(3), offset(3), angles(21), log image scale(1)
_A = slice(6, 6 + D)
_XS = np.r_[0.3 * np.ones(3), 0.1 * np.ones(3), 0.5 * np.ones(D), 0.1]


def _bounds(ls_lo=-1e-6, ls_hi=1e-6):
    return (np.r_[-np.pi * np.ones(3), -4 * np.ones(3), LIMITS_LO, ls_lo],
            np.r_[np.pi * np.ones(3), 4 * np.ones(3), LIMITS_HI, ls_hi])


class _Frame:
    """Residual model of one frame (canonical right hand, lengths in hand sizes).

    Model points P = R0 exp(w) FK(angles)/H + c.  Residual blocks:
      metric landmarks (3D), image landmarks (2D, optional log-scale), image
      relative depth, DIP~2/3 PIP coupling, no-abduction-in-flexion, temporal
      pull to the previous frame, orientation pull, optional angle prior."""

    def __init__(self, bones, R0, Wt, ww, It, wi, Iz, wz, prev, wt, wo, wc, prior, wp,
                 ls_bounds=(-1e-6, 1e-6), wls=0.0):
        self.bones, self.R0, self.H = bones, R0, bones["hand_size"]
        self.Wt, self.ww, self.It, self.wi, self.Iz, self.wz = Wt, ww, It, wi, Iz, wz
        self.prev, self.wt, self.wo, self.wc, self.prior, self.wp = prev, wt, wo, wc, prior, wp
        self.lo, self.hi = _bounds(*ls_bounds)
        self.wls = wls

    def points(self, X):
        R = self.R0 @ _exp(X[:, :3])
        return np.einsum("nij,nkj->nki", R, _fk_local(X[:, _A], self.bones)) / self.H + X[:, None, 3:6]

    def res(self, X):
        X = np.atleast_2d(X)
        N = len(X)
        ang = X[:, _A]
        P = self.points(X)
        k = np.exp(X[:, -1])[:, None, None]
        out = [((P - self.Wt) * self.ww).reshape(N, -1),
               ((k * P[:, :, :2] - self.It) * self.wi).reshape(N, -1),
               (k[:, 0] * (P[:, :, 2] - P[:, :1, 2]) - self.Iz) * self.wz]
        a = ang[:, :20].reshape(N, 5, 4)
        out.append(self.wc * (a[:, 1:, 3] - DIP_PIP_RATIO * a[:, 1:, 2]))        # DIP ~ 2/3 PIP
        # collateral ligaments tighten in flexion: abduction range shrinks to ~0 at 90 deg MCP flexion
        out.append(0.15 * a[:, 1:, 1] * np.clip(a[:, 1:, 0], 0, None) / (np.pi / 2))
        out.append(self.wt * (ang - self.prev))
        out.append(self.wo * X[:, :3])
        out.append(self.wls * X[:, -1:])
        if self.prior is not None:
            out.append(self.wp * self.prior.residuals(ang))
        return np.concatenate(out, axis=1)

    def fun(self, x):
        return self.res(x)[0]

    def jac(self, x, h=1e-6):
        r = self.res(np.vstack([x, x + h * np.eye(_NP)]))
        return ((r[1:] - r[0]) / h).T

    def clip(self, x):
        return np.clip(x, self.lo + 1e-7, self.hi - 1e-7)

    def cost(self, x):
        r = self.fun(x) / 0.05
        return (np.sqrt(1 + r * r) - 1).sum()                    # same soft-L1 as the solver

    def solve(self, x0, nfev=30):
        s = least_squares(self.fun, self.clip(x0), jac=self.jac, bounds=(self.lo, self.hi), x_scale=_XS,
                          loss="soft_l1", f_scale=0.05, max_nfev=nfev, xtol=1e-5, ftol=1e-5, gtol=1e-8)
        return s.x, self.cost(s.x)


def _robust_series(v, ok):
    """Median-of-5 then log-linear interpolation of a positive per-frame scalar."""
    T = len(v)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    vv = np.array([np.median(v[idx[max(0, i - 2):i + 3]]) for i in range(len(idx))])
    return np.exp(np.interp(np.arange(T), idx, np.log(vv)))


def _local_poly(Y, w, half=3, sigma=None, ridge=0.02):
    """Zero-phase weighted Savitzky-Golay (window 2*half+1, order 2, Gaussian
    taper) along axis 0.  Frames with tiny weight are re-estimated from their
    neighbours; the ridge on slope/curvature keeps sparse windows tame."""
    T = len(Y)
    sigma = SMOOTH_SIGMA if sigma is None else sigma
    Y2 = Y.reshape(T, -1)
    offs = np.arange(-half, half + 1)
    idx = np.arange(T)[:, None] + offs
    inb = (idx >= 0) & (idx < T)
    idx = np.clip(idx, 0, T - 1)
    Wk = np.exp(-0.5 * (offs / sigma) ** 2) * w[idx] * inb
    X = np.vander(offs.astype(float), 3, increasing=True)
    A = np.einsum("tk,ki,kj->tij", Wk, X, X)
    A += ridge * Wk.sum(1)[:, None, None] * np.diag([0.0, 1.0, 1.0]) + 1e-12 * np.eye(3)
    B = np.einsum("tk,ki,tkd->tid", Wk, X, Y2[idx])
    return np.linalg.solve(A, B)[:, 0].reshape(Y.shape)


def _interp_fill(Y, ok):
    idx = np.flatnonzero(ok)
    out = Y.reshape(len(Y), -1).copy()
    for j in range(out.shape[1]):
        out[:, j] = np.interp(np.arange(len(Y)), idx, out[idx, j])
    return out.reshape(Y.shape)


def _runs(mask):
    """[(start, stop)] of True runs."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2], d[1::2]))


def stabilize_handshapes(angles, weight, speed_deg=2.5, min_len=4, strength=0.8):
    """SGNify-style hand-pose invariance.

    A sign holds a handshape for most of its duration, so frames in a static
    segment (finger-angle speed below `speed_deg` per frame for >= min_len
    frames) are pulled towards the confidence-weighted mean pose of the
    segment: weakly for confident frames (0.3*strength), strongly for
    low-confidence ones (strength).  Low-confidence frames *between* two
    static segments are pulled towards the linear (angle-space) blend of the
    two reference poses.  Returns (angles, static mask)."""
    a = np.asarray(angles, float).copy()
    T = len(a)
    w = np.asarray(weight, float)
    if T < min_len + 2:
        return a, np.zeros(T, bool)
    v = np.zeros(T)
    v[1:-1] = np.sqrt((((a[2:] - a[:-2]) / 2)[:, :20] ** 2).mean(1))
    v[0], v[-1] = v[1], v[-2]
    v = uniform_filter1d(v, 3, mode="nearest")
    static = v < _d(speed_deg)
    segs = [(s, e) for s, e in _runs(static) if e - s >= min_len]
    static = np.zeros(T, bool)
    wn = w / (np.percentile(w[w > 0], 90) + 1e-9) if (w > 0).any() else w
    lam_t = strength * (1 - 0.7 * np.clip(wn, 0, 1))
    refs = []
    out = a.copy()
    for s, e in segs:
        ww = w[s:e] + 1e-6
        ref = (a[s:e] * ww[:, None]).sum(0) / ww.sum()
        refs.append(ref)
        ramp = np.minimum(np.minimum(np.arange(e - s) + 1, np.arange(e - s)[::-1] + 1) / 2.0, 1.0)
        if s == 0:
            ramp[: (e - s) // 2] = 1.0
        if e == T:
            ramp[(e - s) // 2:] = 1.0
        lam = (lam_t[s:e] * ramp)[:, None]
        out[s:e] = (1 - lam) * a[s:e] + lam * ref
        static[s:e] = True
    for (s0, e0), (s1, e1), r0, r1 in zip(segs[:-1], segs[1:], refs[:-1], refs[1:]):
        n = s1 - e0
        if n <= 0:
            continue
        u = ((np.arange(n) + 1) / (n + 1))[:, None]
        u = 0.5 - 0.5 * np.cos(np.pi * u)
        lam = (strength * np.clip(1 - 2 * wn[e0:s1], 0, 1))[:, None]   # only clearly unreliable frames
        out[e0:s1] = (1 - lam) * a[e0:s1] + lam * ((1 - u) * r0 + u * r1)
    return out, static


def _temporal_pass(raw, Rf, obs, w, smooth, stabilize):
    """Gap filling + robust zero-phase smoothing of angles and orientation quaternions."""
    T = len(raw)
    q = _mat_to_quat(Rf[obs])
    for i in range(1, len(q)):                                   # keep quaternions in one hemisphere
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    Q = np.zeros((T, 4))
    Q[obs] = q
    ang = _interp_fill(raw, obs)                                 # gap filling in ANGLE space
    Q = _interp_fill(Q, obs)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    ww = w + 1e-4
    static = np.zeros(T, bool)
    if smooth and T >= 3:
        for it in range(3):                                      # Tukey re-weighting against glitch frames
            sa, sq = _local_poly(ang, ww), _local_poly(Q, ww)
            if it == 2:
                break
            r = np.sqrt(((ang - sa) ** 2).mean(1) + ((Q - sq) ** 2).sum(1))
            sc = 1.4826 * np.median(r[obs]) + 2e-2
            ww = (w + 1e-4) * np.clip(1 - (r / (TUKEY_C * sc)) ** 2, 0, 1) ** 2 + 1e-6
        ang, Q = sa, sq / np.linalg.norm(sq, axis=1, keepdims=True)
        if stabilize:
            ang, static = stabilize_handshapes(ang, ww)
            ang = _local_poly(ang, np.ones(T), half=1, sigma=0.7, ridge=0.0) if T >= 3 else ang
    return np.clip(ang, LIMITS_LO, LIMITS_HI), Q, ww, static


def fit_hand_sequence(img, world, conf, src, occl=None, side=None, prior=None, bones=None,
                      smooth=True, stabilize=True, prior_weight=0.03, max_gap=12):
    """Fit the anatomical hand to one hand of one clip.

    img (T,21,3) px (+relative z), world (T,21,3) metres camera axes (NaN when
    missing), conf (T,), src (T,) 0..3, occl (T,) hand-over-hand flag.
    side 'L'/'R' = the subject's hand (None: guessed from finger curl).
    prior: object with residuals(angles) (see train_angle_prior / load_prior).

    Metric landmarks are used for finger articulation and hand orientation
    only (never for where the hand is); image landmarks pin the 2D shape.

    Returns dict
      angles   (T,21) rad, inside LIMITS, smoothed, gaps filled in angle space
      joints   (T,21,3) metres, wrist at origin, camera axes (= forward(...))
      orient   (T,3,3) hand-local -> camera
      quality  (T,) 0..1 (0 = never observed; 0.25 = interpolated short gap)
      observed (T,) bool, raw_angles (T,21) per-frame fit before the temporal pass
      static   (T,) bool frames inside a detected static-handshape segment
      wrist_px (T,2), px_per_m (T,): image = wrist_px + px_per_m * joints[..., :2]
      bones, side, fit_rms (T,) 2D residual in hand sizes
    """
    img = np.asarray(img, float)
    world = np.asarray(world, float)
    src = np.asarray(src).astype(int)
    T = len(img)
    occl = np.zeros(T, int) if occl is None else np.asarray(occl).astype(int)
    conf = np.ones(T) if conf is None else np.asarray(conf, float)
    has_i = np.isfinite(img[:, :, :2]).all((1, 2)) & (src > 0)
    has_w = np.isfinite(world).all((1, 2)) & has_i & (src >= 2)
    if side is None:
        side = (bones or {}).get("side") or detect_side(world[has_w])
    if bones is None:
        bones = estimate_bones(world, src, side)
    bones = dict(bones, side="R")                                # fit in the canonical right frame
    mir = _MIRROR if side == "L" else np.ones(3)
    I = np.nan_to_num(img) * mir
    W = (world - world[:, :1]) * mir
    H = bones["hand_size"]
    tmpl = np.vstack([np.zeros(3), bones["palm"][1:]])

    # per-frame size normalisation of the metric landmarks (rigid palm bones only)
    pb = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)]
    ref = np.array([np.linalg.norm(tmpl[a] - tmpl[b]) for a, b in pb])
    pk = np.array([0, 5, 9, 13, 17])
    Wn = W.copy()
    px_per_m = np.full(T, np.nan)
    for t in np.flatnonzero(has_w):
        k = np.median([np.linalg.norm(W[t, pk[a]] - W[t, pk[b]]) for a, b in pb] / ref)
        Wn[t] = W[t] / max(k, 1e-6)
        a = Wn[t, :, :2] - Wn[t, :, :2].mean(0)
        b = I[t, :, :2] - I[t, :, :2].mean(0)
        px_per_m[t] = (a * b).sum() / (a * a).sum()
    s = _robust_series(px_per_m, has_w & (px_per_m > 0))
    if s is None:                                                # no metric frames: palm seen face-on
        span = np.linalg.norm(I[has_i][:, 9, :2] - I[has_i][:, 0, :2], axis=1) if has_i.any() else np.array([40.0])
        s = np.full(T, np.percentile(span, 90) / H)

    X = np.full((T, _NP), np.nan)
    Rf = np.full((T, 3, 3), np.nan)
    rms = np.full(T, np.nan)
    prev_x, prev_R, gap = None, None, 0
    for t in range(T):
        if not has_i[t]:
            gap += 1
            continue
        unsure = (src[t] == 1) or bool(occl[t])
        wt = 0.0 if prev_x is None else (0.15 if unsure else 0.03) / (1 + gap)
        kw = dict(prev=(prev_x[_A] if prev_x is not None else np.zeros(D)), wt=wt,
                  wc=0.35 if unsure else 0.08, prior=prior, wp=prior_weight * (3.0 if unsure else 1.0))
        cands = []
        if has_w[t]:
            u = I[t, :, :2].mean(0) - s[t] * Wn[t, :, :2].mean(0)
            It = (I[t, :, :2] - u) / (s[t] * H)
            kw.update(Wt=Wn[t] / H, ww=WORLD_W, It=It, wi=1.0, Iz=np.zeros(21), wz=0.0, wo=0.0,
                      ls_bounds=(-LS_W, LS_W), wls=0.3)
            ang0, R0, t0 = _ik_right(Wn[t][None], bones)
            cands.append((np.r_[np.zeros(3), t0[0] / H, ang0[0], 0.0], R0[0]))
            if prev_x is not None:
                cands.append((np.r_[np.zeros(3), It[0], Wn[t, 0, 2] / H, prev_x[_A], 0.0], prev_R))
        else:
            # Holistic-only hand: 2D + coarse relative depth; the image scale is only
            # interpolated from metric frames, so let it float a little.
            It = (I[t, :, :2] - I[t, 0, :2]) / (s[t] * H)
            Iz = (I[t, :, 2] - I[t, 0, 2]) / (s[t] * H)
            kw.update(Wt=np.zeros((21, 3)), ww=0.0, It=It, wi=1.0, Iz=Iz, wz=0.25,
                      wo=0.1 if prev_x is not None else 0.0, ls_bounds=(-0.35, 0.35), wls=0.3)
            if prev_x is not None:
                cands.append((np.r_[np.zeros(3), It[0], 0.0, prev_x[_A], 0.0], prev_R))
            else:                                                # bootstrap from the 2.5D image hand
                ang0, R0, t0 = _ik_right((np.c_[It, Iz] * H)[None], bones)
                cands.append((np.r_[np.zeros(3), t0[0] / H, ang0[0], 0.0], R0[0]))
        best = None
        for x0, R0 in cands:
            fr = _Frame(bones, R0, **kw)
            x0 = fr.clip(x0)
            c = fr.cost(x0)
            if best is None or c < best[0]:
                best = (c, fr, x0, R0)
        _, fr, x0, R0 = best
        x, _ = fr.solve(x0)
        r2d = lambda f, x: np.sqrt(((np.exp(x[-1]) * f.points(x[None])[0][:, :2] - It) ** 2).sum(1).mean())
        rms[t] = r2d(fr, x)
        if has_w[t] and rms[t] > RESCUE_RMS and prev_x is not None:
            # metric landmarks look wrong (e.g. palm/back flip): trust the image + previous pose
            fr2 = _Frame(bones, prev_R, **dict(kw, ww=0.1 * WORLD_W, wt=0.1, wo=0.05, wc=0.35))
            x2, _ = fr2.solve(np.r_[np.zeros(3), It[0], Wn[t, 0, 2] / H, prev_x[_A], 0.0])
            if r2d(fr2, x2) < 0.7 * rms[t]:
                x, R0, rms[t] = x2, prev_R, r2d(fr2, x2)
        X[t], Rf[t] = x, R0 @ _exp(x[:3])
        prev_x, prev_R, gap = x, Rf[t], 0

    obs = has_i & np.isfinite(X[:, 0])
    out = dict(side=side, bones=dict(bones, side=side), observed=obs, fit_rms=rms)
    if not obs.any():
        out.update(angles=np.full((T, D), np.nan), raw_angles=np.full((T, D), np.nan),
                   joints=np.full((T, 21, 3), np.nan), orient=np.full((T, 3, 3), np.nan),
                   quality=np.zeros(T), wrist_px=np.full((T, 2), np.nan), px_per_m=s,
                   static=np.zeros(T, bool))
        return out

    raw = X[:, _A].copy()
    w = SRC_WEIGHT[src] * np.where(occl > 0, OCCL_WEIGHT, 1.0) * np.clip(conf, 0.3, 1.0) * obs
    w = w / (1 + (np.nan_to_num(rms, nan=1.0) / 0.07) ** 4)
    ang, Q, ww, static = _temporal_pass(raw, Rf, obs, w, smooth, stabilize)
    Rr = _quat_to_mat(Q)
    Jr = np.einsum("tij,tkj->tki", Rr, _fk_local(ang, bones))
    # image placement: per-frame scale (log, smoothed) and optimal wrist position, same smoother
    ls = _interp_fill(np.nan_to_num(X[:, -1]), obs)
    if smooth and T >= 3:
        ls = _local_poly(ls, ww)
    s = s * np.exp(ls)
    wp = np.zeros((T, 2))
    wp[obs] = (I[obs, :, :2] - s[obs, None, None] * Jr[obs, :, :2]).mean(1)
    wp = _interp_fill(wp, obs)
    if smooth and T >= 3:
        wp = _local_poly(wp, ww, half=2, sigma=0.8)

    quality = np.clip(ww, 0, 1) * obs
    idx = np.flatnonzero(obs)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            quality[a + 1:b] = 0.25
    M = np.outer(mir, mir)
    out.update(angles=ang, raw_angles=raw, joints=Jr * mir, orient=Rr * M, quality=quality,
               wrist_px=wp * mir[:2], px_per_m=s, static=static)
    return out


def symmetrize(fit_L, fit_R, weight=0.25, max_diff_deg=18.0, min_quality=0.3):
    """Weak two-hand symmetry (e.g. NAME: two H-hands).  Both hands are fitted
    in the same canonical right-hand angle space, so "mirrored" simply means
    equal angles.  If the finger angles of the two hands agree (median
    difference < max_diff_deg over co-observed frames) each hand is pulled
    towards the other by `weight` x the other hand's share of the confidence.
    Modifies the fits in place (angles + joints); returns True if applied."""
    aL, aR, qL, qR = fit_L["angles"], fit_R["angles"], fit_L["quality"], fit_R["quality"]
    both = (qL >= min_quality) & (qR >= min_quality)
    if weight <= 0 or both.sum() < 5:
        return False
    d = np.degrees(np.sqrt(np.median((aL[both, :20] - aR[both, :20]) ** 2, axis=0).mean()))
    if d > max_diff_deg:
        return False
    ok = np.isfinite(aL).all(1) & np.isfinite(aR).all(1)
    sh = (qR / (qL + qR + 1e-6))[:, None]
    nL = aL + weight * sh * (aR - aL)
    nR = aR + weight * (1 - sh) * (aL - aR)
    for f, n in ((fit_L, nL), (fit_R, nR)):
        f["angles"] = np.where(ok[:, None], n, f["angles"])
        f["joints"][ok] = forward(f["angles"][ok], f["orient"][ok], f["bones"])
    return True


def fit_track(track, prior=None, symmetry=0.25, **kw):
    """Fit both hands of a track dict (asl_v2.track.track) -> {'L': result, 'R': result}.
    symmetry: weight of the two-hand same-handshape tie (0 disables)."""
    out = {S: fit_hand_sequence(track[f"img_{S}"], track[f"world_{S}"], track[f"conf_{S}"],
                                track[f"src_{S}"], track.get("occl"), side=S, prior=prior, **kw)
           for S in "LR"}
    out["symmetric"] = symmetrize(out["L"], out["R"], symmetry) if symmetry else False
    return out


def place_in_body_frame(fit, center_px, scale_px, wrist_z=0.0):
    """Put a fitted hand into canonical body units (origin = center_px, 1 unit =
    scale_px image pixels, e.g. the shoulder midpoint / shoulder width of
    motion.py; y down, z away).  The wrist depth is NOT known from the hand
    fit (metric hand landmarks say nothing about the depth between the hands):
    pass wrist_z (scalar or (T,)) from the arm/pose solver.
    Returns (joints (T,21,3) in units, units_per_m (float))."""
    k = fit["px_per_m"] / scale_px
    J = fit["joints"] * k[:, None, None]
    J[:, :, :2] += ((fit["wrist_px"] - np.asarray(center_px, float)[:2]) / scale_px)[:, None]
    J[:, :, 2] += np.broadcast_to(np.asarray(wrist_z, float), (len(J),))[:, None]
    return J, float(np.nanmedian(k))


def _fit_job(args):
    name, tr, prior, kw = args
    if tr is None:
        from .track import track as _track
        tr = _track(name)
    return name, fit_track(tr, prior=prior, **kw)


def fit_clips(clips, prior=None, workers=4, **kw):
    """Fit many clips in parallel.  clips: list of words or dict word -> track dict.
    Returns dict word -> {'L': result, 'R': result}."""
    jobs = [(n, (clips[n] if isinstance(clips, dict) else None), prior, kw) for n in clips]
    if workers and workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            return dict(pool.map(_fit_job, jobs, chunksize=1))
    return dict(map(_fit_job, jobs))


def collect_angles(fits, min_quality=0.5):
    """Angle arrays (one per hand) of well-observed frames, ready for train_angle_prior."""
    out = []
    for f in fits.values():
        for S in "LR":
            m = f[S]["quality"] >= min_quality
            if m.any():
                out.append(f[S]["angles"][m])
    return out


# --------------------------------------------------------------------------
# Angle fits of 2D(+weak depth) landmark collections, prior training / loading
# --------------------------------------------------------------------------
PRIOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_prior.npz")


def _fit_2d_chunk(args):
    hands, right, group, prior, z_gain = args
    bones = dict(DEFAULT_BONES)
    H = bones["hand_size"]
    N = len(hands)
    ang = np.full((N, D), np.nan)
    rms = np.full(N, np.nan)
    prev_x, prev_R = None, None
    for i in range(N):
        P = np.asarray(hands[i], float) * (np.ones(3) if right[i] else _MIRROR)
        if not np.isfinite(P).all():
            prev_x = None
            continue
        if i == 0 or group[i] != group[i - 1]:
            prev_x = None
        It = P[:, :2] - P[0, :2]
        Iz = (P[:, 2] - P[0, 2]) * z_gain
        kw = dict(Wt=np.zeros((21, 3)), ww=0.0, It=It, wi=1.0, Iz=Iz, wz=0.15 if z_gain else 0.0,
                  prev=(prev_x[_A] if prev_x is not None else np.zeros(D)), wt=0.03 if prev_x is not None else 0.0,
                  wo=0.0, wc=0.25, prior=prior, wp=0.05, ls_bounds=(-0.1, 0.8), wls=0.1)
        ang0, R0, t0 = _ik_right((np.c_[It, Iz] * H)[None], bones)
        cands = [(np.r_[np.zeros(3), t0[0] / H, ang0[0], 0.0], R0[0])]
        if prev_x is not None:
            cands.append((np.r_[np.zeros(3), np.zeros(3), prev_x[_A], prev_x[-1]], prev_R))
        best = None
        for x0, R0 in cands:
            fr = _Frame(bones, R0, **kw)
            x, c = fr.solve(x0, nfev=40)
            if best is None or c < best[0]:
                best = (c, x, R0, fr)
        _, x, R0, fr = best
        ang[i] = x[_A]
        rms[i] = np.sqrt(((np.exp(x[-1]) * fr.points(x[None])[0][:, :2] - It) ** 2).sum(1).mean())
        prev_x, prev_R = x, R0 @ _exp(x[:3])
    return ang, rms


def fit_hands_2d(hands, right=None, groups=None, prior=None, z_gain=0.0, workers=4):
    """Angles of independent hand frames given as wrist-relative landmarks scaled
    to |landmark 9| = 1 in image axes (e.g. the ASL Citizen prior set).

    hands (N,21,3); right (N,) bool (left hands are mirrored in x and fitted
    with the same right-hand model); groups (N,) id of the clip+hand each frame
    belongs to (consecutive frames of one group are warm-started); z_gain
    rescales the depth column to hand-size units (0 = ignore depth).
    Orthographic fit with a free scale >= ~1 (a tilted palm looks shorter).
    Returns angles (N,21) and the 2D fit rms (N,) in hand sizes."""
    hands = np.asarray(hands)
    N = len(hands)
    right = np.ones(N, bool) if right is None else np.asarray(right).astype(bool)
    groups = np.arange(N) if groups is None else np.asarray(groups)
    cuts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    nchunk = max(1, min(len(cuts), 4 * max(workers, 1)))
    starts = [cuts[k] for k in np.linspace(0, len(cuts), nchunk + 1).astype(int)[:-1]] + [N]
    jobs = [(hands[a:b], right[a:b], groups[a:b], prior, z_gain) for a, b in zip(starts[:-1], starts[1:]) if b > a]
    if workers and workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_fit_2d_chunk, jobs, chunksize=1)
    else:
        res = list(map(_fit_2d_chunk, jobs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def prior_report(prior, angles):
    """Held-out diagnostics: mean Gaussian NLL (nats / frame, incl. the
    normalisation constant) and PCA reconstruction RMS error in degrees."""
    A = np.asarray(angles, float)
    A = A[np.isfinite(A).all(1)]
    logdet = np.log(prior.std).sum() + (D - prior.k) * np.log(prior.resid_std)
    nll = prior.neg_log_lik(A).mean() + logdet + 0.5 * D * np.log(2 * np.pi)
    z = (A - prior.mean) @ prior.components.T
    rec = np.degrees(np.sqrt(((A - prior.mean - z @ prior.components) ** 2).mean()))
    return dict(nll=float(nll), nll_per_dof=float(nll / D), recon_rms_deg=float(rec), n=len(A))


def train_prior_from_landmarks(path, holdout=0.1, n_components=10, max_rms=0.08, z_gain=1000.0,
                               save=PRIOR_PATH, workers=4, seed=0, cache=None):
    """Fit the angle parametrisation to a landmark collection (npz with `hands`
    (N,21,3) and `meta` (N,3) = file index, is_right, frame index), hold out
    `holdout` of the clips by file index, train the PCA+Gaussian prior on the
    rest and save it.  z_gain: the ASL Citizen file stores depth ~1000x smaller
    than x/y (not multiplied by the image width), 1000 brings it to hand-size
    order; it is only used as a weak cue.  Returns (prior, report dict)."""
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].astype(int)
    grp = meta[:, 0] * 2 + meta[:, 1]
    grp = np.cumsum(np.r_[0, (np.diff(grp) != 0) | (np.abs(np.diff(meta[:, 2])) > 6)])
    if cache and os.path.exists(cache):
        c = np.load(cache)
        ang, rms = c["angles"], c["rms"]
    else:
        ang, rms = fit_hands_2d(d["hands"], meta[:, 1] > 0, grp, z_gain=z_gain, workers=workers)
        if cache:
            np.savez_compressed(cache, angles=ang, rms=rms)
    files = np.unique(meta[:, 0])
    rng = np.random.default_rng(seed)
    held = rng.choice(files, max(1, int(round(holdout * len(files)))), replace=False)
    good = np.isfinite(ang).all(1) & (rms <= max_rms)
    te = np.isin(meta[:, 0], held)
    prior = train_angle_prior([ang[good & ~te]], n_components=n_components)
    rep = dict(train=prior_report(prior, ang[good & ~te]), heldout=prior_report(prior, ang[good & te]),
               n_frames=int(len(ang)), n_used=int(good.sum()), median_fit_rms=float(np.nanmedian(rms)),
               heldout_files=int(len(held)), n_components=int(prior.k))
    base = train_angle_prior([ang[good & ~te]], n_components=0)      # isotropic baseline (no synergies)
    rep["heldout_isotropic_nll"] = prior_report(base, ang[good & te])["nll"]
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez(save, **prior.to_dict(), angle_names=np.array(ANGLE_NAMES),
                 **{f"report_{k}": np.array(v if not isinstance(v, dict) else [v["nll"], v["recon_rms_deg"], v["n"]])
                    for k, v in rep.items()})
    return prior, rep


def load_prior(path=None):
    """Load the trained angle prior (asl_v2/models/hand_prior.npz); None if absent."""
    path = path or PRIOR_PATH
    if not os.path.exists(path):
        return None
    return AnglePrior.from_dict(np.load(path, allow_pickle=True))


# --------------------------------------------------------------------------
# Contacts: capsule hands (sampled as spheres), depth-only de-penetration
# --------------------------------------------------------------------------
# (a, b, kind, radius factor): finger phalanges taper; the palm is a fan of
# metacarpal capsules + the knuckle line + the thenar eminence.
_CAPS = ([(0, 5, "palm", 1.0), (0, 9, "palm", 1.0), (0, 13, "palm", 1.0), (0, 17, "palm", 0.9),
          (5, 9, "palm", 0.9), (9, 13, "palm", 0.9), (13, 17, "palm", 0.85),
          (0, 1, "palm", 1.0), (1, 2, "palm", 0.95), (2, 3, "finger", 1.15), (3, 4, "finger", 1.05)]
         + [(int(c[k]), int(c[k + 1]), "finger", f) for c in CHAIN[1:] for k, f in enumerate((1.0, 0.9, 0.8))])
R_FINGER_M, R_PALM_M = 0.0075, 0.011      # finger radius / palm half-thickness in metres
UNITS_PER_M = 1 / 0.38                    # canonical units: 1 = shoulder width ~ 0.38 m


def hand_spheres(joints, units_per_m=UNITS_PER_M, r_finger=R_FINGER_M, r_palm=R_PALM_M, n=4):
    """Sample the capsule hand as spheres: centres (...,S,3), radii (S,)."""
    J = np.asarray(joints, float)
    ts = (np.arange(n) + 0.5) / n
    cs, rs = [], []
    for a, b, kind, f in _CAPS:
        r = (r_palm if kind == "palm" else r_finger) * f * units_per_m
        for t in ts:
            cs.append((1 - t) * J[..., a, :] + t * J[..., b, :])
            rs.append(r)
    for tip in (4, 8, 12, 16, 20):
        cs.append(J[..., tip, :])
        rs.append(r_finger * 0.8 * units_per_m)
    return np.stack(cs, axis=-2), np.array(rs)


def _depth_need(CA, rA, CB, rB):
    """Extra depth separation needed so that A (in front, smaller z) clears B.
    Shifting along z leaves the xy distances unchanged, so this is exact for
    spheres: > 0 push apart, < 0 there is a free gap of that size, None = the
    hands do not overlap in the image at all."""
    d2 = ((CA[:, None, :2] - CB[None, :, :2]) ** 2).sum(-1)
    R2 = (rA[:, None] + rB[None]) ** 2
    ov = d2 < R2
    if not ov.any():
        return None
    g = (CB[None, :, 2] - CA[:, None, 2]) - np.sqrt(np.maximum(R2 - d2, 0))
    return float(-g[ov].min())


def penetration(joints_a, joints_b, **kw):
    """Deepest sphere-sphere interpenetration per frame (>0 = interpenetrating;
    <0 = clearance), same units as the joints."""
    A, rA = hand_spheres(joints_a, **kw)
    B, rB = hand_spheres(joints_b, **kw)
    A, B = A.reshape(-1, len(rA), 3), B.reshape(-1, len(rB), 3)
    d = np.linalg.norm(A[:, :, None] - B[:, None], axis=-1)
    return ((rA[:, None] + rB[None]) - d).max((1, 2))


def front_hand_from_evidence(src_L, src_R, conf_L, conf_R, occl, min_margin=0.1, bridge=3):
    """Which hand is nearer the camera in hand-over-hand frames.

    The hand whose landmarks are still detected confidently while the hands
    overlap is the unoccluded one, i.e. in front.  `occl`: (T,) overlap flag;
    pass  track['occl'] | image_overlap(JL, JR)  because the tracker's flag is
    only set when BOTH hands were found by the crop model, which is exactly
    when there is no evidence.  Returns (T,) int: +1 right hand in front, -1
    left hand in front, 0 no evidence.  One decision per overlap episode
    (episodes separated by <= `bridge` frames are merged)."""
    occl = np.asarray(occl) > 0
    T = len(occl)
    sc = lambda s, c: SRC_WEIGHT[np.asarray(s).astype(int)] * np.clip(np.asarray(c, float), 0.3, 1.0)
    e = (sc(src_R, conf_R) - sc(src_L, conf_L)) * occl
    run = maximum_filter1d(occl.astype(int), 2 * bridge + 1) > 0
    out = np.zeros(T, int)
    t = 0
    while t < T:
        if not run[t]:
            t += 1
            continue
        u = t
        while u < T and run[u]:
            u += 1
        n = max(occl[t:u].sum(), 1)
        m = e[t:u].sum() / n
        out[t:u] = 0 if abs(m) < min_margin else int(np.sign(m))
        t = u
    return out


def image_overlap(joints_L, joints_R, units_per_m=UNITS_PER_M, **kw):
    """(T,) bool: the capsule hands overlap in the image plane (hand-over-hand)."""
    JL, JR = np.asarray(joints_L, float).reshape(-1, 21, 3), np.asarray(joints_R, float).reshape(-1, 21, 3)
    SL, rL = hand_spheres(JL, units_per_m=units_per_m, **kw)
    SR, rR = hand_spheres(JR, units_per_m=units_per_m, **kw)
    ov = np.zeros(len(JL), bool)
    for t in range(len(JL)):
        if np.isfinite(SL[t]).all() and np.isfinite(SR[t]).all():
            ov[t] = _depth_need(SL[t], rL, SR[t], rR) is not None
    return ov


def detect_contact(joints_L, joints_R, occl=None, units_per_m=UNITS_PER_M, max_speed=0.04, **kw):
    """Frames where the hands are probably touching: their capsules overlap in
    the image (or the tracker flagged hand-over-hand) AND the relative hand
    speed is low / at a local minimum (contacts are holds or reversals).
    max_speed in units per frame.  Returns (T,) bool."""
    JL, JR = np.asarray(joints_L, float), np.asarray(joints_R, float)
    T = len(JL)
    ov = image_overlap(JL, JR, units_per_m, **kw)
    if occl is not None:
        ov |= np.asarray(occl) > 0
    rel = np.nan_to_num((JR.mean(1) - JL.mean(1))[:, :2])
    v = np.zeros(T)
    if T > 2:
        v[1:-1] = np.linalg.norm(rel[2:] - rel[:-2], axis=1) / 2
        v[0], v[-1] = v[1], v[-2]
    locmin = np.r_[False, (v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]), False] & (v < 3 * max_speed)
    return ov & ((v < max_speed) | locmin)


def _smooth_need(need, k, lower=None):
    """Temporal smoothing of a depth correction that never drops below the
    hard requirement `lower` (max filter + box filter of the same width is an
    upper envelope of `need`)."""
    if k <= 0 or len(need) < 2:
        out = need.copy()
    else:
        n = 2 * k + 1
        pos = uniform_filter1d(maximum_filter1d(np.maximum(need, 0), n, mode="nearest"), n, mode="nearest")
        neg = uniform_filter1d(np.minimum(need, 0), n, mode="nearest")
        if lower is not None:                                    # frames without overlap do not dilute a snap
            cnt = uniform_filter1d(np.isfinite(lower).astype(float), n, mode="nearest")
            neg = np.where(np.isfinite(lower), neg / np.maximum(cnt, 1.0 / n), neg)
        out = pos + neg
    return out if lower is None else np.maximum(out, lower)


def resolve_contacts(joints_L, joints_R, front=None, contact=None, units_per_m=UNITS_PER_M,
                     r_finger=R_FINGER_M, r_palm=R_PALM_M, share=(0.5, 0.5), snap=True,
                     max_snap=0.3, smooth=1):
    """Push interpenetrating hands apart along the camera depth axis (z).

    joints_L/R (T,21,3) or (21,3): both hands in ONE body frame (canonical
    units, z away from the camera; `units_per_m` converts the metric capsule
    radii).  front (T,) +1 right hand in front / -1 left / 0 unknown (see
    front_hand_from_evidence); unknown frames keep the current depth order.
    contact (T,) bool: frames where the hands are known to touch (track
    `occl`): with snap=True a depth gap up to `max_snap` is closed so that the
    hands touch instead of floating apart.  share = fraction of the correction
    taken by (L, R).  Hands only move in z, rigidly, by the minimum amount.

    Returns dict(L, R corrected joints, dz_L, dz_R (T,) applied depth shifts,
    front (T,) order used, need (T,) raw per-frame correction, overlap (T,)
    bool image overlap, gap (T,) depth clearance left (0 = touching, NaN = no
    overlap)).
    """
    JL, JR = np.asarray(joints_L, float), np.asarray(joints_R, float)
    single = JL.ndim == 2
    JL, JR = JL.reshape(-1, 21, 3).copy(), JR.reshape(-1, 21, 3).copy()
    T = len(JL)
    front = np.zeros(T, int) if front is None else np.broadcast_to(np.asarray(front), (T,)).astype(int)
    contact = np.zeros(T, bool) if contact is None else np.broadcast_to(np.asarray(contact) > 0, (T,))
    kw = dict(units_per_m=units_per_m, r_finger=r_finger, r_palm=r_palm)
    SL, rL = hand_spheres(JL, **kw)
    SR, rR = hand_spheres(JR, **kw)
    need = np.zeros(T)
    used = np.zeros(T, int)
    both = np.full((T, 2), np.nan)                               # need if R in front / if L in front
    ok = np.isfinite(JL).all((1, 2)) & np.isfinite(JR).all((1, 2))
    for t in np.flatnonzero(ok):
        nR = _depth_need(SR[t], rR, SL[t], rL)
        if nR is None:
            continue
        both[t] = nR, _depth_need(SL[t], rL, SR[t], rR)
    ov = np.isfinite(both[:, 0])
    # depth order: evidence first, otherwise the cheaper (current) order, held per overlap episode
    t = 0
    while t < T:
        if not ov[t]:
            t += 1
            continue
        u = t
        while u < T and ov[u]:
            u += 1
        ev = front[t:u][front[t:u] != 0]
        if len(ev):
            f = 1 if ev.sum() >= 0 else -1
        else:
            f = 1 if np.median(both[t:u, 0] - both[t:u, 1]) <= 0 else -1
        used[t:u] = f
        t = u
    lower = np.full(T, -np.inf)                                  # hard requirement (no penetration)
    for t in np.flatnonzero(ov):
        n = both[t, 0] if used[t] > 0 else both[t, 1]
        lower[t] = n
        if n > 0:
            need[t] = n                                          # interpenetrating / wrong order: push apart
        elif snap and contact[t] and -n <= max_snap:
            need[t] = n                                          # known contact: close the gap
    shift = _smooth_need(need, smooth, lower)
    if ov.any():                                                 # carry the order into the fade-out frames
        idx = np.flatnonzero(ov)
        used = used[idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(1)]]
    sgn = np.where(used >= 0, 1.0, -1.0)                         # +1: R in front -> R towards camera (-z)
    dz_R = -sgn * shift * share[1]
    dz_L = sgn * shift * share[0]
    JL[:, :, 2] += dz_L[:, None]
    JR[:, :, 2] += dz_R[:, None]
    res = dict(L=JL, R=JR, dz_L=dz_L, dz_R=dz_R, front=used, need=need, overlap=ov,
               gap=np.where(ov, shift - lower, np.nan))            # remaining depth clearance (0 = touching)
    if single:
        res.update(L=JL[0], R=JR[0])
    return res


def body_proxies(pose):
    """Torso + head proxies from a canonical pose (33,3) or (T,33,3) (shoulder
    midpoint origin, 1 unit = shoulder width, y down, z away).  Rough but
    conservative: the chest/face surface is ~0.25 units in front of the
    shoulder line."""
    p = np.asarray(pose, float)[..., :3]
    sh = (p[..., 11, :] + p[..., 12, :]) / 2
    ears = (p[..., 7, :] + p[..., 8, :]) / 2
    z0 = sh[..., 2]
    torso = dict(type="ellipsoid", c=sh + np.array([0.0, 0.80, 0.07]), radii=np.array([0.56, 1.05, 0.30]))
    head = dict(type="ellipsoid", c=np.stack([ears[..., 0], ears[..., 1], z0], -1), radii=np.array([0.21, 0.30, 0.25]))
    neck = dict(type="capsule", a=np.stack([ears[..., 0], ears[..., 1] + 0.2, z0 + 0.05], -1),
                b=sh + np.array([0.0, 0.05, 0.05]), r=0.15)
    return [torso, head, neck]


def _front_surface(proxy, xy, r, t, n=10):
    """z of the camera-facing surface of a proxy inflated by r, at image points xy (S,2).
    +inf where the ray misses the proxy."""
    def at(v):
        v = np.asarray(v, float)
        return v[t] if v.ndim == 2 else v
    if proxy["type"] == "ellipsoid":
        c, rad = at(proxy["c"]), np.asarray(proxy["radii"], float)
        A, B, C = rad[0] + r, rad[1] + r, rad[2] + r
        q = 1 - ((xy[:, 0] - c[0]) / A) ** 2 - ((xy[:, 1] - c[1]) / B) ** 2
        return np.where(q > 0, c[2] - C * np.sqrt(np.maximum(q, 0)), np.inf)
    a, b, R = at(proxy["a"]), at(proxy["b"]), proxy["r"] + r
    ts = np.linspace(0, 1, n)[:, None]
    pts = a + ts * (b - a)                                       # (n,3) spheres along the capsule axis
    d2 = ((xy[:, None] - pts[None, :, :2]) ** 2).sum(-1)
    z = np.where(d2 < R[:, None] ** 2, pts[None, :, 2] - np.sqrt(np.maximum(R[:, None] ** 2 - d2, 0)), np.inf)
    return z.min(1)


def keep_in_front(joints, proxies, units_per_m=UNITS_PER_M, r_finger=R_FINGER_M, r_palm=R_PALM_M,
                  margin=0.0, smooth=1):
    """Keep a hand in front of body proxies (touching is allowed).

    joints (T,21,3) or (21,3) in canonical units; proxies: list of
    dict(type='ellipsoid', c=(3,) or (T,3), radii=(3,)) or
    dict(type='capsule', a, b = (3,) or (T,3), r) in the same units.
    The hand is moved rigidly towards the camera (-z) just enough that none of
    its capsules is inside a proxy.  Returns (joints_new, dz (T,) <= 0).
    """
    J = np.asarray(joints, float)
    single = J.ndim == 2
    J = J.reshape(-1, 21, 3).copy()
    S, r = hand_spheres(J, units_per_m=units_per_m, r_finger=r_finger, r_palm=r_palm)
    need = np.zeros(len(J))
    for t in np.flatnonzero(np.isfinite(J).all((1, 2))):
        zf = np.full(len(r), np.inf)
        for p in proxies:
            zf = np.minimum(zf, _front_surface(p, S[t, :, :2], r + margin, t))
        need[t] = max(float((S[t, :, 2] - zf).max()), 0.0)
    dz = -_smooth_need(need, smooth)
    J[:, :, 2] += dz[:, None]
    return (J[0], dz[0]) if single else (J, dz)
