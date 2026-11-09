"""
Turn raw per-clip MediaPipe keypoints into one clean, continuous motion.

Fixes over the Space's video_gen.py:
  1. Normalisation  - every clip is moved/scaled so the signer's shoulders sit
     in the same place with the same width, so bodies don't jump between
     words recorded by different signers or framed differently.
  2. Resting hands  - a hand that is not signing (often hanging below the
     clip's bottom edge, so never detected) is placed relaxed at the side,
     with a natural hand shape, from the very first frame.
  3. Gap filling    - short hand-tracking dropouts are interpolated; the hand
     is always attached to the body's wrist.
  4. Transitions    - eased (cosine) in-betweens whose length depends on how
     far the hands travel, a lead-in from and lead-out to the rest pose, and
     idle frames at clip edges are trimmed.
  5. Smoothing      - light temporal Gaussian filter to remove tracker jitter.

Output frame layout (canonical units: origin = shoulder midpoint, 1 unit =
shoulder width, y down, z towards the camera is negative):
  [0:33] pose, [33:54] left hand, [54:75] right hand, [75:553] face mesh.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

P, LH, RH, FACE, N = 0, 33, 54, 75, 553
SIDES = {  # subject's side: (shoulder, elbow, wrist, hip, hand offset, outward sign)
    "L": (11, 13, 15, 23, LH, +1.0),
    "R": (12, 14, 16, 24, RH, -1.0),
}
UPPER_ARM, FOREARM = 0.85, 0.75
MAX_GAP = 12          # frames of missing hand tracking that get interpolated
BLEND = 13            # frames over which a hand eases between rest and signing
FPS = 30


# --------------------------------------------------------------------------
# Relaxed hand template (subject's left hand hanging at the side)
# --------------------------------------------------------------------------
def _rest_hand_left():
    pts = np.zeros((21, 3))
    flex_dir = np.array([-1.0, 0.0, 0.0])          # palm faces the thigh (-x)
    fingers = {  # MCP index: (mcp position, segment lengths, flexion degrees)
        5: ((0.0, 0.255, -0.060), (0.105, 0.062, 0.050), (10, 16, 10)),
        9: ((0.0, 0.270, -0.020), (0.115, 0.072, 0.055), (12, 18, 12)),
        13: ((0.0, 0.258, 0.020), (0.105, 0.068, 0.050), (14, 20, 12)),
        17: ((0.0, 0.232, 0.055), (0.082, 0.050, 0.045), (16, 22, 14)),
    }
    for mcp, (pos, lens, angles) in fingers.items():
        p = np.array(pos)
        pts[mcp] = p
        theta = 0.0
        for k, (ln, ang) in enumerate(zip(lens, angles)):
            theta += np.radians(ang)
            d = np.array([0, np.cos(theta), 0]) + flex_dir * np.sin(theta)
            p = p + ln * d
            pts[mcp + k + 1] = p
    pts[1:5] = [(-0.020, 0.060, -0.070), (-0.035, 0.120, -0.095),
                (-0.045, 0.165, -0.100), (-0.050, 0.200, -0.100)]
    yaw = np.radians(20)                             # show a bit of the back of the hand
    rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    return pts @ rot.T


REST_HAND = {"L": _rest_hand_left()}
REST_HAND["R"] = REST_HAND["L"] * np.array([-1.0, 1.0, 1.0])


def _orient(hand, direction):
    """Rotate a hand template (fingers along +y) so it points along `direction` (xy)."""
    d = direction[:2] / (np.linalg.norm(direction[:2]) + 1e-9)
    c, s = d[1], -d[0]                               # rotation taking (0,1) to d
    out = hand.copy()
    out[:, 0] = c * hand[:, 0] - s * hand[:, 1]
    out[:, 1] = s * hand[:, 0] + c * hand[:, 1]
    return out


def rest_arm(shoulder, sign):
    # relaxed: upper arm hanging, elbow softly bent so the hand rests forward at hip level
    elbow = shoulder + np.array([0.12 * sign, 0.83, 0.10])
    wrist = elbow + np.array([0.0 * sign, 0.55, -0.50])
    return elbow, wrist


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fill(arr, valid, max_gap=None):
    """Linear-interpolate invalid frames (per coordinate); hold at the edges.
    Returns the filled array and a mask of frames that were filled by
    interpolation inside a gap no longer than max_gap."""
    T = len(arr)
    out = arr.copy()
    idx = np.flatnonzero(valid)
    short = np.zeros(T, bool)
    if len(idx) == 0:
        return out, short
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(np.arange(T), idx, flat[idx, j])
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= (max_gap or T) + 1:
            short[a + 1:b] = True
    return flat.reshape(arr.shape), short


def _ease(n):
    """Minimum-jerk time profile (Flash & Hogan 1985), the velocity shape of human reaching."""
    t = np.arange(1, n + 1) / (n + 1)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _kabsch(A, B):
    """Rotation R minimising |R A - B| for wrist-relative point sets (N,3)."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def _rot_power(R, t):
    """R^t via axis-angle (slerp from identity)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.eye(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = angle * t
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def blend_hand(A, B, t):
    """Blend two wrist-relative hand shapes like a real hand: rotate the whole hand
    (slerp) while morphing the finger configuration, so a hand turning over never
    collapses through itself as a straight-line point blend would."""
    if t <= 0:
        return A
    if t >= 1:
        return B
    R = _kabsch(A, B)
    local = (1 - t) * A + t * (B @ R)          # B expressed in A's orientation, blended
    return local @ _rot_power(R, t).T


def _hand_turn(a, b):
    """Rotation angle (rad) between two frames' hand orientations, max over both hands."""
    ang = 0.0
    for off in (LH, RH):
        R = _kabsch(a[off:off + 21] - a[off], b[off:off + 21] - b[off])
        ang = max(ang, np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang


def _smooth_weight(active, width):
    if active.all() or not active.any():
        return active.astype(float)
    k = np.hanning(width + 2)[1:-1]
    k /= k.sum()
    pad = width
    a = np.pad(active.astype(float), pad, mode="edge")
    return np.convolve(a, k, mode="same")[pad:-pad]


# --------------------------------------------------------------------------
# Per-clip processing
# --------------------------------------------------------------------------
def process_clip(kp, return_offset=False):
    pose, W, H = kp["pose"].copy(), *kp["size"]
    T = len(pose)
    sh_l, sh_r = pose[:, 11, :3], pose[:, 12, :3]
    center = np.median((sh_l + sh_r) / 2, axis=0)
    scale = np.median(np.linalg.norm((sh_l - sh_r)[:, :2], axis=1))

    def norm(a):
        return (a - center) / scale

    out = np.zeros((T, N, 3))
    body = norm(pose[:, :, :3])
    vis = pose[:, :, 3]
    # Hips are outside the source frame: use a fixed, plausible torso.
    body[:, 23] = np.median((body[:, 11] + body[:, 12]) / 2, axis=0) + [0.36, 1.85, 0]
    body[:, 24] = body[:, 23] - [0.72, 0, 0]
    out[:, P:P + 33] = body

    weights = {}
    for side, (s, e, w, _, off, sign) in SIDES.items():
        hand_raw = kp["lh" if side == "L" else "rh"]
        det = ~np.isnan(hand_raw[:, 0, 0])
        shoulder = body[:, s]
        rest_e, rest_w = rest_arm(shoulder, sign)

        if det.sum() == 0:              # hand never signs in this clip -> rests
            elbow, wrist, hand_off = rest_e, rest_w, None
            weight = np.zeros(T)
        else:
            hand = norm(hand_raw)
            hand_off = hand - hand[:, :1]                     # wrist-relative shape
            world = kp.get("world_lh" if side == "L" else "world_rh")
            if world is not None:
                # Metric 3D hand shape from the hand model (real finger depth), scaled so its
                # image-plane projection matches the 2D landmarks (per-clip median scale).
                wv = ~np.isnan(world[:, 0, 0]) & det
                if wv.sum() >= 3:
                    w2 = world[wv, :, :2] - world[wv, :, :2].mean(1, keepdims=True)
                    i2 = hand_raw[wv, :, :2] - hand_raw[wv, :, :2].mean(1, keepdims=True)
                    px_per_m = np.median(np.linalg.norm(i2, axis=(1, 2)) / (np.linalg.norm(w2, axis=(1, 2)) + 1e-9))
                    w_off = (world - world[:, :1]) * px_per_m / scale
                    hand_off = np.where(wv[:, None, None], w_off, hand_off)
            hand_off, short = _fill(hand_off, det, MAX_GAP)
            wrist = np.where(det[:, None], hand[:, 0], body[:, w])
            # hand z is relative to its own wrist; anchor at the body wrist depth
            wrist[:, 2] = body[:, w, 2]
            wrist_ok = det | ((vis[:, w] > 0.6) & (pose[:, w, 1] < 0.97 * H))
            wrist, _ = _fill(wrist, wrist_ok)
            elbow = body[:, e].copy()
            elbow_ok = vis[:, e] > 0.4
            elbow, _ = _fill(elbow, elbow_ok) if elbow_ok.any() else (rest_e, None)
            active = det | short
            weight = _smooth_weight(active, BLEND)

        a = weight[:, None]
        if hand_off is None:
            elbow_f, wrist_f = rest_e, rest_w
        else:
            elbow_f = a * elbow + (1 - a) * rest_e
            wrist_f = a * wrist + (1 - a) * rest_w
        fore = wrist_f - elbow_f
        rest_off = np.stack([_orient(REST_HAND[side], fore[t]) for t in range(T)])
        if hand_off is None:
            hand_f = rest_off
        else:
            hand_f = np.stack([blend_hand(rest_off[t], hand_off[t], weight[t]) for t in range(T)])
        out[:, P + e], out[:, P + w] = elbow_f, wrist_f
        out[:, off:off + 21] = wrist_f[:, None] + hand_f
        weights[side] = weight

    face = norm(kp["face"])
    face_ok = ~np.isnan(face[:, 0, 0])
    if face_ok.any():
        face, _ = _fill(face, face_ok)
    else:
        face = np.repeat(body[:, :1], 478, axis=1)
    out[:, FACE:] = face

    # Trim idle frames (no hand signing) at the start and end of the clip.
    busy = np.flatnonzero(np.maximum(weights["L"], weights["R"]) > 0.5)
    start = 0
    if len(busy):
        start = max(busy[0] - 2, 0)
        out = out[start: busy[-1] + 3]
    return (out, start) if return_offset else out


def rest_frame(frame):
    """Copy of `frame` with both arms and hands put in the rest pose."""
    f = frame.copy()
    for side, (s, e, w, _, off, sign) in SIDES.items():
        el, wr = rest_arm(f[P + s], sign)
        f[P + e], f[P + w] = el, wr
        f[off:off + 21] = wr + _orient(REST_HAND[side], wr - el)
    return f


def _transition(a, b, n):
    frames = []
    for t in _ease(n):
        f = (1 - t) * a + t * b
        for off in (LH, RH):                      # hands: rotate + morph around the moving wrist
            f[off:off + 21] = f[off] + blend_hand(a[off:off + 21] - a[off], b[off:off + 21] - b[off], t)
        frames.append(f)
    return np.stack(frames)


def _duration(a, b):
    """Transition length from wrist travel (shoulder widths) and hand rotation."""
    return int(np.clip(round(5 + 8 * _travel(a, b) + 7 * _hand_turn(a, b) / np.pi), 6, 22))


def _travel(a, b):
    return max(np.linalg.norm(a[LH] - b[LH]), np.linalg.norm(a[RH] - b[RH]))


def retime(clip, speed):
    """Resample a clip in time (dictionary citation forms are slower than fluent signing)."""
    T = len(clip)
    n = max(int(round(T / speed)), 2)
    src = np.linspace(0, T - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return (1 - w) * clip[lo] + w * clip[hi]


def best_cut(a, b, window=9):
    """Closest pair of frames between the tail of `a` and the head of `b`
    (idea from ZurichNLP spoken-to-signed-translation's connection search)."""
    wa, wb = min(window, len(a) - 1), min(window, len(b) - 1)
    best = (np.inf, len(a) - 1, 0)
    for i in range(len(a) - 1 - wa, len(a)):
        for j in range(0, wb + 1):
            cost = _travel(a[i], b[j]) + 0.02 * ((len(a) - 1 - i) + j)
            if cost < best[0]:
                best = (cost, i, j)
    return best[1], best[2]


def stitch(clips, lead=10, hold=8, speed=1.2):
    """Concatenate processed clips with eased transitions + rest lead-in/out."""
    clips = [retime(c, speed) for c in clips]
    for k in range(len(clips) - 1):
        i, j = best_cut(clips[k], clips[k + 1])
        clips[k], clips[k + 1] = clips[k][:i + 1], clips[k + 1][j:]
    parts = []
    start_rest = rest_frame(clips[0][0])
    parts.append(np.repeat(start_rest[None], hold, axis=0))
    prev = start_rest
    for clip in clips:
        parts.append(_transition(prev, clip[0], _duration(prev, clip[0])))
        parts.append(clip)
        prev = clip[-1]
    end_rest = rest_frame(prev)
    parts.append(_transition(prev, end_rest, max(lead, _duration(prev, end_rest))))
    parts.append(np.repeat(end_rest[None], hold, axis=0))
    seq = np.concatenate(parts)
    seq[:, :LH] = gaussian_filter1d(seq[:, :LH], 1.2, axis=0, mode="nearest")
    seq[:, LH:FACE] = gaussian_filter1d(seq[:, LH:FACE], 0.9, axis=0, mode="nearest")
    seq[:, FACE:] = gaussian_filter1d(seq[:, FACE:], 1.0, axis=0, mode="nearest")
    return seq
