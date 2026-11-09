"""
2D illustrated renderer for asl_v2 motion: filled, shaded hands with knuckle
joints and nails, sleeved arms, and a face built from the tracked face mesh
(brows, eyes with irises, lips), so facial grammar is visible.
"""

import subprocess

import cv2
import numpy as np

from .motion import FACE, LH, P, RH

W, H, SS = 1280, 720, 2            # output size, supersampling factor
UNIT, CX, CY = 180, 640, 250       # canonical unit -> pixels

SKIN = (150, 185, 225)             # BGR
SKIN_DARK = (105, 140, 185)
SKIN_SHADE = (130, 165, 208)
NAIL = (190, 210, 235)
SHIRT = (90, 60, 40)
SHIRT_DARK = (60, 40, 25)
HAIR = (40, 55, 80)
LIP = (95, 105, 185)
BG_TOP, BG_BOT = np.array([245, 242, 238]), np.array([215, 210, 205])

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
FINGER_W = [0.075, 0.060, 0.062, 0.058, 0.050]     # base width per finger (units)

# MediaPipe face-mesh contours
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LIPS_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
            181, 91, 146]
LIPS_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87,
           178, 88, 95]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
NOSE_BASE = [98, 97, 2, 326, 327]
IRIS_L, IRIS_R = range(473, 478), range(468, 473)


def _px(p):
    return np.stack([CX + UNIT * p[..., 0], CY + UNIT * p[..., 1]], -1) * SS


def _pt(p):
    return tuple(int(round(v)) for v in p)


def _capsule(img, a, b, ra, rb, color):
    """Tapered capsule from a (radius ra) to b (radius rb), in pixels."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-3:
        cv2.circle(img, _pt(a), int(max(ra, rb)), color, -1, cv2.LINE_AA)
        return
    n = np.array([-d[1], d[0]]) / L
    poly = np.array([a + n * ra, b + n * rb, b - n * rb, a - n * ra])
    cv2.fillConvexPoly(img, poly.round().astype(np.int32), color, cv2.LINE_AA)
    cv2.circle(img, _pt(a), int(ra), color, -1, cv2.LINE_AA)
    cv2.circle(img, _pt(b), int(rb), color, -1, cv2.LINE_AA)


def _poly(img, pts, color, outline=None, thick=2):
    pts = pts.round().astype(np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(img, [pts], True, outline, thick * SS, cv2.LINE_AA)


def _background():
    g = np.linspace(0, 1, H * SS)[:, None, None]
    return np.broadcast_to(BG_TOP * (1 - g) + BG_BOT * g, (H * SS, W * SS, 3)).astype(np.uint8).copy()


def _draw_arm(img, f, s, e, w):
    S, E, Wr = _px(f[P + s]), _px(f[P + e]), _px(f[P + w])
    r = UNIT * SS
    _capsule(img, S, E, 0.135 * r, 0.115 * r, SHIRT_DARK)
    _capsule(img, E, Wr, 0.118 * r, 0.090 * r, SHIRT_DARK)
    _capsule(img, S, E, 0.120 * r, 0.100 * r, SHIRT)
    _capsule(img, E, Wr, 0.103 * r, 0.075 * r, SHIRT)


def _draw_hand(img, hand):
    """hand: (21,3) canonical. Painter's order: palm, then fingers far-to-near."""
    px = _px(hand)
    r = UNIT * SS
    palm_idx = [0, 1, 5, 9, 13, 17]
    palm = px[palm_idx]
    hull = cv2.convexHull(palm.astype(np.float32)).reshape(-1, 2)
    # wrist stub joining forearm and palm
    _capsule(img, px[0], (px[5] + px[17]) / 2 * 0.3 + px[0] * 0.7, 0.07 * r, 0.075 * r, SKIN_DARK)
    _poly(img, hull, SKIN, SKIN_DARK, 2)
    # back-of-hand shading toward the wrist
    cv2.circle(img, _pt(px[0] * 0.6 + px[9] * 0.4), int(0.045 * r), SKIN_SHADE, -1, cv2.LINE_AA)
    order = sorted(range(5), key=lambda i: -hand[FINGERS[i][-1], 2])
    for i in order:
        chain = [0, *FINGERS[i]] if i == 0 else list(FINGERS[i])
        base = FINGER_W[i] / 2 * r
        widths = np.linspace(base, base * 0.72, len(chain))
        for k in range(len(chain) - 1):          # outline pass
            _capsule(img, px[chain[k]], px[chain[k + 1]], widths[k] + SS * 1.5, widths[k + 1] + SS * 1.5, SKIN_DARK)
        for k in range(len(chain) - 1):          # fill pass
            _capsule(img, px[chain[k]], px[chain[k + 1]], widths[k], widths[k + 1], SKIN)
        for k in range(1, len(chain) - 1):       # knuckle creases
            cv2.circle(img, _pt(px[chain[k]]), max(1, int(widths[k] * 0.35)), SKIN_SHADE, -1, cv2.LINE_AA)
        tip, pre = px[chain[-1]], px[chain[-2]]
        nail = tip * 0.72 + pre * 0.28
        d = tip - pre
        ang = np.degrees(np.arctan2(d[1], d[0]))
        cv2.ellipse(img, _pt(nail), (max(1, int(widths[-1] * 0.55)), max(1, int(widths[-1] * 0.42))),
                    ang, 0, 360, NAIL, -1, cv2.LINE_AA)


def _draw_head(img, f):
    face = f[FACE:]
    px = _px(face)
    oval = px[FACE_OVAL]
    top = px[10]
    chin = px[152]
    up = (top - chin) / (np.linalg.norm(top - chin) + 1e-9)
    # hair: enlarged oval shifted up, drawn behind the face
    c = oval.mean(0)
    hair = c + (oval - c) * 1.12 + up * 0.05 * UNIT * SS
    _poly(img, hair[[i for i in range(len(hair)) if np.dot(hair[i] - c, up) > -0.2 * UNIT * SS]], HAIR)
    for ear in (234, 454):
        cv2.ellipse(img, _pt(px[ear]), (int(0.05 * UNIT * SS), int(0.09 * UNIT * SS)),
                    0, 0, 360, SKIN_DARK, -1, cv2.LINE_AA)
    _poly(img, oval, SKIN, SKIN_DARK, 2)
    for brow in (BROW_L, BROW_R):
        b = px[brow]
        half = len(b) // 2
        poly = np.concatenate([b[:half], b[half:][::-1]])
        _poly(img, poly, HAIR)
    for eye, iris in ((EYE_L, IRIS_L), (EYE_R, IRIS_R)):
        _poly(img, px[eye], (250, 250, 250), SKIN_DARK, 1)
        ir = px[list(iris)]
        rad = max(2, int(np.linalg.norm(ir[1] - ir[3]) / 2))
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [px[eye].round().astype(np.int32)], 255)
        tmp = img.copy()
        cv2.circle(tmp, _pt(ir[0]), rad, (70, 60, 50), -1, cv2.LINE_AA)
        cv2.circle(tmp, _pt(ir[0]), max(1, rad // 2), (20, 20, 20), -1, cv2.LINE_AA)
        img[mask > 0] = tmp[mask > 0]
    cv2.polylines(img, [px[NOSE].round().astype(np.int32)], False, SKIN_DARK, SS, cv2.LINE_AA)
    cv2.polylines(img, [px[NOSE_BASE].round().astype(np.int32)], False, SKIN_DARK, SS * 2, cv2.LINE_AA)
    _poly(img, px[LIPS_OUT], LIP)
    _poly(img, px[LIPS_IN], (60, 50, 110))


def render_frame(f):
    img = _background()
    r = UNIT * SS
    ls, rs, lhip, rhip = (_px(f[P + i]) for i in (11, 12, 23, 24))
    neck_base = (ls + rs) / 2
    chin = _px(f[FACE + 152])
    _capsule(img, neck_base, chin, 0.13 * r, 0.11 * r, SKIN_DARK)
    torso = np.array([ls + [0, -0.06 * r], rs + [0, -0.06 * r], rs + [-0.08 * r, 0.25 * r],
                      rs + [-0.02 * r, 0.9 * r], rhip, lhip, ls + [0.02 * r, 0.9 * r], ls + [0.08 * r, 0.25 * r]])
    _poly(img, torso, SHIRT, SHIRT_DARK, 2)
    cv2.ellipse(img, _pt(neck_base + [0, -0.02 * r]), (int(0.17 * r), int(0.07 * r)), 0, 0, 180,
                SHIRT_DARK, -1, cv2.LINE_AA)
    _draw_head(img, f)
    # far arm first (by wrist depth), each hand drawn right after its arm
    sides = sorted([(11, 13, 15, LH), (12, 14, 16, RH)], key=lambda s: -f[P + s[2], 2])
    for s, e, w, off in sides:
        _draw_arm(img, f, s, e, w)
        _draw_hand(img, f[off:off + 21])
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def write_video(seq, path, fps=30):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for f in seq:
        proc.stdin.write(render_frame(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
