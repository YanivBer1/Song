"""
Frame-by-frame tracking v3: Holistic (body, face, coarse hands) + MediaPipe
HandLandmarker run on upscaled crops around each wrist.

Why: in the dictionary clips hands are only ~40-60 px tall. Running the
dedicated hand model on a 256 px upscaled crop gives much more precise finger
joints, and HandLandmarker also returns *metric 3D world landmarks* (metres,
hand-centred), i.e. real finger depth instead of MediaPipe's image-z guess.

Per frame and per hand (subject's L / R), saved arrays:
  img_L/img_R     (T,21,3) image px (x,y) + relative z (px scale)
  world_L/world_R (T,21,3) metres, hand-centred, camera axes (x right, y down, z away)
  conf_L/conf_R   (T,)     detection score (0 = not found)
  src_L/src_R     (T,)     0 none, 1 holistic only, 2 crop model, 3 both agree
  occl            (T,)     1 if the hands overlap in the image (hand-over-hand)
plus pose (T,33,4), face (T,478,3), size, fps (as in extract.py).
"""

import os

import cv2
import numpy as np

from .extract import CLIP_DIR, clip_path

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
TRACK_DIR = os.path.join(os.path.dirname(HERE), "cache", "tracks")
CROP = 256
POSE_HAND = {"L": (15, 17, 19, 21, 13), "R": (16, 18, 20, 22, 14)}  # wrist, pinky, index, thumb, elbow


def _landmarker():
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL), num_hands=2,
        running_mode=vision.RunningMode.IMAGE, min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25)
    return vision.HandLandmarker.create_from_options(opts)


def _roi(pose_px, side, hol_hand):
    """Square ROI (cx, cy, size) around a hand, from Holistic hand or pose hand points."""
    if hol_hand is not None:
        pts = hol_hand[:, :2]
        c = (pts.min(0) + pts.max(0)) / 2
        size = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.9
    else:
        w, pk, ix, th, el = POSE_HAND[side]
        pts = pose_px[[w, pk, ix, th], :2]
        fore = pose_px[w, :2] - pose_px[el, :2]
        c = pts.mean(0) + 0.25 * fore
        size = max(np.linalg.norm(fore) * 1.3, np.ptp(pts[:, 0]) * 3, 40)
    return c, max(size, 36.0)


def _crop(frame, c, size):
    h, w = frame.shape[:2]
    s = CROP / size
    M = np.array([[s, 0, CROP / 2 - s * c[0]], [0, s, CROP / 2 - s * c[1]]], np.float32)
    patch = cv2.warpAffine(frame, M, (CROP, CROP), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return patch, M


def track_clip(video_path):
    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(model_complexity=2, refine_face_landmarks=True,
                                              smooth_landmarks=True)
    hands = _landmarker()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rec = {k: [] for k in ("pose", "face", "img_L", "img_R", "world_L", "world_R",
                           "conf_L", "conf_R", "src_L", "src_R", "occl")}
    last_pose = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = holistic.process(rgb)
        if r.pose_landmarks is not None:
            pose = np.array([[l.x * w, l.y * h, l.z * w, l.visibility] for l in r.pose_landmarks.landmark], np.float32)
            last_pose = pose
        elif last_pose is not None:
            pose = last_pose
        else:
            continue
        face = (np.array([[l.x * w, l.y * h, l.z * w] for l in r.face_landmarks.landmark], np.float32)
                if r.face_landmarks else np.full((478, 3), np.nan, np.float32))
        hol = {}
        for side, lms in (("L", r.left_hand_landmarks), ("R", r.right_hand_landmarks)):
            hol[side] = (np.array([[l.x * w, l.y * h, l.z * w] for l in lms.landmark], np.float32)
                         if lms else None)

        found = {}
        for side in ("L", "R"):
            wrist_vis = pose[POSE_HAND[side][0], 3]
            if hol[side] is None and (wrist_vis < 0.5 or pose[POSE_HAND[side][0], 1] > 0.98 * h):
                continue                                   # hand clearly out of frame
            c, size = _roi(pose, side, hol[side])
            patch, M = _crop(frame, c, size)
            res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            R2 = np.eye(2)
            if not res.hand_landmarks:
                # Rotation augmentation: the detector is trained mostly on upright hands; a hand
                # pointing down/sideways is often found once its crop is rotated upright.
                for ang in (180, 90, 270):
                    R = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), ang, 1.0)
                    rot = cv2.warpAffine(patch, R, (CROP, CROP), borderValue=(255, 255, 255))
                    r2 = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)))
                    if r2.hand_landmarks:
                        # fold the rotation into the crop transform so the code below maps back
                        M = np.vstack([R, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
                        M = M[:2].astype(np.float32)
                        R2 = R[:, :2]
                        res = r2
                        break
            Minv = cv2.invertAffineTransform(M)
            expect = hol[side][0, :2] if hol[side] is not None else pose[POSE_HAND[side][0], :2]
            best = None
            for k, lms in enumerate(res.hand_landmarks):
                pts = np.array([[l.x * CROP, l.y * CROP, l.z * CROP] for l in lms], np.float32)
                full = pts.copy()
                full[:, :2] = pts[:, :2] @ Minv[:, :2].T + Minv[:, 2]
                full[:, 2] = pts[:, 2] / np.sqrt(abs(np.linalg.det(M[:, :2])))
                d = np.linalg.norm(full[0, :2] - expect)
                score = res.handedness[k][0].score
                if best is None or d < best[0]:
                    world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[k]], np.float32)
                    world[:, :2] = world[:, :2] @ R2          # back to the unrotated image axes
                    best = (d, full, world, score)
            if best is not None and best[0] < 1.2 * size:
                found[side] = best

        # identity check: a crop detection must match *its own* Holistic hand better than the
        # other hand (in hand-over-hand frames the crop often contains both hands and the
        # model can lock onto the wrong one), and must not stray far from it.
        for side in list(found):
            other = "R" if side == "L" else "L"
            img = found[side][1][:, :2]
            span = max(np.ptp(img[:, 1]), np.ptp(img[:, 0]), 1e-6)
            d_own = np.linalg.norm(hol[side][:, :2] - img, axis=1).mean() if hol[side] is not None else None
            d_oth = np.linalg.norm(hol[other][:, :2] - img, axis=1).mean() if hol[other] is not None else None
            if d_oth is not None and ((d_own is None and d_oth < 0.5 * span) or
                                      (d_own is not None and d_oth < d_own)):
                del found[side]       # it is actually the other hand
            elif d_own is not None and d_own > 0.45 * span:
                del found[side]

        # hand-over-hand: both crops locked onto the same physical hand
        occl = 0
        if "L" in found and "R" in found:
            gap = np.linalg.norm(found["L"][1][:, :2] - found["R"][1][:, :2], axis=1).mean()
            span = np.ptp(found["L"][1][:, 1]) + 1e-6
            if gap < 0.25 * span:
                drop = "L" if found["L"][0] > found["R"][0] else "R"
                del found[drop]
                occl = 1
            else:
                boxes = [np.r_[found[s][1][:, :2].min(0), found[s][1][:, :2].max(0)] for s in "LR"]
                ix = min(boxes[0][2], boxes[1][2]) - max(boxes[0][0], boxes[1][0])
                iy = min(boxes[0][3], boxes[1][3]) - max(boxes[0][1], boxes[1][1])
                occl = int(ix > 0 and iy > 0)

        for side in ("L", "R"):
            if side in found:
                _, img, world, score = found[side]
                agree = hol[side] is not None and np.linalg.norm(hol[side][:, :2] - img[:, :2], axis=1).mean() < 0.15 * np.ptp(img[:, 1])
                rec[f"img_{side}"].append(img)
                rec[f"world_{side}"].append(world)
                rec[f"conf_{side}"].append(score)
                rec[f"src_{side}"].append(3 if agree else 2)
            elif hol[side] is not None:
                rec[f"img_{side}"].append(hol[side])
                rec[f"world_{side}"].append(np.full((21, 3), np.nan, np.float32))
                rec[f"conf_{side}"].append(0.5)
                rec[f"src_{side}"].append(1)
            else:
                rec[f"img_{side}"].append(np.full((21, 3), np.nan, np.float32))
                rec[f"world_{side}"].append(np.full((21, 3), np.nan, np.float32))
                rec[f"conf_{side}"].append(0.0)
                rec[f"src_{side}"].append(0)
        rec["pose"].append(pose)
        rec["face"].append(face)
        rec["occl"].append(occl)
    cap.release()
    holistic.close()
    hands.close()
    out = {k: np.asarray(v, np.float32) for k, v in rec.items()}
    out["size"] = np.array([w, h])
    out["fps"] = np.array(fps)
    return out


def track(word, url=None):
    os.makedirs(TRACK_DIR, exist_ok=True)
    cache = os.path.join(TRACK_DIR, f"{word}.npz")
    if not os.path.exists(cache):
        np.savez_compressed(cache, **track_clip(clip_path(word, url)))
    return dict(np.load(cache))
