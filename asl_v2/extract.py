"""
Keypoint extraction with MediaPipe Holistic (body + both hands + face, tracked
jointly), cached per sign clip.

Why Holistic instead of the Space's separate Pose + Hands models:
  * hands are cropped from the body's wrist ROI, so left/right are assigned
    from the body instead of guessed by the hand model's handedness label
    (which flips on mirrored or fast-moving hands);
  * the 478-point face mesh is tracked too, so eyebrows, eyes and mouth
    (ASL non-manual markers) can be rendered.

Arrays are in source-image pixels (z uses the same scale as x, per MediaPipe):
  pose (T,33,4) x,y,z,visibility | lh, rh (T,21,3) | face (T,478,3);
  undetected parts are NaN.
"""

import os
import subprocess
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(HERE, "cache", "clips")
KP_DIR = os.path.join(HERE, "cache", "keypoints")
VIDEO_BASE = "https://pub-a31536c185d04e47a1aa8bf7c4297b7e.r2.dev"


def clip_path(word, url=None):
    os.makedirs(CLIP_DIR, exist_ok=True)
    path = os.path.join(CLIP_DIR, f"{word}.mp4")
    if not os.path.exists(path):
        # Python's default urllib client is refused by the CDN; curl (its own identity) works.
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", path + ".part", url or f"{VIDEO_BASE}/{word}.mp4"],
                       check=True)
        os.replace(path + ".part", path)
    return path


def _to_array(landmarks, n, w, h, with_vis=False):
    if landmarks is None:
        return np.full((n, 4 if with_vis else 3), np.nan, np.float32)
    rows = []
    for lm in landmarks.landmark:
        row = [lm.x * w, lm.y * h, lm.z * w]
        if with_vis:
            row.append(lm.visibility)
        rows.append(row)
    return np.asarray(rows, np.float32)


def extract(video_path):
    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=2, refine_face_landmarks=True, smooth_landmarks=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose, lh, rh, face = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        r = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks is None:
            continue
        pose.append(_to_array(r.pose_landmarks, 33, w, h, with_vis=True))
        lh.append(_to_array(r.left_hand_landmarks, 21, w, h))
        rh.append(_to_array(r.right_hand_landmarks, 21, w, h))
        face.append(_to_array(r.face_landmarks, 478, w, h))
    cap.release()
    holistic.close()
    return dict(pose=np.stack(pose), lh=np.stack(lh), rh=np.stack(rh),
                face=np.stack(face), size=np.array([w, h]), fps=np.array(fps))


def keypoints(word, url=None):
    os.makedirs(KP_DIR, exist_ok=True)
    cache = os.path.join(KP_DIR, f"{word}.npz")
    if not os.path.exists(cache):
        np.savez_compressed(cache, **extract(clip_path(word, url)))
    return dict(np.load(cache))
