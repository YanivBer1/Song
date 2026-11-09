"""
Calibrate the avatar in the *same measurement convention* as the source signers:
render the MakeHuman character at rest (orthographic, front) and run the same
MediaPipe Holistic tracker on that render. The result gives the avatar's own
MediaPipe shoulder midpoint/width, face landmarks (chin, mouth, nose, forehead,
cheeks) and wrist-landmark offsets. Retargeting then maps signer landmarks onto
the avatar's matching landmarks, so contact signs land on the avatar's body.

  .venv-blender/bin/python asl_v2/calibrate.py render   # -> cache/calib/avatar_rest.png + camera.json
  .venv/bin/python -m asl_v2.calibrate measure          # -> asl_v2/models/avatar_calib.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAL_DIR = os.path.join(ROOT, "cache", "calib")
OUT = os.path.join(HERE, "models", "avatar_calib.json")
RES = 1024


def render():
    import bpy
    import numpy as np
    sys.path.insert(0, HERE)
    import mh_avatar
    from render_human import setup_scene

    os.makedirs(CAL_DIR, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig, objs = mh_avatar.build_character()
    setup_scene(rig, (RES, RES), 24)
    bones = rig.data.bones
    head = np.array(bones["head"].head_local)
    top, bottom = head[2] + 0.30, np.array(bones["pelvis"].head_local)[2] - 0.35
    cam = bpy.context.scene.camera
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = top - bottom
    cam.location = (0, -5, (top + bottom) / 2)
    sc = bpy.context.scene
    sc.render.filepath = os.path.join(CAL_DIR, "avatar_rest.png")
    bpy.ops.render.render(write_still=True)
    rest = {b.name: list(map(float, b.head_local)) for b in bones}
    tails = {b.name: list(map(float, b.tail_local)) for b in bones}
    with open(os.path.join(CAL_DIR, "camera.json"), "w") as fh:
        json.dump(dict(center_x=0.0, center_z=float((top + bottom) / 2), scale=float(top - bottom), res=RES,
                       bone_heads=rest, bone_tails=tails), fh)


def measure():
    import cv2
    import mediapipe as mp
    import numpy as np

    cam = json.load(open(os.path.join(CAL_DIR, "camera.json")))
    img = cv2.imread(os.path.join(CAL_DIR, "avatar_rest.png"))
    h, w = img.shape[:2]
    r = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2,
                                       refine_face_landmarks=True).process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert r.pose_landmarks and r.face_landmarks, "MediaPipe did not find the avatar"

    def world(lm):   # image -> Blender world (x, z) through the orthographic camera
        x = cam["center_x"] + (lm.x - 0.5) * cam["scale"]
        z = cam["center_z"] - (lm.y - 0.5) * cam["scale"]
        return np.array([x, z])

    pose = np.array([world(l) for l in r.pose_landmarks.landmark])
    face = np.array([world(l) for l in r.face_landmarks.landmark])
    mid = (pose[11] + pose[12]) / 2
    sw = float(np.linalg.norm(pose[11] - pose[12]))

    def canon(p):   # world (x, z) -> canonical (x right, y down, units of MP shoulder width)
        return [float((p[0] - mid[0]) / sw), float(-(p[1] - mid[1]) / sw)]

    hands = {}
    for s, lms in (("L", r.left_hand_landmarks), ("R", r.right_hand_landmarks)):
        if lms:
            hands[s] = [canon(world(l)) for l in lms.landmark]
    bh = cam["bone_heads"]
    out = dict(
        mp_mid_world_xz=mid.tolist(), mp_shoulder_width_m=sw,
        face=[canon(p) for p in face[:468]],
        pose={i: canon(pose[i]) for i in (0, 11, 12, 13, 14, 15, 16, 23, 24)},
        hands=hands,
        bone_heads_canon={k: canon(np.array([v[0], v[2]])) for k, v in bh.items()
                          if k.split("_")[0] in ("upperarm", "lowerarm", "hand", "middle", "index", "pinky", "head", "neck")},
    )
    # The avatar's own relaxed hand (MakeHuman rest pose) as the motion's rest-hand template:
    # 21 landmarks from bone heads/tails, canonical axes (x right, y down, z away), wrist at the
    # origin, hand axis (wrist->middle knuckle) rotated onto +y, units of MP shoulder width.
    bt = cam["bone_tails"]
    lmap = {0: ("hand", "h")}
    for fn, base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
        lmap.update({base: (f"{fn}_01", "h"), base + 1: (f"{fn}_02", "h"), base + 2: (f"{fn}_03", "h"),
                     base + 3: (f"{fn}_03", "t")})
    for s in "lr":
        pts = np.array([(bh if e == "h" else bt)[f"{b}_{s}"] for b, e in (lmap[i] for i in range(21))])
        c = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], 1)          # Blender -> canonical axes
        c = (c - c[0]) / sw
        a = c[9] / np.linalg.norm(c[9])                               # minimal 3D rotation: a -> +y
        y = np.array([0.0, 1.0, 0.0])
        v, cs = np.cross(a, y), float(a @ y)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K / (1 + cs)
        c = c @ R.T
        out[f"rest_hand_{s.upper()}"] = c.tolist()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    up = np.linalg.norm(np.subtract(out["bone_heads_canon"]["lowerarm_l"], out["bone_heads_canon"]["upperarm_l"]))
    fo = np.linalg.norm(np.subtract(out["bone_heads_canon"]["hand_l"], out["bone_heads_canon"]["lowerarm_l"]))
    print(f"avatar MP shoulder width {sw:.3f} m; upper arm {up:.3f}, forearm {fo:.3f} (MP shoulder units); "
          f"MP shoulders at {out['pose'][11]}, {out['pose'][12]}; bone shoulders "
          f"{out['bone_heads_canon']['upperarm_l']}, {out['bone_heads_canon']['upperarm_r']}; hands found {list(hands)}")


if __name__ == "__main__":
    {"render": render, "measure": measure}[sys.argv[1]]()
