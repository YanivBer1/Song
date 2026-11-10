"""
Quantitative evaluation of the *final avatar pose* (what gets rendered), run
with Blender's Python (no rendering needed, ~0.1 s/frame):

  .venv-blender/bin/python asl_v2/evaluate.py outputs/v2/john.motion.npy [--valid john.valid.npy] [--json out.json]

Every frame, the MakeHuman character is posed exactly as the renderer does it,
then its joints are read back from the posed armature and the evaluated body
mesh and compared with the target motion.

Metrics (higher is better unless noted; target >= 95%):
  hand_shape_pck   % finger joints whose screen position relative to the knuckle centre is
                   within 10% of hand length (wrist->middle fingertip) of the target
  signing_handshape_pck  PRIMARY handshape score: frames where the hand is signing, the 15
                   visible finger/thumb joints (MCP/IP/tip of thumb, PIP/DIP/tip of fingers),
                   each hand normalised by its own size, within 0.25 of the hand's RMS radius
  hand_placement_pck  % frames*hands whose knuckle+fingertip centroid is within 0.10 shoulder
                   widths of the signer's (mapped onto the avatar's body)
  wrist_pck        % frames*hands with the wrist within 0.10 shoulder widths of its
                   body-relative target (signer landmarks warped onto the avatar)
  elbow_pck        % frames*hands with the upper-arm screen direction within 15 deg
                   (avatar arm proportions differ from the signers', so position can't match)
  joint_limits_ok  % frames*hands where every avatar finger bone's bend relative to its
                   parent (from the avatar's rest pose) is inside anatomical range of motion
                   (MCP<=95, PIP<=110, DIP<=90; thumb CMC/MCP<=70, IP<=85)
  no_body_penetration  % frames where no finger joint is inside the body mesh
  no_hand_penetration  % frames where the hands' finger capsules don't intersect
  smooth_frames    % frames without a velocity spike (> 4x the clip's median joint speed
                   change) - catches pops and jumps
  coverage         % frames where both hands exist (always rendered)
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mh_avatar  # noqa: E402

LMAP = {}  # landmark index -> (bone suffix, "head"/"tail")
LMAP[0] = ("hand", "head")
for f, base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
    LMAP[base] = (f"{f}_01", "head")
    LMAP[base + 1] = (f"{f}_02", "head")
    LMAP[base + 2] = (f"{f}_03", "head")
    LMAP[base + 3] = (f"{f}_03", "tail")
P, LH, RH = 0, 33, 54
LIMITS = {"mcp": 95, "pip": 110, "dip": 90, "thumb_mcp": 70, "thumb_ip": 85}


def _ang(u, v):
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def flexions(h):
    out = {}
    for f, (a, b, c, d) in {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
                            "pinky": (17, 18, 19, 20)}.items():
        out[f"{f}_mcp"] = _ang(h[a] - h[0], h[b] - h[a])
        out[f"{f}_pip"] = _ang(h[b] - h[a], h[c] - h[b])
        out[f"{f}_dip"] = _ang(h[c] - h[b], h[d] - h[c])
    out["thumb_mcp"] = _ang(h[2] - h[1], h[3] - h[2])
    out["thumb_ip"] = _ang(h[3] - h[2], h[4] - h[3])
    return out


BONE_LIMITS = {"01": 95.0, "02": 110.0, "03": 90.0}          # MCP, PIP, DIP flexion (deg)
THUMB_LIMITS = {"01": 70.0, "02": 70.0, "03": 85.0}          # CMC swing, MCP, IP


def bone_angles_ok(rig, s, rest_angle):
    """Anatomical validity on the rig itself: each finger bone's bend relative to its
    parent, measured from the avatar's own rest pose."""
    pb = rig.pose.bones
    for fn in ("thumb", "index", "middle", "ring", "pinky"):
        for k in ("01", "02", "03"):
            name = f"{fn}_{k}_{s}"
            par = pb[name].parent
            yc = np.array(pb[name].matrix.col[1][:3])
            yp = np.array(par.matrix.col[1][:3])
            ang = np.degrees(np.arccos(np.clip(yc @ yp, -1, 1))) - rest_angle[name]
            if ang > (THUMB_LIMITS if fn == "thumb" else BONE_LIMITS)[k]:
                return False
    return True


def limits_ok(h):
    for k, v in flexions(h).items():
        lim = LIMITS["thumb_mcp" if k == "thumb_mcp" else "thumb_ip" if k == "thumb_ip" else k.split("_")[1]]
        if v > lim:
            return False
    return True


def seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments p1q1 and p2q2 (3D)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    s = np.clip((b * t - c) / a, 0, 1) if a > 1e-12 else 0.0
    return np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t))


SEGS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (0, 17), (5, 17)]


def evaluate(seq, valid=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig, objs = mh_avatar.build_character()
    rt = mh_avatar.Retargeter(rig, objs)
    rt.use_hinge = os.environ.get("ASL_HINGE", "0") == "1"
    cal = rt.cal
    sw = cal["mp_shoulder_width_m"]
    mid = np.array([cal["mp_mid_world_xz"][0], 0.0, cal["mp_mid_world_xz"][1]])
    # Collision surface = what is actually visible around the torso: exposed skin of
    # head/neck/torso plus the clothes (MakeHuman deletes the skin under clothing), with
    # arms, sleeves and hands excluded (a hand's own skin must not count as "body").
    torso_names = ("pelvis", "spine", "neck", "head", "clavicle", "thigh")
    collide = []
    for key in ("body", "clothes"):
        o = objs[key]
        tg = {g.index for g in o.vertex_groups if g.name.split("_")[0] in torso_names}
        vmask = np.zeros(len(o.data.vertices), bool)
        for v in o.data.vertices:
            if v.groups:
                g = max(v.groups, key=lambda x: x.weight)
                vmask[v.index] = g.group in tg and g.weight > 0.5
        collide.append((o, [p.index for p in o.data.polygons if all(vmask[i] for i in p.vertices)]))
    finger_r = 0.009  # metres, finger capsule radius for hand-hand test
    rest_angle = {}
    for b in rig.data.bones:
        if b.parent is not None and b.name.split("_")[0] in ("thumb", "index", "middle", "ring", "pinky"):
            yc, yp = np.array(b.matrix_local.col[1][:3]), np.array(b.parent.matrix_local.col[1][:3])
            rest_angle[b.name] = np.degrees(np.arccos(np.clip(yc @ yp, -1, 1)))
    T = len(seq)
    rt.plan_depth(seq)
    rec = {k: [] for k in ("shape", "wrist", "elbow", "limits", "body", "hands")}
    joints_all = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for t in range(T):
        f = seq[t]
        rt.pose(f)
        bpy.context.view_layer.update()
        pb = rig.pose.bones
        av = {}
        for s, off in (("l", LH), ("r", RH)):
            pts = np.zeros((21, 3))
            for i, (bone, end) in LMAP.items():
                b = pb[f"{bone}_{s}"]
                pts[i] = np.array(rig.matrix_world @ (b.head if end == "head" else b.tail))
            av[s] = pts
            el = np.array(rig.matrix_world @ pb[f"lowerarm_{s}"].head)
            # screen-space canonical coords of the avatar: x right, y down, unit = shoulder width
            scr = np.stack([(pts[:, 0] - mid[0]) / sw, -(pts[:, 2] - mid[2]) / sw], 1)
            tgt = f[off:off + 21, :2]
            ok = valid is None or valid[t, 0 if s == "l" else 1]
            if ok:
                hand_len = np.linalg.norm(tgt[12] - tgt[0]) + 1e-9
                knuckles = [5, 9, 13, 17]
                c_av, c_tg = scr[knuckles].mean(0), tgt[knuckles].mean(0)
                err = np.linalg.norm((scr - c_av) - (tgt - c_tg), axis=1)[1:] / hand_len
                rec["shape"].extend((err < 0.10).tolist())
                # size-independent handshape: each hand normalised by its own RMS radius
                ca, ct = scr[1:] - c_av, tgt[1:] - c_tg
                ca /= np.sqrt((ca ** 2).sum(1).mean()) + 1e-9
                ct /= np.sqrt((ct ** 2).sum(1).mean()) + 1e-9
                rec.setdefault("shape_n", []).extend((np.linalg.norm(ca - ct, axis=1) < 0.25).tolist())
                if tgt[0, 1] < 1.15:      # hand is signing (not resting low at the side)
                    vis = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]) - 1
                    rec.setdefault("shape_sign", []).extend((np.linalg.norm(ca[vis] - ct[vis], axis=1) < 0.25).tolist())
                L = rt.last[s]
                u = scr[9] - scr[0]
                wrist_mp = scr[0] + rt.wrist_off[s] * u / (np.linalg.norm(u) + 1e-9) * np.linalg.norm(L["hand_dir2"])
                rec["wrist"].append(np.linalg.norm(wrist_mp - L["wrist_mp_target"]) < 0.10)
                eff = [5, 9, 13, 17, 8, 12]
                hand_tgt = np.mean([rt.warp(f[off + i], f) for i in eff], 0)
                rec.setdefault("hand_place", []).append(np.linalg.norm(scr[eff].mean(0) - hand_tgt) < 0.10)
                e_scr = np.array([(el[0] - mid[0]) / sw, -(el[2] - mid[2]) / sw])
                ua, ut = e_scr - L["shoulder"], L["elbow_target"] - L["shoulder"]
                cosv = ua @ ut / (np.linalg.norm(ua) * np.linalg.norm(ut) + 1e-9)
                rec["elbow"].append(np.degrees(np.arccos(np.clip(cosv, -1, 1))) < 15)
            rec["limits"].append(bone_angles_ok(rig, s, rest_angle))
        joints_all.append(np.concatenate([av["l"], av["r"]]))
        # body penetration: finger joints inside the evaluated body mesh
        verts, polys = [], []
        for o, pidx in collide:
            ev = o.evaluated_get(depsgraph)
            mesh = ev.to_mesh()
            base = len(verts)
            verts.extend(ev.matrix_world @ v.co for v in mesh.vertices)
            polys.extend([base + i for i in mesh.polygons[k].vertices] for k in pidx)
            ev.to_mesh_clear()
        bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False)
        inside = False
        for s in "lr":
            for p in av[s][[4, 8, 12, 16, 20, 6, 10, 14, 18]]:
                loc, normal, _, dist = bvh.find_nearest(p)
                # finger joint centre more than 5 mm behind the surface => finger sunk into the body
                if loc is not None and dist < 0.10 and np.dot(np.array(p) - np.array(loc), np.array(normal)) < -0.005:
                    inside = True
        rec["body"].append(not inside)
        # hand-hand penetration via finger capsules
        hit = False
        for a, b in SEGS:
            for c, d in SEGS:
                if seg_dist(av["l"][a], av["l"][b], av["r"][c], av["r"][d]) < 2 * finger_r * 0.6:
                    hit = True
                    break
            if hit:
                break
        rec["hands"].append(not hit)

    J = np.array(joints_all)                       # (T, 42, 3) metres
    v = np.linalg.norm(np.diff(J, axis=0), axis=2)  # per-joint speed
    dv = np.abs(np.diff(v, axis=0)).max(1)
    med = np.median(dv) + 1e-6
    spikes = dv > 4 * med + 0.004
    print("spike frames:", (np.flatnonzero(spikes) + 1).tolist(), file=sys.stderr)
    jerk = np.linalg.norm(np.diff(J, 3, axis=0), axis=2).mean()
    pct = lambda x: round(100.0 * float(np.mean(x)), 1) if len(x) else None
    return {
        "frames": T,
        "hand_shape_pck": pct(rec["shape"]),
        "handshape_pck_size_norm": pct(rec.get("shape_n", [])),
        "signing_handshape_pck": pct(rec.get("shape_sign", [])),
        "hand_placement_pck": pct(rec.get("hand_place", [])),
        "wrist_pck": pct(rec["wrist"]),
        "elbow_pck": pct(rec["elbow"]),
        "joint_limits_ok": pct(rec["limits"]),
        "no_body_penetration": pct(rec["body"]),
        "no_hand_penetration": pct(rec["hands"]),
        "smooth_frames": pct(~spikes),
        "coverage": 100.0,
        "mean_jerk_mm_per_frame3": round(float(jerk) * 1000, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", nargs="+")
    ap.add_argument("--valid")
    ap.add_argument("--json")
    a = ap.parse_args()
    results = {}
    for path in a.motion:
        seq = np.load(path)
        valid = np.load(a.valid) if a.valid else None
        results[os.path.basename(path)] = res = evaluate(seq, valid)
        print(os.path.basename(path), json.dumps(res), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results if len(results) > 1 else res, fh, indent=1)


if __name__ == "__main__":
    main()
