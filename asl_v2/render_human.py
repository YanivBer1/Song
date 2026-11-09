"""
Photoreal-style renderer: a rigged MakeHuman (CC0) character driven by asl_v2
motion, rendered with Cycles on CPU.

  .venv-blender/bin/python asl_v2/render_human.py motion.npy out.mp4 [--samples 16] [--res 960x540]
  ... --frames 0,120     # stills only (out_0000.png, ...)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mh_avatar  # noqa: E402


def setup_scene(rig, res, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 16
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast"

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.52, 0.56, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    sc.world = world

    bones = rig.data.bones
    head = np.array(bones["head"].head_local)
    pelvis = np.array(bones["pelvis"].head_local)
    top = head[2] + 0.30
    bottom = pelvis[2] - 0.22
    centre_z = (top + bottom) / 2
    height = top - bottom

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50          # interpreter-video style: ~2 m away, natural perspective on the hands
    cam = bpy.data.objects.new("cam", cam_data)
    sens_h = cam_data.sensor_width * res[1] / res[0]
    dist = height * cam_data.lens / sens_h
    cam.location = (0, -dist, centre_z)
    cam.rotation_euler = (np.pi / 2, 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam

    def area(name, loc, energy, size, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.energy, d.size, d.color = energy, size, color
        o = bpy.data.objects.new(name, d)
        o.location = loc
        target = Vector((0, 0, centre_z))
        o.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(o)

    area("key", (-1.6, -2.4, centre_z + 1.2), 260, 1.6, (1.0, 0.95, 0.9))
    area("fill", (1.8, -2.2, centre_z + 0.2), 110, 2.2, (0.9, 0.95, 1.0))
    area("rim", (0.8, 1.5, centre_z + 1.0), 180, 1.0)

    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 1.2, centre_z), rotation=(np.pi / 2, 0, 0))
    m = bpy.data.materials.new("backdrop")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.42, 0.45, 0.5, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
    bpy.context.object.data.materials.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("output")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--frames")
    ap.add_argument("--skin", default="young_caucasian_male")
    ap.add_argument("--clothes", default="male_casualsuit01")
    ap.add_argument("--hair", default="short02")
    ap.add_argument("--closeup", action="store_true", help="frame chest+hands (inspection)")
    ap.add_argument("--faceup", action="store_true", help="frame the face (inspection)")
    ap.add_argument("--audit-view", action="store_true",
                    help="orthographic camera matching the source QA crop (calibrated)")
    ap.add_argument("--batch", help="JSON list of {motion, frames, prefix}: render stills for many motions")
    args = ap.parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    t = time.time()
    rig, objs = mh_avatar.build_character(skin=args.skin, clothes=args.clothes, hair=args.hair)
    print(f"character built in {time.time() - t:.1f}s", flush=True)
    res = tuple(int(v) for v in args.res.split("x"))
    setup_scene(rig, res, args.samples)
    if args.closeup:
        cam = bpy.context.scene.camera
        chest_z = np.array(rig.data.bones["spine_03"].head_local)[2]
        cam.data.lens = 85
        sens_h = cam.data.sensor_width * res[1] / res[0]
        cam.location = (0, -0.75 * cam.data.lens / sens_h, chest_z + 0.02)
    if args.faceup:
        cam = bpy.context.scene.camera
        head_z = np.array(rig.data.bones["head"].head_local)[2]
        sens_h = cam.data.sensor_width * res[1] / res[0]
        cam.location = (0, -0.30 * cam.data.lens / sens_h, head_z + 0.035)
    if args.audit_view:
        cal = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "avatar_calib.json")))
        sw = cal["mp_shoulder_width_m"]
        mx, mz = cal["mp_mid_world_xz"]
        cam = bpy.context.scene.camera
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = 2.16 * sw * res[0] / res[1]
        cam.location = (mx, -5.0, mz - 0.30 * sw)
    retarget = mh_avatar.Retargeter(rig, objs)
    sc = bpy.context.scene
    if args.batch:
        for job in json.load(open(args.batch)):
            seq = np.load(job["motion"])
            feats = [mh_avatar.face_features(f[mh_avatar.FACE:mh_avatar.FACE + 468, :2]) for f in seq]
            neutral = mh_avatar.neutral_face(feats)
            retarget.plan_depth(seq)
            for i in range(len(seq)):              # pose every frame: keeps temporal state honest
                retarget.pose(seq[i])
                if i in job["frames"]:
                    retarget.set_expression(seq[i], neutral, objs)
                    bpy.context.view_layer.update()
                    sc.render.filepath = f"{job['prefix']}_{i:04d}.png"
                    bpy.ops.render.render(write_still=True)
        return
    seq = np.load(args.motion)
    feats = [mh_avatar.face_features(f[mh_avatar.FACE:mh_avatar.FACE + 468, :2]) for f in seq]
    neutral = mh_avatar.neutral_face(feats)

    retarget.plan_depth(seq)

    def pose(f):
        retarget.pose(f)
        retarget.set_expression(f, neutral, objs)

    if args.frames:
        for i in map(int, args.frames.split(",")):
            retarget.frame_t = i - 1               # index the depth plan by the real frame
            pose(seq[i])
            bpy.context.view_layer.update()
            sc.render.filepath = f"{os.path.splitext(args.output)[0]}_{i:04d}.png"
            bpy.ops.render.render(write_still=True)
        return

    tmp = tempfile.mkdtemp(prefix="aslhuman_")
    t = time.time()
    for i, f in enumerate(seq):
        pose(f)
        bpy.context.view_layer.update()
        sc.render.filepath = os.path.join(tmp, f"{i:05d}.png")
        bpy.ops.render.render(write_still=True)
        if i % 10 == 0:
            el = time.time() - t
            print(f"frame {i + 1}/{len(seq)}  {el / (i + 1):.1f}s/frame", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                    os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
