"""
3D renderer for asl_v2 motion, run with Blender's Python module (bpy):

  .venv-blender/bin/python asl_v2/render3d.py motion.npy out.mp4 [--samples 12]

The tracked skeleton is turned into real geometry every frame, with no
external assets:
  * hands/arms: joint positions become vertices of a mesh; Blender's Skin +
    Subdivision Surface modifiers grow smooth, organic flesh around them
    (finger tubes tapering to the tips, a solid palm, sleeved arms);
  * face: MediaPipe's 468-point face mesh drives the canonical face-mesh
    topology directly, so expressions (brows, mouth) come from the signer;
  * head, neck, torso: shaped primitives that follow the tracked head and
    shoulders.
Rendered with Cycles on CPU (sub-surface skin, area lights, denoising).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
P, LH, RH, FACE = 0, 33, 54, 75
ZS = 0.6  # MediaPipe depth is noisy: compress it

FINGER_CHAINS = [(0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
FINGER_R = {  # (base, tip) skin radius per finger, canonical units (1 = shoulder width)
    0: (0.040, 0.026), 1: (0.029, 0.021), 2: (0.030, 0.022), 3: (0.028, 0.020), 4: (0.024, 0.018)}


def to_bl(p):
    """canonical (x right, y down, z away) -> Blender (x right, y away, z up)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2] * ZS, -p[..., 1]], -1)


# --------------------------------------------------------------------------
# Scene setup
# --------------------------------------------------------------------------
def material(name, color, rough=0.5, sss=0.0, sheen=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.03
    if sheen:
        b.inputs["Sheen Weight"].default_value = sheen
    return m


def skin_object(name, n_verts, edges, radii, mat, subdiv=2):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0)] * n_verts, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.modifiers.new("skin", "SKIN")
    obj.modifiers["skin"].use_smooth_shade = True
    for v, r in zip(mesh.skin_vertices[0].data, radii):
        v.radius = (r, r)
    mesh.skin_vertices[0].data[0].use_root = True
    sub = obj.modifiers.new("subsurf", "SUBSURF")
    sub.levels = sub.render_levels = subdiv
    obj.data.materials.append(mat)
    return obj


def hand_object(side, mat):
    # verts 0-20 landmarks, 21 forearm stub, 22 palm centre
    edges = []
    for chain in FINGER_CHAINS:
        edges += list(zip(chain[:-1], chain[1:]))
    edges += [(0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (21, 0)]
    edges += [(22, i) for i in (0, 1, 5, 9, 13, 17)]
    radii = np.zeros(23)
    for f, chain in enumerate(FINGER_CHAINS):
        a, b = FINGER_R[f]
        seg = chain if f else chain[1:]
        for k, i in enumerate(seg):
            radii[i] = a + (b - a) * k / (len(seg) - 1)
    radii[0], radii[21], radii[22] = 0.062, 0.068, 0.072
    radii[1] = 0.050
    for i in (5, 9, 13, 17):
        radii[i] = 0.040
    obj = skin_object(f"hand_{side}", 23, edges, radii, mat)
    obj.data.skin_vertices[0].data[0].use_root = False
    obj.data.skin_vertices[0].data[21].use_root = True
    return obj


def face_object(mat):
    verts, faces = [], []
    with open(os.path.join(HERE, "assets", "canonical_face_model.obj")) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    mesh = bpy.data.meshes.new("face")
    mesh.from_pydata(verts, [], faces)
    for poly in mesh.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new("face", mesh)
    bpy.context.collection.objects.link(obj)
    sub = obj.modifiers.new("subsurf", "SUBSURF")
    sub.levels = sub.render_levels = 1
    obj.data.materials.append(mat)
    return obj


def sphere(name, mat, segments=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=segments // 2)
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(mat)
    return obj


def build_scene(res, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.57, 0.6, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    sc.world = world

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 6.4
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (0, -12, -0.35)
    cam.rotation_euler = (np.pi / 2, 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam

    def area(name, loc, energy, size, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, "AREA")
        d.energy, d.size, d.color = energy, size, color
        o = bpy.data.objects.new(name, d)
        o.location = loc
        o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(o)

    area("key", (-3.5, -5, 3), 700, 4, (1.0, 0.96, 0.9))
    area("fill", (4, -5, 0.5), 350, 5, (0.9, 0.95, 1.0))
    area("rim", (1.5, 4, 3), 600, 3)

    backdrop_mat = material("backdrop", (0.78, 0.79, 0.81), rough=0.9)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 1.6, 0), rotation=(np.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(backdrop_mat)

    skin = material("skin", (0.58, 0.38, 0.28), rough=0.45, sss=0.2)
    shirt = material("shirt", (0.05, 0.08, 0.16), rough=0.85, sheen=0.4)
    hair = material("hair", (0.06, 0.035, 0.02), rough=0.55)
    eye_white = material("eye_white", (0.85, 0.85, 0.82), rough=0.15)
    iris = material("iris", (0.08, 0.05, 0.03), rough=0.1)
    mouth = material("mouth", (0.18, 0.04, 0.05), rough=0.6)

    objs = dict(
        hand_L=hand_object("L", skin), hand_R=hand_object("R", skin),
        arm_L=skin_object("arm_L", 3, [(0, 1), (1, 2)], [0.125, 0.105, 0.085], shirt),
        arm_R=skin_object("arm_R", 3, [(0, 1), (1, 2)], [0.125, 0.105, 0.085], shirt),
        # neck base, L shoulder, R shoulder, chest, waist, hips
        torso=skin_object("torso", 6, [(0, 3), (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)],
                          [0.16, 0.125, 0.125, 0.29, 0.26, 0.28], shirt, subdiv=2),
        neck=skin_object("neck", 3, [(0, 1), (1, 2)], [0.13, 0.115, 0.11], skin),
        face=face_object(skin),
        skull=sphere("skull", skin), hair=sphere("hair", hair),
        eye_L=sphere("eye_L", eye_white, 24), eye_R=sphere("eye_R", eye_white, 24),
        iris_L=sphere("iris_L", iris, 16), iris_R=sphere("iris_R", iris, 16),
        mouth=sphere("mouth", mouth, 24),
    )
    return objs


# --------------------------------------------------------------------------
# Per-frame pose
# --------------------------------------------------------------------------
def set_verts(obj, pts):
    obj.data.vertices.foreach_set("co", np.asarray(pts, np.float32).ravel())
    obj.data.update()


def place(obj, center, axes, radii):
    """Ellipsoid primitive at `center` with orthonormal `axes` (3x3 columns) and radii."""
    m = Matrix(np.asarray(axes).tolist()).to_4x4() @ Matrix.Diagonal((*radii, 1))
    m.translation = Vector(center)
    obj.matrix_world = m


def pose_frame(objs, f):
    f = f.copy()
    # keep hands in front of the torso and the face, elbows not inside the body
    for off in (LH, RH):
        f[off:off + 21, 2] = np.minimum(f[off:off + 21, 2], -0.32 + (f[off:off + 21, 2] - f[off, 2]))
        f[off:off + 21, 2] = np.minimum(f[off:off + 21, 2], -0.25)
    for e in (13, 14, 15, 16):
        f[P + e, 2] = np.clip(f[P + e, 2], -1.2, 0.0)
    f[P + 15, 2] = min(f[P + 15, 2], f[LH, 2])
    f[P + 16, 2] = min(f[P + 16, 2], f[RH, 2])

    ls, rs = f[P + 11].copy(), f[P + 12].copy()
    ls[2] = rs[2] = 0.0
    mid = (ls + rs) / 2
    face = f[FACE:FACE + 468].copy()
    face_c = face.mean(0)
    face[:, 2] -= face_c[2]                      # face depth relative to its centre
    face[:, 2] += -0.10

    for side, (s, e, w, off) in dict(L=(ls, 13, 15, LH), R=(rs, 14, 16, RH)).items():
        el, wr = f[P + e], f[off]
        hand = f[off:off + 21]
        fore = wr - el
        sleeve_end = wr - 0.10 * fore / (np.linalg.norm(fore) + 1e-9)
        set_verts(objs[f"arm_{side}"], to_bl([s, el, sleeve_end]))
        stub = wr - 0.14 * fore / (np.linalg.norm(fore) + 1e-9)
        palm_c = hand[[0, 5, 9, 13, 17]].mean(0) * 0.85 + hand[0] * 0.15
        set_verts(objs[f"hand_{side}"], to_bl(np.vstack([hand, stub, palm_c])))

    chin = face[152]
    set_verts(objs["torso"], to_bl([mid + [0, -0.02, 0.02], ls + [-0.02, 0.02, 0.02], rs + [0.02, 0.02, 0.02],
                                    mid + [0, 0.40, 0.04], mid + [0, 1.05, 0.04], mid + [0, 1.75, 0.04]]))

    set_verts(objs["face"], to_bl(face))

    fb = to_bl(face)
    right = fb[454] - fb[234]
    width = np.linalg.norm(right)
    right /= width
    up = fb[10] - fb[152]
    height = np.linalg.norm(up)
    up = up - right * np.dot(up, right)
    up /= np.linalg.norm(up)
    fwd = np.cross(up, right)                    # points into the screen (+y)
    axes = np.stack([right, fwd, up], 1)
    centre = (fb[234] + fb[454]) / 2 * 0.5 + (fb[10] + fb[152]) / 2 * 0.5
    neck_top = centre + fwd * width * 0.30 - up * height * 0.25
    set_verts(objs["neck"], np.stack([to_bl(mid + [0, 0.02, 0.03]), (to_bl(mid) + neck_top) / 2, neck_top]))
    place(objs["skull"], centre + fwd * width * 0.30 + up * height * 0.06, axes,
          (width * 0.49, width * 0.56, height * 0.55))
    place(objs["hair"], centre + fwd * width * 0.38 + up * height * 0.13, axes,
          (width * 0.53, width * 0.56, height * 0.53))
    place(objs["mouth"], (fb[13] + fb[14]) / 2 + fwd * width * 0.12, axes,
          (width * 0.20, width * 0.10, height * 0.10))
    for side, (iris_i, corners) in dict(L=(473, (263, 362)), R=(468, (33, 133))).items():
        eye_w = np.linalg.norm(fb[corners[0]] - fb[corners[1]])
        ec = (fb[corners[0]] + fb[corners[1]]) / 2
        place(objs[f"eye_{side}"], ec + fwd * eye_w * 0.35, axes, (eye_w * 0.55,) * 3)
        place(objs[f"iris_{side}"], to_bl(f[FACE + iris_i]) * [1, 0, 1] + [0, ec[1] - eye_w * 0.12, 0],
              axes, (eye_w * 0.22, eye_w * 0.08, eye_w * 0.22))


def main():
    argv = sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("output")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--frames", help="comma-separated frame indices: render stills only")
    ap.add_argument("--view", help="debug camera: x,z,ortho_scale")
    args = ap.parse_args(argv)
    seq = np.load(args.motion)
    res = tuple(int(v) for v in args.res.split("x"))
    objs = build_scene(res, args.samples)
    sc = bpy.context.scene
    if args.view:
        x, z, o = map(float, args.view.split(","))
        sc.camera.location = (x, -12, z)
        sc.camera.data.ortho_scale = o

    if args.frames:
        for i in map(int, args.frames.split(",")):
            pose_frame(objs, seq[i])
            sc.render.filepath = f"{os.path.splitext(args.output)[0]}_{i:04d}.png"
            bpy.ops.render.render(write_still=True)
        return

    tmp = tempfile.mkdtemp(prefix="asl3d_")
    for i, f in enumerate(seq):
        pose_frame(objs, f)
        sc.render.filepath = os.path.join(tmp, f"{i:05d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"frame {i + 1}/{len(seq)}", flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                    os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output], check=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
