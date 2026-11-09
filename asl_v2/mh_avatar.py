"""
A realistic rigged human for Blender, built directly from MakeHuman's CC0
data files (no MakeHuman/MPFB code is executed), plus retargeting of asl_v2
motion onto its skeleton.

Data used (all CC0):
  third_party/mpfb2/src/mpfb/data/3dobjs/base.obj          base body mesh (hm08)
  .../data/targets/macrodetails/*.target.gz                 morphs -> young adult male
  .../data/rigs/standard/{rig,weights}.game_engine.json     53-bone rig incl. all fingers
  third_party/mh_assets/{skins,clothes,hair,eyes,eyebrows,eyelashes}  CC0 asset pack

Proxy assets (.mhclo) are fitted the way MakeHuman defines them: each proxy
vertex = w1*V[a] + w2*V[b] + w3*V[c] + offset * per-axis body scale, and
inherit skin weights from those three body vertices.
"""

import gzip
import json
import os

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPFB_DATA = os.path.join(ROOT, "third_party", "mpfb2", "src", "mpfb", "data")
ASSETS = os.path.join(ROOT, "third_party", "mh_assets")


# --------------------------------------------------------------------------
# File parsing (MakeHuman space: decimetres, y up, z towards the viewer)
# --------------------------------------------------------------------------
def read_obj(path, keep_groups=None):
    verts, uvs, faces, groups = [], [], [], {}
    group = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("g "):
                group = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                ids = [t.split("/") for t in line.split()[1:]]
                f = ([int(i[0]) - 1 for i in ids],
                     [int(i[1]) - 1 if len(i) > 1 and i[1] else None for i in ids])
                groups.setdefault(group, []).append(len(faces))
                faces.append(f)
    if keep_groups is not None:
        keep = sorted(i for g in keep_groups for i in groups.get(g, []))
        faces = [faces[i] for i in keep]
    obj_groups = {}
    with open(path) as fh:                       # vertex membership of joint cubes
        group = None
        for line in fh:
            if line.startswith("g "):
                group = line.split(None, 1)[1].strip()
            elif line.startswith("f ") and group and group.startswith("joint-"):
                obj_groups.setdefault(group, set()).update(int(t.split("/")[0]) - 1 for t in line.split()[1:])
    return np.array(verts), np.array(uvs), faces, {k: sorted(v) for k, v in obj_groups.items()}


def apply_target(verts, name, weight=1.0):
    path = os.path.join(MPFB_DATA, "targets", name)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[:1].isdigit():
                i, dx, dy, dz = line.split()
                verts[int(i)] += weight * np.array([float(dx), float(dy), float(dz)])


def read_mhclo(path):
    info = dict(refs=[], weights=[], offsets=[], delete=[], scale={}, material=None, obj=None)
    in_verts = in_delete = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            key = s.split()[0]
            if key in ("x_scale", "y_scale", "z_scale"):
                a, b, ref = s.split()[1:4]
                info["scale"][key[0]] = (int(a), int(b), float(ref))
            elif key == "obj_file":
                info["obj"] = s.split(None, 1)[1]
            elif key == "material":
                info["material"] = s.split(None, 1)[1]
            elif key == "verts":
                in_verts, in_delete = True, False
            elif key == "delete_verts":
                in_verts, in_delete = False, True
            elif in_delete:
                tok = s.split()
                i = 0
                while i < len(tok):
                    if i + 2 < len(tok) and tok[i + 1] == "-":
                        info["delete"].extend(range(int(tok[i]), int(tok[i + 2]) + 1))
                        i += 3
                    else:
                        info["delete"].append(int(tok[i]))
                        i += 1
            elif in_verts and key[0].isdigit():
                p = s.split()
                if len(p) == 1:
                    info["refs"].append((int(p[0]),) * 3)
                    info["weights"].append((1.0, 0.0, 0.0))
                    info["offsets"].append((0.0, 0.0, 0.0))
                else:
                    info["refs"].append(tuple(int(x) for x in p[:3]))
                    info["weights"].append(tuple(float(x) for x in p[3:6]))
                    info["offsets"].append(tuple(float(x) for x in p[6:9]))
            elif in_verts and not key[0].isdigit():
                in_verts = False
    return info


def read_mhmat(path):
    mat = {}
    with open(path) as fh:
        for line in fh:
            p = line.split(None, 1)
            if len(p) == 2 and not line.startswith("#"):
                mat[p[0]] = p[1].strip()
    return mat


def to_blender(v):
    """MakeHuman (dm, y up, z to viewer) -> Blender metres (z up, -y to viewer)."""
    v = np.asarray(v) * 0.1
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


# --------------------------------------------------------------------------
# Blender object construction
# --------------------------------------------------------------------------
def _image(path):
    return bpy.data.images.load(path, check_existing=True)


def make_material(name, tex=None, normal=None, color=(0.8, 0.8, 0.8), rough=0.5, sss=0.0,
                  alpha=False, spec=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    if tex and os.path.exists(tex):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _image(tex)
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
        if alpha:
            nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
    if normal and os.path.exists(normal):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _image(normal)
        t.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.6
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.4, 0.25)
        b.inputs["Subsurface Scale"].default_value = 0.008
    if alpha:
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            pass
    return m


def make_mesh(name, verts_bl, faces, uvs=None, material=None, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts_bl.tolist(), [], [f[0] for f in faces])
    if uvs is not None and len(uvs):
        layer = mesh.uv_layers.new(name="UVMap")
        loop_uvs = [uvs[i] if i is not None else (0, 0) for f in faces for i in f[1]]
        layer.data.foreach_set("uv", np.asarray(loop_uvs, np.float32).ravel())
    for p in mesh.polygons:
        p.use_smooth = smooth
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material:
        obj.data.materials.append(material)
    return obj


def build_armature(rig_def, verts_mh, cubes):
    def joint(spec):
        if spec["strategy"] == "CUBE":
            return verts_mh[cubes[spec["cube_name"]]].mean(0)
        return verts_mh[spec["vertex_indices"]].mean(0)

    arm_data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", arm_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, b in rig_def.items():
        eb = arm_data.edit_bones.new(name)
        eb.head = Vector(to_blender(joint(b["head"])))
        eb.tail = Vector(to_blender(joint(b["tail"])))
        eb.roll = b.get("roll", 0.0)
    for name, b in rig_def.items():
        if b["parent"]:
            arm_data.edit_bones[name].parent = arm_data.edit_bones[b["parent"]]
            arm_data.edit_bones[name].use_connect = b.get("use_connect", False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def skin_to_rig(obj, rig, weights):
    """weights: {bone: [(vertex_index_in_obj, w), ...]}"""
    for bone, vw in weights.items():
        g = obj.vertex_groups.new(name=bone)
        by_w = {}
        for i, w in vw:
            by_w.setdefault(round(w, 4), []).append(int(i))
        for w, idx in by_w.items():
            g.add(idx, w, "REPLACE")
    mod = obj.modifiers.new("armature", "ARMATURE")
    mod.object = rig
    obj.parent = rig


def proxy_weights(info, body_weights):
    """Transfer body skin weights to proxy vertices through their 3 reference verts."""
    out = {}
    refs, ws = np.array(info["refs"]), np.array(info["weights"])
    for bone, vw in body_weights.items():
        table = dict(vw)
        acc = np.zeros(len(refs))
        for k in range(3):
            acc += ws[:, k] * np.array([table.get(int(r), 0.0) for r in refs[:, k]])
        nz = np.flatnonzero(acc > 1e-3)
        if len(nz):
            out[bone] = list(zip(nz, acc[nz]))
    return out


def fit_proxy(mhclo_path, verts_mh):
    info = read_mhclo(mhclo_path)
    base = os.path.dirname(mhclo_path)
    _, uvs, faces, _ = read_obj(os.path.join(base, info["obj"]))
    refs, ws, offs = np.array(info["refs"]), np.array(info["weights"]), np.array(info["offsets"])
    scale = np.ones(3)
    for axis, k in (("x", 0), ("y", 1), ("z", 2)):
        if axis in info["scale"]:
            a, b, ref = info["scale"][axis]
            scale[k] = abs(verts_mh[a, k] - verts_mh[b, k]) / ref
    pos = (ws[:, :, None] * verts_mh[refs]).sum(1) + offs * scale
    mat = read_mhmat(os.path.join(base, info["material"])) if info["material"] else {}
    return info, pos, uvs, faces, mat


# --------------------------------------------------------------------------
# Character
# --------------------------------------------------------------------------
def build_character(skin="young_caucasian_male", clothes="male_casualsuit01", hair="short02",
                    eyebrows="eyebrow001", eyelashes="eyelashes01", eye_color="brown"):
    verts, uvs, faces, cubes = read_obj(os.path.join(MPFB_DATA, "3dobjs", "base.obj"), keep_groups=["body"])
    apply_target(verts, "macrodetails/caucasian-male-young.target.gz")
    apply_target(verts, "macrodetails/universal-male-young-averagemuscle-averageweight.target.gz")

    rig_def = json.load(open(os.path.join(MPFB_DATA, "rigs", "standard", "rig.game_engine.json")))
    weights = json.load(open(os.path.join(MPFB_DATA, "rigs", "standard", "weights.game_engine.json")))["weights"]
    rig = build_armature(rig_def, verts, cubes)

    # proxies first: clothes hide the body vertices underneath them
    hidden = set()
    proxies = []
    for kind, name in (("clothes", clothes), ("hair", hair), ("eyebrows", eyebrows),
                       ("eyelashes", eyelashes), ("eyes", "low-poly")):
        if not name:
            continue
        path = os.path.join(ASSETS, kind, name, f"{name}.mhclo")
        info, pos, puv, pfaces, mat = fit_proxy(path, verts)
        hidden.update(info["delete"])
        proxies.append((kind, name, info, pos, puv, pfaces, mat, os.path.dirname(path)))

    skin_dir = os.path.join(ASSETS, "skins", skin)
    skin_mat = read_mhmat(os.path.join(skin_dir, f"{skin}.mhmat"))
    skin_m = make_material("skin", tex=os.path.join(skin_dir, skin_mat.get("diffuseTexture", "")),
                           color=(0.8, 0.6, 0.5), rough=0.48, sss=0.12, spec=0.35)
    body_faces = [f for f in faces if not any(v in hidden for v in f[0])]
    body = make_mesh("body", to_blender(verts), body_faces, uvs, skin_m)
    skin_to_rig(body, rig, weights)

    objs = {"body": body}
    for kind, name, info, pos, puv, pfaces, mat, d in proxies:
        if kind == "eyes":
            tex = os.path.join(ASSETS, "eyes", "materials", f"{eye_color}_eye.png")
            m = make_material("eyes", tex=tex, rough=0.1, spec=0.8)
        else:
            tex = mat.get("diffuseTexture")
            nrm = mat.get("normalmapTexture")
            m = make_material(kind, tex=os.path.join(d, tex) if tex else None,
                              normal=os.path.join(d, nrm) if nrm else None,
                              rough=0.85 if kind == "clothes" else 0.6,
                              alpha=kind in ("hair", "eyebrows", "eyelashes"),
                              spec=0.3)
        o = make_mesh(kind, to_blender(pos), pfaces, puv, m)
        skin_to_rig(o, rig, proxy_weights(info, weights))
        objs[kind] = o

    sub = body.modifiers.new("subsurf", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1
    return rig, objs


# --------------------------------------------------------------------------
# Retargeting asl_v2 motion (canonical units) onto the rig
# --------------------------------------------------------------------------
P, LH, RH, FACE = 0, 33, 54, 75
ZS = 0.8   # MediaPipe depth is noisier than x/y: damp it a little

CHAINS = {}
for s, off, pose_ids in (("l", LH, (11, 13)), ("r", RH, (12, 14))):
    CHAINS[f"upperarm_{s}"] = (P + pose_ids[0], P + pose_ids[1])
    CHAINS[f"lowerarm_{s}"] = (P + pose_ids[1], off + 0)
    CHAINS[f"hand_{s}"] = (off + 0, off + 9)
    for finger, base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
        for k in range(3):
            CHAINS[f"{finger}_0{k + 1}_{s}"] = (off + base + k, off + base + k + 1)
ORDER = ["upperarm", "lowerarm", "hand", "thumb_01", "thumb_02", "thumb_03",
         "index_01", "index_02", "index_03", "middle_01", "middle_02", "middle_03",
         "ring_01", "ring_02", "ring_03", "pinky_01", "pinky_02", "pinky_03"]


def to_dir(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2] * ZS, -p[..., 1]], -1)


def solve_arm(S, E_obs, T_obs, a, b):
    """Two-bone IK in canonical units (x right, y down, z towards camera < 0).

    MediaPipe's monocular depth for elbows/wrists is unreliable, so depth is
    re-derived from 2D bone projections and the avatar's real bone lengths
    (forearms point towards the viewer), then hands are kept in front of the
    torso/face so contact signs touch instead of intersecting. The elbow
    bends towards the observed elbow."""
    S = np.array([S[0], S[1], 0.0])
    dxy1 = np.linalg.norm(E_obs[:2] - S[:2])
    E0 = np.array([E_obs[0], E_obs[1], -0.5 * np.sqrt(max(a * a - dxy1 * dxy1, 0.0))])
    dxy2 = np.linalg.norm(T_obs[:2] - E_obs[:2])
    z_t = E0[2] - np.sqrt(max(b * b - dxy2 * dxy2, 0.0))
    x, y = T_obs[0], T_obs[1]
    if abs(x) < 0.62 and -0.25 < y < 1.75:
        z_t = min(z_t, -0.40)               # in front of chest / belly
    elif abs(x) < 0.55 and y <= -0.25:
        z_t = min(z_t, -0.50)               # in front of the face
    T = np.array([x, y, z_t])
    u = T - S
    D = np.clip(np.linalg.norm(u), abs(a - b) + 1e-3, a + b - 1e-3)
    u /= np.linalg.norm(u)
    T = S + u * D
    cos_a = np.clip((a * a + D * D - b * b) / (2 * a * D), -1, 1)
    pole = E0 - S
    v = pole - u * np.dot(pole, u)
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 1.0, 0.0]) - u * u[1]
    v /= np.linalg.norm(v)
    E = S + a * (cos_a * u + np.sqrt(1 - cos_a ** 2) * v)
    return E, T


def canon_dir(v):
    v = np.asarray(v, float)
    return np.array([v[0], v[2], -v[1]])


def _frame(y, n):
    y = y / np.linalg.norm(y)
    n = n - y * np.dot(n, y)
    n /= np.linalg.norm(n) + 1e-9
    return np.stack([np.cross(y, n), y, n], 1)        # columns: x, y, z


class Retargeter:
    def __init__(self, rig):
        self.rig = rig
        self.rest = {b.name: np.array(b.matrix_local) for b in rig.data.bones}
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
        sw = np.linalg.norm(self.rest["upperarm_l"][:3, 3] - self.rest["upperarm_r"][:3, 3])
        self.arm_len = {s: (np.linalg.norm(self.rest[f"lowerarm_{s}"][:3, 3] - self.rest[f"upperarm_{s}"][:3, 3]) / sw,
                            np.linalg.norm(self.rest[f"hand_{s}"][:3, 3] - self.rest[f"lowerarm_{s}"][:3, 3]) / sw)
                        for s in "lr"}
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
        # palm normals in hand-bone local space at rest (same formula as at runtime)
        self.palm_local = {}
        for s in "lr":
            h = self.rest[f"hand_{s}"]
            p0, p5, p17 = (self.rest[f"{b}_{s}"][:3, 3] for b in ("hand", "index_01", "pinky_01"))
            n = np.cross(p5 - p0, p17 - p0)
            self.palm_local[s] = h[:3, :3].T @ (n / np.linalg.norm(n))
            self.palm_local[f"fore_{s}"] = self.rest[f"lowerarm_{s}"][:3, :3].T @ (n / np.linalg.norm(n))

    def pose(self, f):
        rig = self.rig
        posed = {}

        def world_rot(name):                 # posed armature-space 3x3 of a bone
            return posed.get(name, None)

        def chain_rest(name):
            par = self.parent[name]
            R_rest = self.rest[name][:3, :3]
            if par is None or par not in posed:
                Rp = np.eye(3) if par is None else self.rest[par][:3, :3]
                return Rp @ (self.rest[par][:3, :3].T if par else np.eye(3)) @ R_rest
            return posed[par] @ self.rest[par][:3, :3].T @ R_rest

        for s in "lr":
            off = LH if s == "l" else RH
            sh, el = (11, 13) if s == "l" else (12, 14)
            E, T = solve_arm(f[P + sh], f[P + el], f[off], *self.arm_len[s])
            arm_dirs = {f"upperarm_{s}": canon_dir(E - np.array([f[P + sh][0], f[P + sh][1], 0.0])),
                        f"lowerarm_{s}": canon_dir(T - E)}
            for part in ORDER:
                name = f"{part}_{s}"
                a, b = CHAINS[name]
                d = arm_dirs[name] if name in arm_dirs else to_dir(f[b]) - to_dir(f[a])
                if np.linalg.norm(d) < 1e-6:
                    continue
                d /= np.linalg.norm(d)
                RA = chain_rest(name)
                y0 = RA[:, 1]
                q = Vector(y0).rotation_difference(Vector(d)).to_matrix()
                R_min = np.array(q) @ RA
                R_final = R_min
                if part in ("hand", "lowerarm"):
                    off = LH if s == "l" else RH
                    h = to_dir(f[off:off + 21])
                    n_t = np.cross(h[5] - h[0], h[17] - h[0])
                    n_t /= np.linalg.norm(n_t) + 1e-9
                    if part == "hand":
                        R_final = _frame(d, n_t) @ _frame(np.array([0, 1.0, 0]), self.palm_local[s]).T
                    else:   # forearm takes half of the hand's twist (pronation)
                        R_full = _frame(d, n_t) @ _frame(np.array([0, 1.0, 0]), self.palm_local[f"fore_{s}"]).T
                        q0 = Matrix(R_min.tolist()).to_quaternion()
                        q1 = Matrix(R_full.tolist()).to_quaternion()
                        R_final = np.array(q0.slerp(q1, 0.5).to_matrix())
                posed[name] = R_final
                basis = RA.T @ R_final
                rig.pose.bones[name].rotation_quaternion = Matrix(basis.tolist()).to_quaternion()

        # head orientation from the face mesh
        face = to_dir(f[FACE:FACE + 468])
        right = face[454] - face[234]
        up = face[10] - face[152]
        Rt = _frame(up, -right)           # y = up, z = -right  (x = forward-ish)
        Rr = _frame(np.array([0, 0, 1.0]), np.array([-1.0, 0, 0]))
        R_face = Rt @ Rr.T
        q = Matrix(R_face.tolist()).to_quaternion()
        q_half = Quaternion().slerp(q, 0.5)
        for name, qq in (("neck_01", q_half), ("head", q_half)):
            R_rest = self.rest[name][:3, :3]
            local = R_rest.T @ np.array(qq.to_matrix()) @ R_rest
            rig.pose.bones[name].rotation_quaternion = Matrix(local.tolist()).to_quaternion()
