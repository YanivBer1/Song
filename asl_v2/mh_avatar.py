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


def tint(mat, color, factor):
    """Multiply a material's base-colour texture by `color` (strength `factor`)."""
    nt = mat.node_tree
    tex = [n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name != "Non-Color"]
    if not tex:
        return
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = factor
    mix.inputs["B"].default_value = (*color, 1)
    nt.links.new(tex[0].outputs["Color"], mix.inputs["A"])
    nt.links.new(mix.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Base Color"])


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
                    eyebrows="eyebrow001", eyelashes="eyelashes01", eye_color="brownlight"):
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
                       ("eyelashes", eyelashes), ("eyes", "low-poly"), ("teeth", "teeth_base"),
                       ("tongue", "tongue01")):
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
    # Legibility: the MakeHuman skin renders very pale. A warmer, deeper skin tone against a
    # dark top (what interpreters wear) makes the hands and their finger shadows read clearly.
    tint(skin_m, (0.74, 0.55, 0.45), 0.85)
    body_faces = [f for f in faces if not any(v in hidden for v in f[0])]
    body = make_mesh("body", to_blender(verts), body_faces, uvs, skin_m)
    skin_to_rig(body, rig, weights)

    objs = {"body": body}
    for kind, name, info, pos, puv, pfaces, mat, d in proxies:
        if kind == "eyes":
            tex = os.path.join(ASSETS, "eyes", "materials", f"{eye_color}_eye.png")
            m = make_material("eyes", tex=tex, rough=0.08, spec=0.8)
            # darken the amber "brownlight" iris to a natural brown (texture -> multiply)
            nt = m.node_tree
            tnode = [n for n in nt.nodes if n.type == "TEX_IMAGE"][0]
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 0.9
            mix.inputs["B"].default_value = (0.45, 0.30, 0.22, 1)
            nt.links.new(tnode.outputs["Color"], mix.inputs["A"])
            nt.links.new(mix.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        else:
            tex = mat.get("diffuseTexture")
            nrm = mat.get("normalmapTexture")
            m = make_material(kind, tex=os.path.join(d, tex) if tex else None,
                              normal=os.path.join(d, nrm) if nrm else None,
                              rough={"clothes": 0.85, "teeth": 0.25, "tongue": 0.35}.get(kind, 0.6),
                              alpha=kind in ("hair", "eyebrows", "eyelashes"),
                              spec=0.3)
        if kind == "clothes":
            tint(m, (0.10, 0.11, 0.15), 0.9)             # dark charcoal-navy outfit
        o = make_mesh(kind, to_blender(pos), pfaces, puv, m)
        skin_to_rig(o, rig, proxy_weights(info, weights))
        objs[kind] = o

    add_expression_keys(body, objs, proxies, len(verts))
    sub = body.modifiers.new("subsurf", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1
    return rig, objs


# --------------------------------------------------------------------------
# Facial expression units (MakeHuman CC0 FACS-style targets) as shape keys
# --------------------------------------------------------------------------
EXPR_UNITS = ["mouth-open", "mouth-corner-puller", "mouth-pursing", "mouth-depression",
              "eyebrows-left-up", "eyebrows-right-up", "eyebrows-left-down", "eyebrows-right-down",
              "eyebrows-left-inner-up", "eyebrows-right-inner-up",
              "eye-left-closure", "eye-right-closure", "eye-left-opened-up", "eye-right-opened-up"]


def _target_delta(name, n, race="caucasian"):
    d = np.zeros((n, 3))
    path = os.path.join(MPFB_DATA, "targets", "expression", "units", race, f"{name}.target.gz")
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[:1].isdigit():
                i, dx, dy, dz = line.split()
                d[int(i)] = (float(dx), float(dy), float(dz))
    return to_blender(d)          # linear map, so it converts offsets too


def add_expression_keys(body, objs, proxies, n):
    body.shape_key_add(name="Basis")
    basis = np.array([v.co for v in body.data.vertices])
    proxy_info = {kind: info for kind, name, info, *_ in proxies}
    for kind in ("eyebrows", "eyelashes", "teeth", "tongue"):
        if kind in objs:
            objs[kind].shape_key_add(name="Basis")
    for unit in EXPR_UNITS:
        delta = _target_delta(unit, n)
        k = body.shape_key_add(name=unit, from_mix=False)
        k.data.foreach_set("co", (basis + delta).astype(np.float32).ravel())
        for kind in ("eyebrows", "eyelashes", "teeth", "tongue"):  # proxies follow the skin via their 3 reference verts
            if kind not in objs:
                continue
            info, o = proxy_info[kind], objs[kind]
            refs, ws = np.array(info["refs"]), np.array(info["weights"])
            pd = (ws[:, :, None] * delta[refs]).sum(1)
            pb = np.array([v.co for v in o.data.vertices])
            pk = o.shape_key_add(name=unit, from_mix=False)
            pk.data.foreach_set("co", (pb + pd).astype(np.float32).ravel())


FACE_FEATS = dict(lip_top=13, lip_bot=14, mouth_l=291, mouth_r=61, top=10, chin=152,
                  brow_l=(334, 296, 293, 300), brow_r=(105, 66, 63, 70),
                  lid_l=(386, 374), lid_r=(159, 145))


def face_features(face2d):
    """Scale-free facial measurements from (468+,2) canonical face landmarks."""
    F = FACE_FEATS
    h = np.linalg.norm(face2d[F["top"]] - face2d[F["chin"]]) + 1e-9
    out = dict(
        open=np.linalg.norm(face2d[F["lip_top"]] - face2d[F["lip_bot"]]) / h,
        width=np.linalg.norm(face2d[F["mouth_l"]] - face2d[F["mouth_r"]]) / h,
    )
    for s in "lr":
        brow = face2d[list(F[f"brow_{s}"])].mean(0)
        lid_top, lid_bot = face2d[F[f"lid_{s}"][0]], face2d[F[f"lid_{s}"][1]]
        out[f"brow_{s}"] = (lid_top[1] - brow[1]) / h
        out[f"eye_{s}"] = np.linalg.norm(lid_top - lid_bot) / h
    return out


def neutral_face(feats):
    """Robust neutral face from a sequence of face_features(): signers keep their mouth
    open much of the time, so the neutral mouth comes from the most-closed frames and the
    neutral eye opening from the more-open frames (blinks excluded)."""
    arr = {k: np.array([f[k] for f in feats]) for k in feats[0]}
    closed = arr["open"] <= np.percentile(arr["open"], 25)
    out = {"open": float(np.percentile(arr["open"], 5)), "width": float(np.median(arr["width"][closed]))}
    for s in "lr":
        out[f"brow_{s}"] = float(np.median(arr[f"brow_{s}"]))
        out[f"eye_{s}"] = float(np.percentile(arr[f"eye_{s}"], 65))
    return out


def expression_weights(feat, neutral):
    c = lambda v: float(np.clip(v, 0.0, 1.0))
    w = {
        "mouth-open": c((feat["open"] - neutral["open"]) / 0.10),
        # +0.15 baseline: MakeHuman's resting mouth reads as a slight frown; interpreters keep a
        # relaxed, pleasant neutral face
        "mouth-corner-puller": c(0.15 + (feat["width"] - neutral["width"]) / 0.07),
        "mouth-pursing": c((neutral["width"] - feat["width"]) / 0.07),
    }
    for s, side in (("l", "left"), ("r", "right")):
        db = feat[f"brow_{s}"] - neutral[f"brow_{s}"]
        de = (feat[f"eye_{s}"] - neutral[f"eye_{s}"]) / (neutral[f"eye_{s}"] + 1e-9)
        w[f"eyebrows-{side}-up"] = c(db / 0.035)
        w[f"eyebrows-{side}-down"] = c(-db / 0.03)
        w[f"eye-{side}-closure"] = c(-de / 0.75)
        w[f"eye-{side}-opened-up"] = c(de / 0.6)
    return w


# --------------------------------------------------------------------------
# Retargeting asl_v2 motion (canonical units) onto the rig
# --------------------------------------------------------------------------
P, LH, RH, FACE = 0, 33, 54, 75
ZS = 1.0   # hand depth now comes from metric 3D hand landmarks: no damping

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


def solve_arm(S, E_obs, T_obs, a, b, z_bias=0.0):
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
    T = np.array([x, y, z_t + z_bias])
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


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = np.clip((b * s + f) / e, 0, 1) if e > 1e-12 else 0.0
    s = np.clip((b * t - c) / a, 0, 1) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


# Hard human range of motion for the rig's finger bones (deg, rotation from rest):
# MCP 95, PIP 110, DIP 90; thumb CMC 70, MCP 70, IP 85
ROM = {"01": 95.0, "02": 110.0, "03": 90.0}
ROM_THUMB = {"01": 70.0, "02": 70.0, "03": 85.0}


FIT_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def _gauss(x, sigma):
    """Gaussian filter along axis 0 with edge padding (numpy only; Blender env has no scipy)."""
    x = np.asarray(x, float)
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.tensordot(k, xp[i:i + 2 * r + 1], axes=(0, 0)) for i in range(len(x))])


def _blocks(mask):
    t, T = 0, len(mask)
    while t < T:
        if not mask[t]:
            t += 1
            continue
        u = t
        while u < T and mask[u]:
            u += 1
        yield t, u
        t = u


def _minjerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


class Retargeter:
    def __init__(self, rig, objs=None):
        self.rig = rig
        self.objs = objs
        self.collide = None
        self.rest = {b.name: np.array(b.matrix_local) for b in rig.data.bones}
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
        sw = np.linalg.norm(self.rest["upperarm_l"][:3, 3] - self.rest["upperarm_r"][:3, 3])
        self.arm_len = {s: (np.linalg.norm(self.rest[f"lowerarm_{s}"][:3, 3] - self.rest[f"upperarm_{s}"][:3, 3]) / sw,
                            np.linalg.norm(self.rest[f"hand_{s}"][:3, 3] - self.rest[f"lowerarm_{s}"][:3, 3]) / sw)
                        for s in "lr"}
        self.cal = None
        cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "avatar_calib.json")
        if os.path.exists(cal_path):
            self._load_calibration(json.load(open(cal_path)))
        self.finger_bones = {f"{fn}_0{k}_{s}" for fn in ("thumb", "index", "middle", "ring", "pinky")
                             for k in (1, 2, 3) for s in "lr"}
        self.finger_len = {s: sum(rig.data.bones[f"{fn}_0{k}_{s}"].length for fn in ("index", "middle", "ring", "pinky")
                                  for k in (1, 2, 3)) for s in "lr"}
        for pb in rig.pose.bones:
            pb.rotation_mode = "QUATERNION"
        if objs is not None:
            self._build_collision(objs)
        self._calibrate_hinges()
        self.hinge_local = {}
        for s in "lr":                       # elbow hinge axis in the upper arm's rest frame
            Ru = self.rest[f"upperarm_{s}"][:3, :3]
            h = np.cross(Ru[:, 1], np.array([0.0, -1.0, 0.0]))
            self.hinge_local[s] = Ru.T @ (h / np.linalg.norm(h))
        # palm normals in hand-bone local space at rest (same formula as at runtime)
        self.palm_local = {}
        for s in "lr":
            h = self.rest[f"hand_{s}"]
            p0, p5, p17 = (self.rest[f"{b}_{s}"][:3, 3] for b in ("hand", "index_01", "pinky_01"))
            n = np.cross(p5 - p0, p17 - p0)
            self.palm_local[s] = h[:3, :3].T @ (n / np.linalg.norm(n))
            self.palm_local[f"fore_{s}"] = self.rest[f"lowerarm_{s}"][:3, :3].T @ (n / np.linalg.norm(n))

    # ---------------- hinge joints (PIP/DIP, thumb MCP/IP) --------------------------------------
    def _calibrate_hinges(self):
        """Find each hinge bone's flexion axis sign on the rig: flexing always brings the
        fingertip closer to the wrist (local X for finger PIP/DIP, local Z for thumb MCP/IP)."""
        self.hinge = {}
        rig = self.rig
        for s in "lr":
            wrist = np.array(rig.data.bones[f"hand_{s}"].head_local)
            for fn in ("thumb", "index", "middle", "ring", "pinky"):
                tip_bone = rig.pose.bones[f"{fn}_03_{s}"]
                for k in ("02", "03"):
                    name = f"{fn}_{k}_{s}"
                    axis = Vector((0, 0, 1)) if fn == "thumb" else Vector((1, 0, 0))
                    pb = rig.pose.bones[name]
                    pb.rotation_mode = "QUATERNION"
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    bpy.context.view_layer.update()
                    d0 = np.linalg.norm(np.array(tip_bone.tail) - wrist)
                    pb.rotation_quaternion = Quaternion(axis, np.radians(20))
                    bpy.context.view_layer.update()
                    d1 = np.linalg.norm(np.array(tip_bone.tail) - wrist)
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    self.hinge[name] = (np.array(axis), 1.0 if d1 < d0 else -1.0)
        bpy.context.view_layer.update()

    # ---------------- collision proxies: torso (static) + head (rigid with the head bone) -----------
    def _build_collision(self, objs):
        from mathutils.bvhtree import BVHTree
        torso_names = ("pelvis", "spine", "clavicle", "thigh")
        head_names = ("neck", "head")
        tv, tp, hv, hp = [], [], [], []
        head_rest_inv = Matrix(self.rest["head"].tolist()).inverted()
        for key in ("body", "clothes", "hair"):
            o = objs.get(key)
            if o is None:
                continue
            gname = {g.index: g.name.split("_")[0] for g in o.vertex_groups}
            dom = {}
            for v in o.data.vertices:
                if v.groups:
                    g = max(v.groups, key=lambda x: x.weight)
                    if g.weight > 0.5:
                        dom[v.index] = gname.get(g.group)
            mw = o.matrix_world
            for poly in o.data.polygons:
                kinds = {dom.get(i) for i in poly.vertices}
                if kinds <= set(torso_names) and None not in kinds:
                    base = len(tv)
                    tv.extend(mw @ o.data.vertices[i].co for i in poly.vertices)
                    tp.append(list(range(base, base + len(poly.vertices))))
                elif kinds <= set(head_names) and None not in kinds:
                    base = len(hv)
                    hv.extend(head_rest_inv @ (mw @ o.data.vertices[i].co) for i in poly.vertices)
                    hp.append(list(range(base, base + len(poly.vertices))))
        self.collide = dict(torso=BVHTree.FromPolygons(tv, tp) if tp else None,
                            head=BVHTree.FromPolygons(hv, hp) if hp else None)

    def _hand_points_world(self, s):
        pts = []
        for fn in ("thumb", "index", "middle", "ring", "pinky"):
            for k in ("02", "03"):
                pb = self.rig.pose.bones[f"{fn}_{k}_{s}"]
                pts.append(np.array(self.rig.matrix_world @ pb.head))
            pts.append(np.array(self.rig.matrix_world @ self.rig.pose.bones[f"{fn}_03_{s}"].tail))
        for fn in ("index", "middle", "ring", "pinky"):
            pts.append(np.array(self.rig.matrix_world @ self.rig.pose.bones[f"{fn}_01_{s}"].head))
        # forearm (wrist -> 30% towards the elbow) must not pass through the torso either
        wr = np.array(self.rig.matrix_world @ self.rig.pose.bones[f"hand_{s}"].head)
        el = np.array(self.rig.matrix_world @ self.rig.pose.bones[f"lowerarm_{s}"].head)
        for t in (0.0, 0.15, 0.3):
            pts.append(wr + t * (el - wr))
        return np.array(pts)

    def _body_penetration(self, s, margin=0.006):
        """Deepest finger-joint penetration (m) into torso/head proxies (0 if none)."""
        if self.collide is None:
            return 0.0
        worst = 0.0
        pts = self._hand_points_world(s)
        head_m = (self.rig.matrix_world @ self.rig.pose.bones["head"].matrix).inverted()
        for p in pts:
            for key, q in (("torso", p), ("head", np.array(head_m @ Vector(p)))):
                bvh = self.collide[key]
                if bvh is None:
                    continue
                loc, nrm, _, dist = bvh.find_nearest(Vector(q))
                if loc is None or dist > 0.12:
                    continue
                d = float(np.dot(np.array(q) - np.array(loc), np.array(nrm)))
                if d < margin:
                    worst = max(worst, margin - d)
        return worst

    @staticmethod
    def _hands_overlap_in_source(f):
        a, b = f[LH:LH + 21, :2], f[RH:RH + 21, :2]
        ix = min(a[:, 0].max(), b[:, 0].max()) - max(a[:, 0].min(), b[:, 0].min())
        iy = min(a[:, 1].max(), b[:, 1].max()) - max(a[:, 1].min(), b[:, 1].min())
        return ix > 0 and iy > 0

    def _hand_hand_overlap(self, r_finger=0.0085):
        """How far (m) the two hands' finger capsules interpenetrate (0 if apart)."""
        segs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (0, 17), (5, 17), (0, 9)]
        if Retargeter.LMAP21 is None:
            self._hand_screen("l")
        H = {}
        for s in "lr":
            m = Retargeter.LMAP21
            H[s] = np.array([np.array(self.rig.matrix_world @ (self.rig.pose.bones[f"{b}_{s}"].head if e == "head"
                                                                else self.rig.pose.bones[f"{b}_{s}"].tail))
                             for b, e in (m[i] for i in range(21))])
        worst, pair = -1.0, None
        for a, b in segs:
            for c, d in segs:
                dd = _seg_dist(H["l"][a], H["l"][b], H["r"][c], H["r"][d])
                if 2 * r_finger - dd > worst:
                    worst, pair = 2 * r_finger - dd, (a, b, c, d)
        # separation direction (right hand -> left hand) of the deepest pair
        a, b, c, d = pair
        n = (H["l"][a] + H["l"][b]) / 2 - (H["r"][c] + H["r"][d]) / 2
        if worst > r_finger or np.linalg.norm(n) < 1e-4:
            # deeply interpenetrating: segment midpoints are meaningless; move the hands apart
            # along the line between their palm centres
            n = H["l"][[0, 5, 9, 13, 17]].mean(0) - H["r"][[0, 5, 9, 13, 17]].mean(0)
        self.sep_dir = n / (np.linalg.norm(n) + 1e-9)
        return worst

    # ---------------- calibration (avatar measured with MediaPipe, see calibrate.py) -------------
    FACE_ANCHORS = (1, 152, 13, 14, 10, 234, 454, 33, 263, 61, 291, 199)
    TORSO_ANCHORS = ((0.0, 0.55), (0.0, 1.1), (0.42, 0.6), (-0.42, 0.6), (0.0, 1.7))

    def _load_calibration(self, cal):
        self.cal = cal
        sw = cal["mp_shoulder_width_m"]
        self.arm_len = {s: (np.linalg.norm(self.rest[f"lowerarm_{s}"][:3, 3] - self.rest[f"upperarm_{s}"][:3, 3]) / sw,
                            np.linalg.norm(self.rest[f"hand_{s}"][:3, 3] - self.rest[f"lowerarm_{s}"][:3, 3]) / sw)
                        for s in "lr"}
        bh = cal["bone_heads_canon"]
        self.shoulder = {s: np.array(bh[f"upperarm_{s}"]) for s in "lr"}
        # distance from the bone wrist joint to MediaPipe's wrist landmark, along the hand
        self.wrist_off = {}
        for s in "lr":
            hand, mid = np.array(bh[f"hand_{s}"]), np.array(bh[f"middle_01_{s}"])
            u = (mid - hand) / np.linalg.norm(mid - hand)
            mpw = cal["hands"].get(s.upper())
            self.wrist_off[s] = float(np.dot(np.array(mpw[0]) - hand, u)) if mpw else 0.0
        face = np.array(cal["face"])
        self.anchor_av = np.vstack([face[list(self.FACE_ANCHORS)], [cal["pose"]["11"], cal["pose"]["12"]],
                                    np.array(self.TORSO_ANCHORS)])
        self.last = {}

    def warp(self, p, f):
        """Map a 2D canonical point of the signer onto the avatar's body (inverse-distance
        weighted displacement field through matching face/shoulder/torso anchors)."""
        a_d = np.vstack([f[FACE + np.array(self.FACE_ANCHORS), :2], f[P + 11, :2], f[P + 12, :2],
                         np.array(self.TORSO_ANCHORS)])
        d2 = np.sum((a_d - p[:2]) ** 2, 1)
        w = 1.0 / (d2 + 0.02) ** 2
        disp = (w[:, None] * (self.anchor_av - a_d)).sum(0) / w.sum()
        return p[:2] + disp

    EFFECTOR = {5: ("index_01", "head"), 9: ("middle_01", "head"), 13: ("ring_01", "head"),
                17: ("pinky_01", "head"), 8: ("index_03", "tail"), 12: ("middle_03", "tail")}

    def _effector_screen(self, s):
        """Avatar's knuckle+fingertip centroid in calibrated canonical screen coords."""
        cal = self.cal
        sw, mx, mz = cal["mp_shoulder_width_m"], cal["mp_mid_world_xz"][0], cal["mp_mid_world_xz"][1]
        pts = []
        for bone, end in self.EFFECTOR.values():
            pb = self.rig.pose.bones[f"{bone}_{s}"]
            p = self.rig.matrix_world @ (pb.head if end == "head" else pb.tail)
            pts.append([(p[0] - mx) / sw, -(p[2] - mz) / sw])
        return np.mean(pts, 0)

    def set_expression(self, f, neutral, objs):
        """Drive facial expression units from the signer's face landmarks for frame f."""
        w = expression_weights(face_features(f[FACE:FACE + 468, :2]), neutral)
        for kind in ("body", "eyebrows", "eyelashes", "teeth", "tongue"):
            o = objs.get(kind)
            if o is None or o.data.shape_keys is None:
                continue
            kb = o.data.shape_keys.key_blocks
            for unit, val in w.items():
                if unit in kb:
                    kb[unit].value = val
        return w

    LMAP21 = None

    def _hand_screen(self, s):
        """Avatar's 21 hand joints of side s in calibrated canonical screen coords."""
        if Retargeter.LMAP21 is None:
            m = {0: ("hand", "head")}
            for fn, base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
                m.update({base: (f"{fn}_01", "head"), base + 1: (f"{fn}_02", "head"),
                          base + 2: (f"{fn}_03", "head"), base + 3: (f"{fn}_03", "tail")})
            Retargeter.LMAP21 = m
        cal = self.cal
        sw, mx, mz = cal["mp_shoulder_width_m"], cal["mp_mid_world_xz"][0], cal["mp_mid_world_xz"][1]
        out = np.zeros((21, 2))
        for i, (bone, end) in Retargeter.LMAP21.items():
            pb = self.rig.pose.bones[f"{bone}_{s}"]
            p = self.rig.matrix_world @ (pb.head if end == "head" else pb.tail)
            out[i] = ((p[0] - mx) / sw, -(p[2] - mz) / sw)
        return out

    @staticmethod
    def _shape_err(a, b):
        vis = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        ka, kb = a[[5, 9, 13, 17]].mean(0), b[[5, 9, 13, 17]].mean(0)
        ca, cb = a[vis] - ka, b[vis] - kb
        ca /= np.sqrt((ca ** 2).sum(1).mean()) + 1e-9
        cb /= np.sqrt((cb ** 2).sum(1).mean()) + 1e-9
        return float(np.sqrt(((ca - cb) ** 2).sum(1).mean()))

    @staticmethod
    def _mirror_depth(f, off):
        g = f.copy()
        z0 = g[off, 2]
        g[off:off + 21, 2] = 2 * z0 - g[off:off + 21, 2]
        return g

    def _choose_depth(self, f):
        """Analysis-by-synthesis depth disambiguation: a monocular hand and its depth-mirror
        project identically; pose the avatar with both and keep the one whose *rendered*
        hand matches the signer's 2D landmarks best (hysteresis against flicker)."""
        if not hasattr(self, "flip"):
            self.flip = {"l": False, "r": False}
        errs = {}
        for flip_l in (False, True):
            for flip_r in (False, True):
                g = f
                if flip_l:
                    g = self._mirror_depth(g, LH)
                if flip_r:
                    g = self._mirror_depth(g, RH)
                self._pose_once(g, {})
                bpy.context.view_layer.update()
                errs[(flip_l, flip_r)] = {s: self._shape_err(self._hand_screen(s), f[off:off + 21, :2])
                                          for s, off in (("l", LH), ("r", RH))}
        for s, idx in (("l", 0), ("r", 1)):
            cur = self.flip[s]
            e_cur = min(v[s] for k, v in errs.items() if k[idx] == cur)
            e_alt = min(v[s] for k, v in errs.items() if k[idx] != cur)
            if e_alt < 0.8 * e_cur:
                self.flip[s] = not cur
        g = f
        if self.flip["l"]:
            g = self._mirror_depth(g, LH)
        if self.flip["r"]:
            g = self._mirror_depth(g, RH)
        return g

    def plan_depth(self, seq, smooth=15, min_run=10):
        """Decide the depth reading of each hand for a whole sequence up front: score
        both readings on every frame, then median-filter the decision and remove runs
        shorter than `min_run`, so a hand never snaps orientation for a few frames."""
        def median_filter(x, size, mode=None):
            h = size // 2
            xp = np.pad(x, h, mode="edge")
            return np.array([np.median(xp[i:i + size]) for i in range(len(x))])
        votes = {"l": [], "r": []}
        for f in seq:
            errs = {}
            for flip in (False, True):
                g = self._mirror_depth(self._mirror_depth(f, LH), RH) if flip else f
                self._pose_once(g, {})
                bpy.context.view_layer.update()
                errs[flip] = {s: self._shape_err(self._hand_screen(s), f[off:off + 21, :2])
                              for s, off in (("l", LH), ("r", RH))}
            for s in "lr":
                votes[s].append(1.0 if errs[True][s] < 0.85 * errs[False][s] else 0.0)
        plan = {}
        for s in "lr":
            d = median_filter(np.array(votes[s]), size=smooth, mode="nearest") > 0.5
            i = 0
            while i < len(d):                      # merge short runs into their neighbours
                j = i
                while j < len(d) and d[j] == d[i]:
                    j += 1
                if j - i < min_run and (i > 0 or j < len(d)):
                    d[i:j] = d[i - 1] if i > 0 else d[j]
                i = j
            plan[s] = d
        self.depth_plan = plan
        self.frame_t = -1
        return plan

    def _fit_frame(self, s):
        fit = getattr(self, "fit", None)
        t = getattr(self, "frame_t", 0)
        if fit is None or t >= len(fit["valid"]) or not fit["valid"][t]:
            return None
        i = "lr".index(s)
        return fit["R"][t, i], fit["F"][t, i]

    BAKE_ARM = ("clavicle", "upperarm", "lowerarm", "hand")

    def bake(self, seq, segments=None, fit=None, objs=None):
        """Solve the whole sequence, then make it temporally coherent:
          1. per-frame solve (IK + end-effector correction + collision push-out);
          2. the per-frame corrections are smoothed over time and the frames re-posed with
             them (no frame-to-frame jitter from independent corrections);
          3. joint rotations are low-pass filtered;
          4. transitions between signs are generated in JOINT space: every joint goes from
             its pose at the end of one sign to its pose at the start of the next along a
             velocity-matched Hermite curve, fingers arriving early (pre-shaping). The arm
             therefore moves like an arm - no hand spinning through impossible wrist angles;
          5. facial expression weights: symmetric, damped, smoothed, neutral in holds."""
        T = len(seq)
        self.fit = fit
        solve = np.ones(T, bool)
        hold = np.zeros(T, bool)
        for kind, a, b, _ in (segments or []):
            if kind == "transition":
                solve[a:b] = False
            elif kind == "hold":
                hold[a:b] = True
        rig = self.rig
        names = [pb.name for pb in rig.pose.bones]
        corr = np.zeros((T, 2, 2))
        zx = np.zeros((T, 2))
        for t in np.flatnonzero(solve):
            self.frame_t = t - 1
            self.pose(seq[t])
            for i, s in enumerate("lr"):
                corr[t, i], zx[t, i] = self.last_corr[s], self.last_zx[s]
        for a, b in _blocks(solve):
            zx[a:b] = _gauss(zx[a:b], 1.5)
            corr[a:b] = _gauss(corr[a:b], 1.5)
        Q = np.zeros((T, len(names), 4))
        Q[:, :, 0] = 1
        last = [None] * T
        for t in np.flatnonzero(solve):
            self.frame_t = t
            c_t, z_t = {"l": corr[t, 0].copy(), "r": corr[t, 1].copy()}, {"l": zx[t, 0], "r": zx[t, 1]}
            self._pose_once(seq[t], c_t, z_t)
            self._resolve(seq[t], c_t, z_t, iters=6)       # smoothing must not re-introduce penetration
            Q[t] = [tuple(pb.rotation_quaternion) for pb in rig.pose.bones]
            last[t] = {k: dict(v) for k, v in self.last.items()}
        # temporal coherence of rotations (sign-continuous quaternions, light low-pass)
        slow = np.array([n.rsplit("_", 1)[0] in self.BAKE_ARM or n.split("_")[0] in ("neck", "head", "spine")
                         for n in names])
        for a, b in _blocks(solve):
            for t in range(a + 1, b):
                flip = np.sum(Q[t] * Q[t - 1], axis=1) < 0
                Q[t, flip] *= -1
            Q[a:b, slow] = _gauss(Q[a:b, slow], 1.5)
            Q[a:b, ~slow] = _gauss(Q[a:b, ~slow], 0.5)
            Q[a:b] /= np.linalg.norm(Q[a:b], axis=2, keepdims=True)
        # The arm was low-pass filtered; restore each hand's fitted WORLD orientation on top of
        # the smoothed arm so the handshape/orientation solved against the image is kept exactly.
        for t in np.flatnonzero(solve):
            if last[t] is None or "hand_R" not in last[t].get("l", {}):
                continue
            for name, q in zip(names, Q[t]):
                rig.pose.bones[name].rotation_quaternion = q
            bpy.context.view_layer.update()
            for s in "lr":
                hb = rig.pose.bones[f"hand_{s}"]
                par = np.array(hb.parent.matrix)[:3, :3]
                chain = par @ self.rest[hb.parent.name][:3, :3].T @ self.rest[hb.name][:3, :3]
                Q[t, names.index(hb.name)] = tuple(Matrix((chain.T @ last[t][s]["hand_R"]).tolist()).to_quaternion())
        # joint-space transitions
        finger = np.array([n in self.finger_bones for n in names])
        for kind, a, b, _ in (segments or []):
            if kind != "transition" or a == 0 or b >= T:
                continue
            n, D = b - a, b - a + 1
            for j in range(len(names)):
                qa, qb = Quaternion(Q[a - 1, j]), Quaternion(Q[b, j])
                if qa.dot(qb) < 0:
                    qb.negate()
                if finger[j]:
                    for k in range(1, n + 1):
                        Q[a + k - 1, j] = tuple(qa.slerp(qb, _minjerk(min(1.0, (k / D) / 0.75))))
                    continue
                if names[j].rsplit("_", 1)[0] in ("upperarm", "lowerarm", "hand"):
                    # Swing-twist: the bend (swing) and the roll about the bone (twist) are
                    # interpolated separately, so an elbow goes monotonically from one flexion
                    # to the other instead of straightening out on the way.
                    sa, ta = qa.to_swing_twist("Y")
                    sb, tb = qb.to_swing_twist("Y")
                    if sa.dot(sb) < 0:
                        sb.negate()
                    dt = (tb - ta + np.pi) % (2 * np.pi) - np.pi
                    rs = np.array((sa.inverted() @ sb).to_exponential_map())
                    for k in range(1, n + 1):
                        h = _minjerk(k / D)
                        v = h * rs
                        sw = sa @ Quaternion(Vector(v).normalized(), float(np.linalg.norm(v))) \
                            if np.linalg.norm(v) > 1e-9 else sa
                        Q[a + k - 1, j] = tuple(sw @ Quaternion((0, 1, 0), ta + h * dt))
                    continue
                r = np.array((qa.inverted() @ qb).to_exponential_map())

                def vel(q0, q1):
                    q0, q1 = Quaternion(q0), Quaternion(q1)
                    if q0.dot(q1) < 0:
                        q1.negate()
                    return np.array((q0.inverted() @ q1).to_exponential_map())
                wa = vel(Q[a - 2, j], Q[a - 1, j]) if a >= 2 and solve[a - 2] else np.zeros(3)
                wb = vel(Q[b, j], Q[b + 1, j]) if b + 1 < T and solve[b + 1] else np.zeros(3)
                cap = (1.2 * np.linalg.norm(r) + 0.1) / D
                for w in (wa, wb):
                    m = np.linalg.norm(w)
                    if m > cap:
                        w *= cap / m
                for k in range(1, n + 1):
                    t = k / D
                    h1 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
                    h4 = -4 * t ** 3 + 7 * t ** 4 - 3 * t ** 5
                    h5 = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
                    v = h1 * wa * D + h4 * wb * D + h5 * r
                    Q[a + k - 1, j] = tuple(qa @ Quaternion(Vector(v).normalized(), float(np.linalg.norm(v)))
                                            if np.linalg.norm(v) > 1e-9 else qa)
        # Arms during transitions: the HAND travels a smooth, nearly straight path in space
        # (how people reach - Flash & Hogan's minimum-jerk hand paths), the elbow follows by
        # inverse kinematics, and the forearm roll is interpolated. (Pure joint interpolation
        # swings the arm out sideways like a wing.)
        def arm_points(t):
            for name, q in zip(names, Q[t]):
                rig.pose.bones[name].rotation_quaternion = q
            bpy.context.view_layer.update()
            return {s: (np.array(rig.pose.bones[f"lowerarm_{s}"].head), np.array(rig.pose.bones[f"hand_{s}"].head))
                    for s in "lr"}

        ju = {s: names.index(f"upperarm_{s}") for s in "lr"}
        jl = {s: names.index(f"lowerarm_{s}") for s in "lr"}
        L = {s: (rig.data.bones[f"upperarm_{s}"].length, rig.data.bones[f"lowerarm_{s}"].length) for s in "lr"}
        for kind, a, b, _ in (segments or []):
            if kind != "transition" or a == 0 or b >= T:
                continue
            n, D = b - a, b - a + 1
            Pa, Pb = arm_points(a - 1), arm_points(b)
            Pa0 = arm_points(a - 2) if a >= 2 and solve[a - 2] else Pa
            Pb1 = arm_points(b + 1) if b + 1 < T and solve[b + 1] else Pb
            tw = {}
            for s in "lr":
                ta = Quaternion(Q[a - 1, jl[s]]).to_swing_twist("Y")[1]
                tb = Quaternion(Q[b, jl[s]]).to_swing_twist("Y")[1]
                tw[s] = (ta, (tb - ta + np.pi) % (2 * np.pi) - np.pi)
            for k in range(1, n + 1):
                t = k / D
                h0 = 1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5
                h1 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
                h4 = -4 * t ** 3 + 7 * t ** 4 - 3 * t ** 5
                h5 = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
                arm_points(a + k - 1)                      # poses clavicle/spine for this frame
                off = {"l": np.zeros(3), "r": np.zeros(3)}
                for attempt in range(5):
                  for s in "lr":
                      (Ea, Wa), (Eb, Wb) = Pa[s], Pb[s]
                      va, vb = Wa - Pa0[s][1], Pb1[s][1] - Wb
                      dist = np.linalg.norm(Wb - Wa)
                      cap = (1.2 * dist + 0.03) / D
                      for v in (va, vb):
                          m = np.linalg.norm(v)
                          if m > cap:
                              v *= cap / m
                      W = h0 * Wa + h1 * va * D + h4 * vb * D + h5 * Wb
                      W[1] -= min(0.05, 0.2 * dist) * np.sin(np.pi * t)       # slight forward arc
                      W = W + off[s]
                      E_hint = (1 - h5) * Ea + h5 * Eb
                      S = np.array(rig.pose.bones[f"upperarm_{s}"].head)
                      la, lb = L[s]
                      u = W - S
                      dd = float(np.clip(np.linalg.norm(u), abs(la - lb) + 1e-3, la + lb - 1e-3))
                      u /= np.linalg.norm(u) + 1e-12
                      W = S + u * dd
                      cos_a = np.clip((la * la + dd * dd - lb * lb) / (2 * la * dd), -1, 1)
                      pole = E_hint - S
                      pv = pole - u * (pole @ u)
                      if np.linalg.norm(pv) < 1e-6:
                          pv = np.array([0, 0, -1.0]) - u * (-u[2])
                      pv /= np.linalg.norm(pv)
                      E = S + la * (cos_a * u + np.sqrt(1 - cos_a ** 2) * pv)
                      du, dv = (E - S) / la, (W - E) / np.linalg.norm(W - E)
                      cl = rig.pose.bones[f"clavicle_{s}"]
                      chain_u = np.array(cl.matrix)[:3, :3] @ self.rest[cl.name][:3, :3].T @ self.rest[f"upperarm_{s}"][:3, :3]
                      R_min = np.array(Vector(chain_u[:, 1]).rotation_difference(Vector(du)).to_matrix()) @ chain_u
                      nrm = np.cross(du, dv)
                      sin_f = np.linalg.norm(nrm)
                      wgt = float(np.clip(sin_f / 0.3, 0.0, 1.0))
                      n_e = wgt * nrm / (sin_f + 1e-9) + (1 - wgt) * (R_min @ self.hinge_local[s])
                      R_u = _frame(du, n_e) @ _frame(np.array([0, 1.0, 0]), self.hinge_local[s]).T
                      chain_l = R_u @ self.rest[f"upperarm_{s}"][:3, :3].T @ self.rest[f"lowerarm_{s}"][:3, :3]
                      R_l0 = np.array(Vector(chain_l[:, 1]).rotation_difference(Vector(dv)).to_matrix()) @ chain_l
                      sw0 = Matrix((chain_l.T @ R_l0).tolist()).to_quaternion().to_swing_twist("Y")[0]
                      Q[a + k - 1, ju[s]] = tuple(Matrix((chain_u.T @ R_u).tolist()).to_quaternion())
                      Q[a + k - 1, jl[s]] = tuple(sw0 @ Quaternion((0, 1, 0), tw[s][0] + h5 * tw[s][1]))
                  # hands must not pass through each other on the way: move the paths apart
                  arm_points(a + k - 1)
                  ov = self._hand_hand_overlap()
                  if ov <= 0:
                        break
                  off["l"] = off["l"] + (ov / 2 + 0.003) * self.sep_dir
                  off["r"] = off["r"] - (ov / 2 + 0.003) * self.sep_dir
        # facial expression
        expr = None
        if objs is not None:
            feats = [face_features(f[FACE:FACE + 468, :2]) for f in seq]
            neutral = neutral_face([feats[t] for t in np.flatnonzero(solve & ~hold)] or feats)
            keys = list(expression_weights(feats[0], neutral))
            W = np.array([[expression_weights(ft, neutral)[k] for k in keys] for ft in feats])
            base = expression_weights(neutral, neutral)
            W[hold] = [base[k] for k in keys]
            for t0, t1 in [(a, b) for kind, a, b, _ in (segments or []) if kind == "transition"]:
                lo, hi = max(t0 - 1, 0), min(t1, T - 1)
                for k in range(t0, t1):
                    u = (k - lo) / max(hi - lo, 1)
                    W[k] = (1 - u) * W[lo] + u * W[hi]
            for kl in [k for k in keys if "-left-" in k]:          # symmetric brows / eyes
                kr = kl.replace("-left-", "-right-")
                m = (W[:, keys.index(kl)] + W[:, keys.index(kr)]) / 2
                W[:, keys.index(kl)] = W[:, keys.index(kr)] = m
            gain = {"mouth-open": 0.55, "mouth-pursing": 0.4, "mouth-corner-puller": 1.0}
            for k in keys:
                g = gain.get(k, 0.7 if k.startswith("eyebrows") else 0.6)
                W[:, keys.index(k)] *= g
            W = np.clip(_gauss(W, 3.0), 0, 1)
            # natural blinking (people blink every 3-5 s): a living face, never a fixed stare
            rng = np.random.default_rng(7)
            t_b = int(rng.integers(40, 80))
            prof = np.array([0.45, 0.95, 1.0, 0.7, 0.3])
            while t_b + len(prof) < T:
                for kk in ("eye-left-closure", "eye-right-closure"):
                    if kk in keys:
                        j = keys.index(kk)
                        W[t_b:t_b + len(prof), j] = np.maximum(W[t_b:t_b + len(prof), j], prof)
                t_b += int(rng.integers(90, 150))
            expr = (keys, W)
        self.baked = dict(names=names, Q=Q, expr=expr, last=last, solve=solve, hold=hold)
        return self.baked

    def apply(self, t, objs=None):
        bk = self.baked
        for name, q in zip(bk["names"], bk["Q"][t]):
            self.rig.pose.bones[name].rotation_quaternion = q
        if bk["last"][t] is not None:
            self.last = bk["last"][t]
        if objs is not None and bk["expr"] is not None:
            keys, W = bk["expr"]
            for kind in ("body", "eyebrows", "eyelashes", "teeth", "tongue"):
                o = objs.get(kind)
                if o is None or o.data.shape_keys is None:
                    continue
                kb = o.data.shape_keys.key_blocks
                for k, v in zip(keys, W[t]):
                    if k in kb:
                        kb[k].value = float(v)
        bpy.context.view_layer.update()

    def _resolve(self, f, corr, zx, iters=10):
        """Contact without penetration: push fingers/forearms out of the body along the camera
        depth axis, and separate interpenetrating hands along the line between their closest
        finger segments (a fist resting on a palm moves up off the palm). Keeps the best
        configuration seen, so it can never end worse than it started."""
        sw = self.cal["mp_shoulder_width_m"]
        best = None
        for it in range(iters + 1):
            bpy.context.view_layer.update()
            pen = {s: self._body_penetration(s) for s in "lr"}
            ov = self._hand_hand_overlap()
            score = max(ov, 0.0) + max(pen.values())
            if best is None or score < best[0] - 1e-5:
                best = (score, {k: np.array(v) for k, v in corr.items()}, dict(zx))
            if it == iters:
                break
            moved = False
            for s in "lr":
                if pen[s] > 0:
                    zx[s] -= (pen[s] + 0.004) / sw
                    moved = True
            front = "l" if f[LH, 2] < f[RH, 2] - 1e-6 else "r"
            back = "r" if front == "l" else "l"
            if ov > 0:
                step = (ov / 2 + 0.002) * self.sep_dir
                for sd, sg in (("l", 1.0), ("r", -1.0)):
                    corr[sd] = corr[sd] + sg * np.array([step[0], -step[2]]) / sw
                    zx[sd] += sg * step[1] / sw
                moved = True
            elif self._hands_overlap_in_source(f) and -0.03 < ov < -0.004:
                # the signer's hands overlap in the image and the avatar's are close but apart
                # in depth: bring the back hand forward until they touch (contact, not a gap)
                zx[back] -= (-ov - 0.002) / sw
                moved = True
            if not moved:
                return
            self._pose_once(f, corr, zx)
        if best[0] < score - 1e-5:                     # ended worse than the best seen: restore it
            corr.update(best[1])
            zx.update(best[2])
            self._pose_once(f, corr, zx)

    def pose(self, f, iters=2):
        """Retarget one frame. With calibration, a closed-loop end-effector correction
        moves each wrist target until the avatar's knuckles+fingertips land where the
        signer's do (mapped onto the avatar's body), so hand-hand and hand-body
        relationships survive differences in finger proportions."""
        if self.cal is None:
            return self._pose_once(f, {})
        self.frame_t = getattr(self, "frame_t", -1) + 1
        plan = getattr(self, "depth_plan", None)
        if getattr(self, "fit", None) is not None:
            pass                                   # the hand fit already resolved depth
        elif plan is not None and self.frame_t < len(plan["l"]):
            if plan["l"][self.frame_t]:
                f = self._mirror_depth(f, LH)
            if plan["r"][self.frame_t]:
                f = self._mirror_depth(f, RH)
        elif getattr(self, "disambiguate", True):
            f = self._choose_depth(f)
        corr = {"l": np.zeros(2), "r": np.zeros(2)}
        idx = list(self.EFFECTOR)
        zx = {"l": 0.0, "r": 0.0}
        for it in range(iters + 1):
            self._pose_once(f, corr, zx)
            if it == iters:
                break
            bpy.context.view_layer.update()
            for s, off in (("l", LH), ("r", RH)):
                tgt = np.mean([self.warp(f[off + i], f) for i in idx], 0)
                corr[s] = corr[s] + np.clip(tgt - self._effector_screen(s), -0.15, 0.15)
        self._resolve(f, corr, zx)
        self.last_corr, self.last_zx = corr, zx

    def _pose_once(self, f, corr, zx=None):
        zx = zx or {}
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

        def set_world(name, R_world):
            posed[name] = R_world
            basis = chain_rest(name).T @ R_world
            rig.pose.bones[name].rotation_quaternion = Matrix(basis.tolist()).to_quaternion()

        def rot(axis, deg):
            return np.array(Matrix.Rotation(np.radians(deg), 3, axis))

        # Ambient life: slow breathing (chest pitch, ~0.25 Hz) and subtle postural sway, so the
        # signer is never frozen, even in holds and pauses between sentences.
        t = getattr(self, "frame_t", 0) / 30.0
        set_world("spine_02", rot("Z", 0.5 * np.sin(2 * np.pi * t / 7.3)) @ rot("Y", 0.35 * np.sin(2 * np.pi * t / 9.1))
                  @ chain_rest("spine_02"))
        set_world("spine_03", rot("X", -0.6 * np.sin(2 * np.pi * t / 4.0)) @ chain_rest("spine_03"))

        for s in "lr":
            off = LH if s == "l" else RH
            sh, el = (11, 13) if s == "l" else (12, 14)
            if self.cal is not None:
                S2 = self.shoulder[s]
                T_mp = self.warp(f[off], f)                       # MediaPipe-wrist target on the avatar
                E2 = self.warp(f[P + el], f)
                # Scapulohumeral rhythm: above ~60 deg of arm elevation the shoulder girdle
                # rotates upward too (~1 deg per 2 deg of arm), so the shoulder rises naturally
                # instead of the mesh pinching.
                v = E2 - S2
                elev = np.degrees(np.arccos(np.clip(v[1] / (np.linalg.norm(v) + 1e-9), -1, 1)))
                clav = float(np.clip((elev - 60.0) / 3.0, 0.0, 22.0))
                cname = f"clavicle_{s}"
                R_c = rot("Y", -clav if s == "l" else clav)
                set_world(cname, R_c @ chain_rest(cname))
                if clav > 0:
                    hc, hu = self.rest[cname][:3, 3], self.rest[f"upperarm_{s}"][:3, 3]
                    moved = hc + R_c @ (hu - hc) - hu
                    S2 = S2 + np.array([moved[0], -moved[2]]) / self.cal["mp_shoulder_width_m"]
                hd = to_dir(f[off + 9]) - to_dir(f[off])           # 3D hand direction (Blender axes)
                hd /= np.linalg.norm(hd) + 1e-9
                hd2 = np.array([hd[0], -hd[2]])                     # its screen (x, y-down) component
                T2 = T_mp - self.wrist_off[s] * hd2 + corr.get(s, 0.0)   # where the wrist *joint* must go
                E, T = solve_arm(S2, np.array([*E2, 0.0]), np.array([*T2, 0.0]), *self.arm_len[s],
                                 z_bias=float(np.clip(f[off, 2], -0.15, 0.15)) + zx.get(s, 0.0))
                self.last[s] = dict(wrist_mp_target=T_mp, elbow_target=E2, shoulder=S2, hand_dir2=hd2)
                S3 = np.array([S2[0], S2[1], 0.0])
            else:
                E, T = solve_arm(f[P + sh], f[P + el], f[off], *self.arm_len[s])
                S3 = np.array([f[P + sh][0], f[P + sh][1], 0.0])
            arm_dirs = {f"upperarm_{s}": canon_dir(E - S3),
                        f"lowerarm_{s}": canon_dir(T - E)}
            head_rel, tgt = {}, None
            fit = self._fit_frame(s)
            for part in ORDER:
                name = f"{part}_{s}"
                a, b = CHAINS[name]
                if fit is not None and (part == "hand" or name in self.finger_bones):
                    # avatar-space hand fit (asl_v2.handfit): hand orientation + finger joint
                    # rotations solved against the 2D landmarks under joint limits
                    if part == "hand":
                        R_final = fit[0]
                        qb = Matrix((chain_rest(name).T @ R_final).tolist()).to_quaternion()
                    else:
                        fn_, k_ = part.split("_")
                        Lm = fit[1][FIT_ORDER.index(fn_) * 3 + int(k_) - 1]
                        R_final = chain_rest(name) @ Lm
                        qb = Matrix(Lm.tolist()).to_quaternion()
                    posed[name] = R_final
                    rig.pose.bones[name].rotation_quaternion = qb
                    if part == "hand":
                        self.last[s]["hand_R"] = R_final
                    continue
                d = arm_dirs[name] if name in arm_dirs else to_dir(f[b]) - to_dir(f[a])
                if tgt is not None and name in self.finger_bones and part.endswith("_01"):
                    # Position-based finger retargeting: aim each bone from its *actual posed
                    # head* at where the signer's next joint is inside the hand (anchored on the
                    # knuckles, scaled by finger length). Absorbs differences in joint placement
                    # (e.g. the thumb base) between the avatar's anatomy and MediaPipe's landmarks.
                    par = self.parent[name]
                    head_rel[name] = head_rel[par] + posed[par] @ self.rest[par][:3, :3].T @ (
                        self.rest[name][:3, 3] - self.rest[par][:3, 3])
                    d = tgt[b - off] - head_rel[name]
                elif tgt is not None and name in self.finger_bones:
                    par = self.parent[name]
                    head_rel[name] = head_rel[par] + posed[par] @ self.rest[par][:3, :3].T @ (
                        self.rest[name][:3, 3] - self.rest[par][:3, 3])
                    if name in self.hinge and getattr(self, "use_hinge", False):
                        # Hinge joint: apply the signer's measured bend angle about the joint's
                        # true flexion axis (exact curl; aiming at points under-curls fists).
                        D = to_dir(f[off:off + 21])
                        u = D[a - off] - D[a - off - 1] if (a - off) % 4 != 1 else D[a - off] - D[0]
                        w = D[b - off] - D[a - off]
                        ang = np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))
                        axis, sgn = self.hinge[name]
                        RA = chain_rest(name)
                        R_final = RA @ np.array(Quaternion(Vector(axis), sgn * ang).to_matrix())
                        qb = Matrix((RA.T @ R_final).tolist()).to_quaternion()
                        fn_, k_ = part.split("_")
                        lim = np.radians((ROM_THUMB if fn_ == "thumb" else ROM)[k_])
                        if qb.angle > lim:
                            qb = Quaternion().slerp(qb, lim / qb.angle)
                            R_final = RA @ np.array(qb.to_matrix())
                        posed[name] = R_final
                        rig.pose.bones[name].rotation_quaternion = qb
                        continue
                if np.linalg.norm(d) < 1e-6:
                    continue
                d /= np.linalg.norm(d)
                RA = chain_rest(name)
                y0 = RA[:, 1]
                q = Vector(y0).rotation_difference(Vector(d)).to_matrix()
                R_min = np.array(q) @ RA
                R_final = R_min
                if part == "upperarm":
                    # The elbow is a hinge: roll the upper arm (humeral rotation) so its hinge
                    # axis is the axis the forearm actually bends about. The elbow then deforms
                    # like an elbow and joint-space transitions flex it monotonically.
                    v_f = arm_dirs[f"lowerarm_{s}"] / (np.linalg.norm(arm_dirs[f"lowerarm_{s}"]) + 1e-9)
                    nrm = np.cross(d, v_f)
                    sin_f = np.linalg.norm(nrm)
                    n_min = R_min @ self.hinge_local[s]
                    wgt = float(np.clip(sin_f / 0.3, 0.0, 1.0))
                    n_e = wgt * nrm / (sin_f + 1e-9) + (1 - wgt) * n_min
                    R_final = _frame(d, n_e) @ _frame(np.array([0, 1.0, 0]), self.hinge_local[s]).T
                if part in ("hand", "lowerarm"):
                    off = LH if s == "l" else RH
                    h = to_dir(f[off:off + 21])
                    n_t = np.cross(h[5] - h[0], h[17] - h[0])
                    n_t /= np.linalg.norm(n_t) + 1e-9
                    if fit is not None:
                        n_t = fit[0] @ self.palm_local[s]
                    if part == "hand":
                        R_final = _frame(d, n_t) @ _frame(np.array([0, 1.0, 0]), self.palm_local[s]).T
                    else:   # forearm takes half of the hand's twist (pronation)
                        R_full = _frame(d, n_t) @ _frame(np.array([0, 1.0, 0]), self.palm_local[f"fore_{s}"]).T
                        q0 = Matrix(R_min.tolist()).to_quaternion()
                        q1 = Matrix(R_full.tolist()).to_quaternion()
                        R_final = np.array(q0.slerp(q1, 0.5).to_matrix())
                posed[name] = R_final
                if part == "hand" and self.cal is not None:
                    head_rel = {name: np.zeros(3)}
                    knuckles = []
                    for fb in ("index_01", "middle_01", "ring_01", "pinky_01"):
                        bn = f"{fb}_{s}"
                        knuckles.append(R_final @ self.rest[name][:3, :3].T @ (self.rest[bn][:3, 3] - self.rest[name][:3, 3]))
                    D = to_dir(f[off:off + 21])
                    data_len = sum(np.linalg.norm(D[base + k + 1] - D[base + k]) for base in (5, 9, 13, 17) for k in range(3))
                    k_scale = self.finger_len[s] / (data_len + 1e-9)
                    cD, cA = D[[5, 9, 13, 17]].mean(0), np.mean(knuckles, 0)
                    tgt = cA + k_scale * (D - cD)
                basis = RA.T @ R_final
                qb = Matrix(basis.tolist()).to_quaternion()
                if name in self.finger_bones:
                    fn, k = part.split("_")
                    lim = np.radians((ROM_THUMB if fn == "thumb" else ROM)[k])
                    if qb.angle > lim:                       # clamp to human range of motion
                        qb = Quaternion().slerp(qb, lim / qb.angle)
                        posed[name] = RA @ np.array(qb.to_matrix())
                rig.pose.bones[name].rotation_quaternion = qb

        # head orientation from the face mesh
        face = to_dir(f[FACE:FACE + 468])
        right = face[454] - face[234]
        up = face[10] - face[152]
        Rt = _frame(up, -right)           # y = up, z = -right  (x = forward-ish)
        Rr = _frame(np.array([0, 0, 1.0]), np.array([-1.0, 0, 0]))
        R_face = Rt @ Rr.T
        q = Matrix(R_face.tolist()).to_quaternion()
        q_half = Quaternion().slerp(q, 0.5 * 0.85)   # neck + head share the turn; damp tracker noise
        for name, qq in (("neck_01", q_half), ("head", q_half)):
            R_rest = self.rest[name][:3, :3]
            local = R_rest.T @ np.array(qq.to_matrix()) @ R_rest
            rig.pose.bones[name].rotation_quaternion = Matrix(local.tolist()).to_quaternion()
