"""
Fast renderer (same look and options as render_human.py, about 4x less time per frame).

  .venv-blender/bin/python asl_v2/render_fast.py motion.npy out.mp4 [--samples 12] [--res 1280x720]
  ... --frames 30,100        # stills (out_0030.png, ...)
  ... --draft                # quick preview: 4 samples at 2/3 resolution, upscaled by ffmpeg
  ... --workers 2            # frame-parallel worker processes (threads are split between them)
  ... --mode direct          # no plate/alpha trick, only the cheaper sampling/denoise settings

What changes versus render_human.py (measured on 4 CPU cores, 1280x720):
  1. The backdrop is rendered ONCE per clip ("plate": character hidden from the camera but still
     casting its soft shadow). Each frame then renders and denoises only the character's screen
     bounding box (+margin) and the compositor blends that box over the plate with a feathered
     edge (scene-linear, before the AgX view transform). ~84% of the pixels are backdrop and they
     cost as much per sample as the character; the box is ~30% of the frame.
  2. OIDN prefilter ACCURATE -> FAST (the accurate prefilter alone was ~3 s of the 12 s/frame).
  3. 12 samples instead of 14 (8 is ~30% faster again with slightly wobblier edges at 4x zoom);
     adaptive sampling off (it never triggers below 16 samples).
  4. Skin subsurface scattering off (weight 0.12 / 8 mm radius is not visible at this framing).
Scene sync, BVH and subdivision were measured at < 0.1 s/frame and are left alone.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import bpy
import numpy as np

ASL_V2 = os.environ.get("ASL_V2_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ASL_V2)
import mh_avatar  # noqa: E402
import render_human  # noqa: E402  (setup_scene)


# --------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------
def fast_settings(sc, samples, sss=False, prefilter="FAST"):
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = False          # no effect below ~32 samples, only bookkeeping
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = prefilter        # ACCURATE costs ~2.8 s/frame at 720p on 4 cores
    cy.denoising_quality = "HIGH"
    if os.environ.get("NO_DENOISE"):
        cy.use_denoising = False
    sc.render.use_persistent_data = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15
    if not sss:
        skin = bpy.data.materials.get("skin")
        if skin is not None:
            skin.node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.0


def backdrop_object():
    for o in bpy.data.objects:
        if o.type == "MESH" and o.data.materials and o.data.materials[0] and o.data.materials[0].name == "backdrop":
            return o
    raise RuntimeError("backdrop plane not found")


def render_plate(sc, objs, path, samples):
    """Backdrop without the character (which still casts shadows / bounces light), scene-linear EXR."""
    cy, r = sc.cycles, sc.render
    keep = (cy.samples, cy.denoising_prefilter, r.film_transparent, r.use_border, r.use_compositing, r.filepath)
    for o in objs.values():
        o.visible_camera = False
    backdrop_object().visible_camera = True
    cy.samples, cy.denoising_prefilter = samples, "ACCURATE"
    r.film_transparent, r.use_border, r.use_compositing = False, False, False
    ims = r.image_settings
    ims.file_format, ims.color_mode, ims.color_depth, ims.exr_codec = "OPEN_EXR", "RGB", "16", "ZIP"
    r.filepath = path
    bpy.ops.render.render(write_still=True)
    ims.file_format, ims.color_mode, ims.color_depth, ims.compression = "PNG", "RGB", "8", 15
    for o in objs.values():
        o.visible_camera = True
    cy.samples, cy.denoising_prefilter, r.film_transparent, r.use_border, r.use_compositing, r.filepath = keep


FEATHER = 16      # px: blend width between the freshly rendered box and the plate
MARGIN = 10       # px: box margin around the character's screen bounding box (before the feather)


def use_plate(sc, plate_path):
    """Compositor: plate everywhere, the rendered border box blended over it through a feathered
    mask (scene-linear, before the view transform). The backdrop INSIDE the box is still path
    traced every frame, so contact/near shadows and silhouette anti-aliasing are exactly as before."""
    r = sc.render
    W, H = r.resolution_x, r.resolution_y
    plate = bpy.data.images.load(plate_path)
    mask = bpy.data.images.new("box_mask", W, H, alpha=False, float_buffer=True, is_data=True)
    tree = bpy.data.node_groups.new("plate_comp", "CompositorNodeTree")
    rl = tree.nodes.new("CompositorNodeRLayers")
    im_plate = tree.nodes.new("CompositorNodeImage")
    im_plate.image = plate
    im_mask = tree.nodes.new("CompositorNodeImage")
    im_mask.image = mask
    sa = tree.nodes.new("CompositorNodeSetAlpha")          # premultiplies the box by the mask
    sa.inputs["Type"].default_value = "Apply Mask"
    ao = tree.nodes.new("CompositorNodeAlphaOver")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    gout = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rl.outputs["Image"], sa.inputs["Image"])
    tree.links.new(im_mask.outputs["Image"], sa.inputs["Alpha"])
    tree.links.new(im_plate.outputs["Image"], ao.inputs["Background"])
    tree.links.new(sa.outputs["Image"], ao.inputs["Foreground"])
    tree.links.new(ao.outputs[0], gout.inputs[0])
    sc.compositing_node_group = tree
    r.use_compositing = True
    r.use_border, r.use_crop_to_border = True, False
    return mask


def character_bbox(sc, objs):
    """Screen-space bounding box (pixels, origin bottom-left) of the posed character meshes."""
    cam = sc.camera
    W, H = sc.render.resolution_x, sc.render.resolution_y
    dg = bpy.context.evaluated_depsgraph_get()
    inv = np.array(cam.matrix_world.inverted())
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for o in objs.values():
        ev = o.evaluated_get(dg)
        co = np.empty(len(ev.data.vertices) * 3, np.float32)
        ev.data.vertices.foreach_get("co", co)
        M = inv @ np.array(ev.matrix_world)
        p = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        if cam.data.type == "ORTHO":
            x, y = p[:, 0] / cam.data.ortho_scale, p[:, 1] / cam.data.ortho_scale
        else:
            k = cam.data.lens / cam.data.sensor_width
            z = np.maximum(-p[:, 2], 1e-6)
            x, y = k * p[:, 0] / z, k * p[:, 1] / z
        if W >= H:                      # sensor fit AUTO: the larger dimension spans the sensor
            y = y * W / H
        else:
            x = x * H / W
        lo = np.minimum(lo, [x.min(), y.min()])
        hi = np.maximum(hi, [x.max(), y.max()])
    return (lo[0] + 0.5) * W, (lo[1] + 0.5) * H, (hi[0] + 0.5) * W, (hi[1] + 0.5) * H


def set_border(sc, objs, mask):
    """Render border = character bbox + margin + feather; update the blend mask. Call after apply()."""
    r = sc.render
    W, H = r.resolution_x, r.resolution_y
    bx0, by0, bx1, by1 = character_bbox(sc, objs)
    pad = MARGIN + FEATHER + 2
    x0, y0 = int(max(0, np.floor(bx0) - pad)), int(max(0, np.floor(by0) - pad))
    x1, y1 = int(min(W, np.ceil(bx1) + pad)), int(min(H, np.ceil(by1) + pad))
    r.border_min_x, r.border_max_x = (x0 + 0.01) / W, (x1 + 0.01) / W
    r.border_min_y, r.border_max_y = (y0 + 0.01) / H, (y1 + 0.01) / H
    xs, ys = np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
    # 0 on the two outermost box pixels, 1 FEATHER px further in; no ramp where the box meets the frame edge
    mx = np.clip((xs - x0 - 1.5) / FEATHER, 0, 1) if x0 > 0 else (xs >= 0).astype(np.float32)
    mx = mx * (np.clip((x1 - 2.5 - xs) / FEATHER, 0, 1) if x1 < W else 1.0)
    my = np.clip((ys - y0 - 1.5) / FEATHER, 0, 1) if y0 > 0 else (ys >= 0).astype(np.float32)
    my = my * (np.clip((y1 - 2.5 - ys) / FEATHER, 0, 1) if y1 < H else 1.0)
    m = np.outer(my, mx).astype(np.float32)
    mask.pixels.foreach_set(np.repeat(m.ravel(), 4))
    mask.update()
    return (x1 - x0) * (y1 - y0) / (W * H)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("output")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--frames")
    ap.add_argument("--skin", default="young_caucasian_male")
    ap.add_argument("--clothes", default="male_casualsuit01")
    ap.add_argument("--hair", default="short02")
    ap.add_argument("--closeup", action="store_true")
    ap.add_argument("--faceup", action="store_true")
    ap.add_argument("--audit-view", action="store_true")
    ap.add_argument("--mode", choices=["auto", "plate", "direct"], default="auto",
                    help="plate: backdrop once + character-only frames; direct: full frames; auto: plate for >= 12 frames")
    ap.add_argument("--plate-samples", type=int, default=32)
    ap.add_argument("--plate-frame", type=int, help="pose used for the backdrop shadow (default: middle frame)")
    ap.add_argument("--sss", action="store_true", help="keep skin subsurface scattering (~+15%% time)")
    ap.add_argument("--prefilter", default="FAST", choices=["FAST", "ACCURATE", "NONE"])
    ap.add_argument("--draft", action="store_true", help="4 samples at 2/3 resolution, upscaled (Lanczos)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="threads per process (0 = cores / workers)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--range", help="start:end (python slice) of the clip to render, e.g. 90:150")
    ap.add_argument("--bake-cache", help="pickle of the baked motion: reused if present (draft -> final), else written")
    ap.add_argument("--report", help="write per-frame timings (json)")
    ap.add_argument("--_worker", help=argparse.SUPPRESS)       # "k/N"
    ap.add_argument("--_dir", help=argparse.SUPPRESS)          # shared temp dir (bake.pkl, plate.exr, frames)
    args = ap.parse_args()

    worker = tuple(int(v) for v in args._worker.split("/")) if args._worker else None
    out_res = tuple(int(v) for v in args.res.split("x"))
    res, samples = out_res, args.samples
    if args.draft:
        res = (int(out_res[0] * 2 / 3) // 2 * 2, int(out_res[1] * 2 / 3) // 2 * 2)
        samples = min(samples, 4)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    t = time.time()
    rig, objs = mh_avatar.build_character(skin=args.skin, clothes=args.clothes, hair=args.hair)
    print(f"character built in {time.time() - t:.1f}s", flush=True)
    render_human.setup_scene(rig, res, samples)
    sc = bpy.context.scene
    cam = sc.camera
    if args.closeup:
        chest_z = np.array(rig.data.bones["spine_03"].head_local)[2]
        cam.data.lens = 85
        sens_h = cam.data.sensor_width * res[1] / res[0]
        cam.location = (0, -0.75 * cam.data.lens / sens_h, chest_z + 0.02)
    if args.faceup:
        head_z = np.array(rig.data.bones["head"].head_local)[2]
        sens_h = cam.data.sensor_width * res[1] / res[0]
        cam.location = (0, -0.30 * cam.data.lens / sens_h, head_z + 0.035)
    if args.audit_view:
        cal = json.load(open(os.path.join(ASL_V2, "models", "avatar_calib.json")))
        sw = cal["mp_shoulder_width_m"]
        mx, mz = cal["mp_mid_world_xz"]
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = 2.16 * sw * res[0] / res[1]
        cam.location = (mx, -5.0, mz - 0.30 * sw)
    fast_settings(sc, samples, sss=args.sss, prefilter=args.prefilter)
    ncpu = os.cpu_count() or 4
    threads = args.threads or max(1, ncpu // max(1, args.workers))
    if args.workers > 1 or args.threads:
        sc.render.threads_mode, sc.render.threads = "FIXED", threads

    retarget = mh_avatar.Retargeter(rig, objs)
    tmp = args._dir or tempfile.mkdtemp(prefix="aslfast_")
    bake_pkl = os.path.join(tmp, "bake.pkl")
    t = time.time()
    if worker:
        retarget.baked = pickle.load(open(bake_pkl, "rb"))
    elif args.bake_cache and os.path.exists(args.bake_cache):
        retarget.baked = pickle.load(open(args.bake_cache, "rb"))
        print(f"baked motion loaded from {args.bake_cache}", flush=True)
    else:
        seq = np.load(args.motion)
        base = args.motion[:-4]
        fit = dict(np.load(base + ".fit.npz")) if os.path.exists(base + ".fit.npz") else None
        seg = json.load(open(base + ".segments.json")) if os.path.exists(base + ".segments.json") else None
        retarget.bake(seq, seg, fit, objs)
        print(f"motion solved and baked in {time.time() - t:.1f}s", flush=True)
        if args.bake_cache:
            pickle.dump(retarget.baked, open(args.bake_cache, "wb"))
    n_total = len(retarget.baked["Q"])
    stills = [int(v) for v in args.frames.split(",")] if args.frames else None
    frames = stills if stills else list(range(n_total))
    if args.range and not stills:
        a, b = (int(v) if v else None for v in args.range.split(":"))
        frames = frames[a:b]
    prefix = os.path.splitext(args.output)[0]

    def frame_path(i):
        return f"{prefix}_{i:04d}.png" if stills else os.path.join(tmp, f"{i:05d}.png")

    mode = args.mode if args.mode != "auto" else ("plate" if len(frames) >= 12 else "direct")
    plate_path = os.path.join(tmp, "plate.exr")
    procs = []
    if not worker:
        if args.workers > 1 and len(frames) >= 2 * args.workers:
            pickle.dump(retarget.baked, open(bake_pkl, "wb"))
            for k in range(1, args.workers):                  # they build the character while we render the plate
                cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:]] + \
                      ["--_worker", f"{k}/{args.workers}", "--_dir", tmp, "--threads", str(threads)]
                procs.append(subprocess.Popen(cmd))
            worker_self = (0, args.workers)
        else:
            worker_self = None
        if mode == "plate":
            t = time.time()
            pf = args.plate_frame if args.plate_frame is not None else frames[len(frames) // 2]
            retarget.apply(pf, objs)
            render_plate(sc, objs, plate_path + ".tmp.exr", args.plate_samples)
            os.replace(plate_path + ".tmp.exr", plate_path)
            print(f"backdrop plate rendered in {time.time() - t:.1f}s", flush=True)
    else:
        worker_self = worker
        while mode == "plate" and not os.path.exists(plate_path):
            time.sleep(0.2)
    mask = use_plate(sc, plate_path) if mode == "plate" else None
    if not stills:
        sc.render.image_settings.compression = 1          # temp frames: zlib level 15 costs ~0.1 s each
    mine = [i for n, i in enumerate(frames) if worker_self is None or n % worker_self[1] == worker_self[0]]

    times = []
    stats = []
    if os.environ.get("PROFILE"):
        bpy.app.handlers.render_stats.append(lambda msg, *a: stats.append((time.time(), str(msg))))
    t_all = time.time()
    for n, i in enumerate(mine):
        t0, c0 = time.time(), sum(os.times()[:2])
        retarget.apply(i, objs)
        frac = set_border(sc, objs, mask) if mode == "plate" else 1.0
        sc.render.filepath = frame_path(i)
        t1 = time.time()
        stats.clear()
        bpy.ops.render.render(write_still=True)
        if stats:        # PROFILE=1: wall time between consecutive Cycles status messages
            ph, prev_t, prev_k = {"setup(apply+border+mask)": t1 - t0}, t1, "depsgraph"
            for ts, msg in stats + [(time.time(), "end")]:
                m = [q.strip() for q in msg.split("|") if not q.strip().startswith(("Mem:", "Remaining:", "Time:"))]
                k = ("sample" if m and m[0].startswith("Sample") else "finished->saved(compositor+png)" if m and m[0].startswith("Finished")
                     else "after-save" if msg.startswith("Time:") else "end" if msg == "end" else "sync+bvh+update")
                ph[prev_k] = ph.get(prev_k, 0) + ts - prev_t
                prev_t, prev_k = ts, k
            if os.environ.get("PROFILE") == "2":
                tt = [stats[0][0] - t1] + [b[0] - a[0] for a, b in zip(stats, stats[1:])]
                for dt, (_, msg) in zip(tt[1:], stats):
                    if dt > 0.01:
                        print(f"   {dt:.3f}s after '{msg}'")
            print("PROFILE", i, {k: round(v, 3) for k, v in ph.items()}, [m for _, m in stats if m.startswith("Time:")][-1:], flush=True)
        times.append(dict(frame=i, wall=time.time() - t0, cpu=sum(os.times()[:2]) - c0, box=frac))
        if n % 10 == 0 and not stills:
            print(f"[{worker_self}] frame {n + 1}/{len(mine)}  {(time.time() - t_all) / (n + 1):.2f}s/frame", flush=True)
    if args.report:
        tag = f".w{worker[0]}" if worker else ""
        json.dump(dict(mode=mode, samples=samples, res=res, frames=times), open(args.report + tag, "w"), indent=1)
    if worker:
        return
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("render worker failed")
    print(f"{len(frames)} frames rendered in {time.time() - t_all:.1f}s "
          f"({(time.time() - t_all) / len(frames):.2f}s/frame, mode={mode}, workers={args.workers})", flush=True)

    if stills:
        if res != out_res:
            for i in frames:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", frame_path(i), "-vf",
                                f"scale={out_res[0]}:{out_res[1]}:flags=lanczos", frame_path(i) + ".up.png"], check=True)
                os.replace(frame_path(i) + ".up.png", frame_path(i))
    else:
        vf = ["-vf", f"scale={out_res[0]}:{out_res[1]}:flags=lanczos"] if res != out_res else []
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-start_number", str(frames[0]), "-i", os.path.join(tmp, "%05d.png"), *vf, "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output], check=True)
    if not args._dir:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
