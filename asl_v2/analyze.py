"""
Frame-by-frame analysis of tracked sign clips.

For every frame and each hand it reports: tracking source/reliability, every
finger joint angle (MCP/PIP/DIP flexion, MCP spread, thumb CMC/MCP/IP),
palm facing, hand location relative to the body (forehead, eyes, mouth/chin,
neck, chest, belly, side, out of frame), speed, hand-over-hand overlap with
depth order, and likely contact (hand held still at a body location).

  python -m asl_v2.analyze name john dollars -o docs/analysis   # JSON + markdown
"""

import argparse
import json
import os

import numpy as np

from .track import track

FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _angle(u, v):
    c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def joint_angles(h):
    """h: (21,3) landmarks (world metres preferred). Flexion = deviation from straight."""
    out = {}
    for name, (a, b, c, d) in FINGERS.items():
        base = h[0] if name != "thumb" else h[0]
        segs = [h[a] - base, h[b] - h[a], h[c] - h[b], h[d] - h[c]]
        if name == "thumb":
            out["thumb_cmc"] = _angle(h[a] - h[0], h[b] - h[a])
            out["thumb_mcp"] = _angle(segs[1], segs[2])
            out["thumb_ip"] = _angle(segs[2], segs[3])
        else:
            out[f"{name}_mcp"] = _angle(segs[0], segs[1])
            out[f"{name}_pip"] = _angle(segs[1], segs[2])
            out[f"{name}_dip"] = _angle(segs[2], segs[3])
    # spread between neighbouring fingers' proximal phalanges
    for f1, f2 in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        out[f"spread_{f1}_{f2}"] = _angle(h[FINGERS[f1][1]] - h[FINGERS[f1][0]], h[FINGERS[f2][1]] - h[FINGERS[f2][0]])
    out["thumb_index_spread"] = _angle(h[2] - h[1], h[6] - h[5])
    return {k: float(np.round(v, 1)) for k, v in out.items()}


def palm_facing(h, side):
    """Palm normal in camera axes -> coarse facing label."""
    n = np.cross(h[5] - h[0], h[17] - h[0])
    if side == "L":
        n = -n
    n /= np.linalg.norm(n) + 1e-9
    labels = {"camera": np.array([0, 0, -1.0]), "away": np.array([0, 0, 1.0]),
              "up": np.array([0, -1.0, 0]), "down": np.array([0, 1.0, 0]),
              "signer's left": np.array([1.0, 0, 0]), "signer's right": np.array([-1.0, 0, 0])}
    best = max(labels, key=lambda k: np.dot(n, labels[k]))
    return best, n


def location(p, pose, face):
    """Location of a 2D point relative to body landmarks (image px)."""
    ls, rs = pose[11, :2], pose[12, :2]
    sw = np.linalg.norm(ls - rs) + 1e-9
    mid = (ls + rs) / 2
    x = (p[0] - mid[0]) / sw
    if not np.isnan(face[0, 0]):
        top, chin = face[10, :2], face[152, :2]
        eyes = (face[33, 1] + face[263, 1]) / 2
        mouth = face[13, 1]
        fx = (p[0] - face[1, 0]) / sw
        if abs(fx) < 0.45:
            if top[1] - 0.15 * sw <= p[1] < eyes - 0.03 * sw:
                return "forehead"
            if eyes - 0.03 * sw <= p[1] < mouth - 0.05 * sw:
                return "eyes/nose"
            if mouth - 0.05 * sw <= p[1] < chin[1] + 0.08 * sw:
                return "mouth/chin"
        if abs(fx) < 0.75 and top[1] <= p[1] < chin[1] + 0.08 * sw:
            return "side of head"
    y = (p[1] - mid[1]) / sw
    if y < -0.3:
        return "above shoulders"
    if y < 0.15 and abs(x) < 0.35:
        return "neck"
    if abs(x) > 0.75:
        return "side/neutral space"
    if y < 0.75:
        return "chest"
    if y < 1.3:
        return "belly"
    return "low/rest"


def analyze_clip(word):
    t = track(word)
    T = len(t["pose"])
    fps = float(t["fps"])
    frames = []
    for i in range(T):
        rec = {"frame": i, "time_s": round(i / fps, 3), "hands_overlap": bool(t["occl"][i])}
        palms = {}
        for s in "LR":
            src = int(t[f"src_{s}"][i])
            img = t[f"img_{s}"][i]
            world = t[f"world_{s}"][i]
            if src == 0:
                rec[s] = {"visible": False}
                continue
            geom = world if not np.isnan(world[0, 0]) else img
            facing, n = palm_facing(geom, s)
            palms[s] = img[[0, 5, 9, 13, 17], :2].mean(0)
            span = float(np.ptp(img[:, 1]) + np.ptp(img[:, 0]))
            prev = t[f"img_{s}"][i - 1] if i else img
            speed = float(np.nanmean(np.linalg.norm(img[:, :2] - prev[:, :2], axis=1)) * fps) if i else 0.0
            rec[s] = {
                "visible": True,
                "tracking": {1: "holistic only (coarse)", 2: "hand model", 3: "hand model + holistic agree"}[src],
                "confidence": round(float(t[f"conf_{s}"][i]), 2),
                "angles_deg": joint_angles(geom),
                "angles_from": "metric 3D" if geom is world else "image (2.5D)",
                "palm_facing": facing,
                "location": location(palms[s], t["pose"][i], t["face"][i]),
                "speed_px_s": round(speed, 1),
                "hand_size_px": round(span / 2, 1),
            }
        if len(palms) == 2:
            d = float(np.linalg.norm(palms["L"] - palms["R"]))
            rec["hand_distance_px"] = round(d, 1)
            if rec["hands_overlap"]:
                # the hand tracked by the dedicated model (src>=2) in an overlap frame is the visible / front one
                front = [s for s in "LR" if int(t[f"src_{s}"][i]) >= 2]
                rec["front_hand"] = front[0] if len(front) == 1 else "both visible"
        frames.append(rec)
    # contact / hold detection: low speed for >= 4 frames at a body location
    for s in "LR":
        sp = np.array([f[s]["speed_px_s"] if f[s].get("visible") else np.nan for f in frames])
        for i, f in enumerate(frames):
            if f[s].get("visible"):
                win = sp[max(0, i - 2): i + 3]
                held = np.nanmax(win) < 0.6 * f[s]["hand_size_px"] * 30 / 10 if np.isfinite(win).any() else False
                f[s]["held"] = bool(held)
                f[s]["likely_contact"] = bool(held and f[s]["location"] in
                                              ("forehead", "eyes/nose", "mouth/chin", "neck", "chest", "belly", "side of head"))
    return {"word": word, "fps": fps, "frames": frames}


def summarize(an):
    fr = an["frames"]
    lines = [f"### {an['word'].upper()} ({len(fr)} frames, {len(fr) / an['fps']:.2f} s)", ""]
    for s, name in (("L", "left (non-dominant)"), ("R", "right (dominant)")):
        vis = [f for f in fr if f[s].get("visible")]
        if not vis:
            lines.append(f"- **{name} hand:** never in frame; it hangs at rest below the crop.")
            continue
        locs = {}
        for f in vis:
            locs[f[s]["location"]] = locs.get(f[s]["location"], 0) + 1
        facing = {}
        for f in vis:
            facing[f[s]["palm_facing"]] = facing.get(f[s]["palm_facing"], 0) + 1
        src3 = sum(f[s]["tracking"].startswith("hand model") for f in vis)
        ang = {k: np.array([f[s]["angles_deg"][k] for f in vis]) for k in vis[0][s]["angles_deg"]}
        rng = ", ".join(f"{k} {v.min():.0f}–{v.max():.0f}°" for k, v in ang.items()
                        if k.endswith(("_pip", "_ip")) or k.startswith("spread_index"))
        contact = sum(f[s]["likely_contact"] for f in vis)
        lines.append(f"- **{name} hand:** visible {len(vis)}/{len(fr)} frames; precise hand model in {src3}; "
                     f"locations {dict(sorted(locs.items(), key=lambda x: -x[1]))}; "
                     f"palm faces {dict(sorted(facing.items(), key=lambda x: -x[1]))}; "
                     f"likely contact/hold frames {contact}.")
        lines.append(f"  - flexion ranges: {rng}")
    ov = [f for f in fr if f["hands_overlap"]]
    if ov:
        fronts = {}
        for f in ov:
            fronts[f.get("front_hand", "?")] = fronts.get(f.get("front_hand", "?"), 0) + 1
        lines.append(f"- **hand-over-hand:** {len(ov)} frames overlap; front hand per frame {fronts}.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="+")
    ap.add_argument("-o", "--outdir", default="docs/analysis")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    md = ["# Frame-by-frame analysis of the source sign clips", ""]
    for w in a.words:
        an = analyze_clip(w)
        with open(os.path.join(a.outdir, f"{w}.json"), "w") as fh:
            json.dump(an, fh, indent=1)
        md.append(summarize(an))
    with open(os.path.join(a.outdir, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
