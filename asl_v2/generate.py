#!/usr/bin/env python3
"""
v2 text -> ASL video.

  python -m asl_v2.generate "Hi, my name is John" -o outputs/v2/john.mp4
  python -m asl_v2.generate "..." --renderer human -o outputs/v2/john_realistic.mp4

Sign lookup is the same as the Space (word2vec nearest neighbour over the
Space's sign dictionary); everything after that is replaced:
Holistic keypoints -> asl_v2.motion (rest hands, normalisation, gap filling,
eased transitions, smoothing) -> renderer:
  2d        illustrated 2D person (seconds)
  human     realistic MakeHuman character in Blender/Cycles (~7 s/frame on CPU)
  mannequin procedural Blender skin-modifier figure (experiment)
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "upstream"))


def lookup(words):
    os.chdir(HERE)  # upstream Vectorizer loads gensim-data/ relative to CWD
    from vectorizer import Vectorizer
    from local_backend import LocalVectorIndex

    vec = Vectorizer()
    index = LocalVectorIndex(vec.kv)
    matches = []
    for w in words:
        emb = vec.encode(w)
        if emb is None:
            print(f"skip '{w}': not in vocabulary")
            continue
        best = index.match(emb)[0]
        matches.append(dict(query=w, sign=best["word"], url=best["video_url"],
                            similarity=round(best["similarity"], 3)))
    return matches


def save_motion(path, seq, info):
    """Write the motion, its phrase segments and the avatar-space hand fit (same basename)."""
    from .handfit import fit_sequence
    np.save(path, seq.astype(np.float32))
    segs = [list(map(lambda v: v if isinstance(v, str) else int(v), s)) for s in info.get("segments", [])]
    json.dump(segs, open(path[:-4] + ".segments.json", "w"))
    np.savez_compressed(path[:-4] + ".fit.npz", **fit_sequence(seq, segs))


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALE = {"hundred", "thousand", "million", "billion", "trillion"}


def number_words(n):
    """0..999999 -> English number words (each word has a sign in the dictionary)."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10 - 2]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    return number_words(n // 1000) + ["thousand"] + (number_words(n % 1000) if n % 1000 else [])


def tokens(text):
    """Words to sign. Digits become number words and a currency/percent symbol becomes the
    word after the whole amount: "$7 billion" -> seven billion dollars, "50%" -> fifty percent."""
    out, pending = [], None
    for tok in text.split():
        unit = "dollars" if "$" in tok else "percent" if "%" in tok else None
        core = re.sub(r"[^\w\s.]", "", tok).lower().strip(".")
        if re.fullmatch(r"\d[\d,]*(\.\d+)?", core.replace(",", "")) or core.replace(",", "").isdigit():
            whole, _, frac = core.replace(",", "").partition(".")
            words = number_words(int(whole)) if int(whole) < 1000000 else [_ONES[int(d)] for d in whole]
            if frac:
                words += ["point"] + [_ONES[int(d)] for d in frac]
            out += words
            if unit:
                pending = unit
            continue
        word = re.sub(r"[^\w\s]", "", tok).lower().strip()
        if not word:
            continue
        if pending and word not in _SCALE:            # the amount has ended: say its unit
            out.append(pending)
            pending = None
        out.append(word)
        if unit and not pending and word != unit:
            out.append(unit)
    if pending:
        out.append(pending)
    return out


_PRIOR = None
_FIT_CACHE = {}


def tracked_keypoints(word, url=None, solver=True):
    """v3 tracking (crop-model hands with identity checks) in the keypoints() layout,
    plus the anatomical hand fit (asl_v2.kinematics) when solver=True."""
    global _PRIOR
    from .track import track
    t = track(word, url)
    from .track_rtm import augment, fused
    t = augment(word)                      # adds the second detector's hands (cached)
    fu = fused(t)                          # per-frame temporally consistent choice of the two
    kp = dict(pose=t["pose"], face=t["face"], lh=fu["img_L"], rh=fu["img_R"], size=t["size"], fps=t["fps"],
              world_lh=t["world_L"], world_rh=t["world_R"], noz_lh=fu["noz_L"], noz_rh=fu["noz_R"])
    if solver:
        from . import kinematics as K
        if word not in _FIT_CACHE:
            if _PRIOR is None:
                _PRIOR = K.load_prior()
            _FIT_CACHE[word] = K.fit_track(t, prior=_PRIOR)
        kp["fit"] = _FIT_CACHE[word]
        kp["track"] = t
    return kp


def build_motion(matches, tracking="v3", assembly="phrase", return_info=False):
    """assembly='phrase': coarticulated (strokes joined from exactly where each sign ends,
    velocity-continuous transitions); 'stitch': the earlier clip-level stitching."""
    from .extract import keypoints
    from .motion import process_clip, stitch
    get = tracked_keypoints if tracking == "v3" else keypoints
    if assembly == "phrase":
        from .phrase import assemble
        items = [process_clip(get(m["sign"], m["url"]), return_meta=True, trim=False) for m in matches]
        seq, info = assemble(items)
        return (seq, info) if return_info else seq
    seq = stitch([process_clip(get(m["sign"], m["url"])) for m in matches])
    return (seq, {"segments": stitch.last_segments}) if return_info else seq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="outputs/v2/asl.mp4")
    ap.add_argument("--renderer", choices=["2d", "human", "mannequin"], default="2d")
    ap.add_argument("--signs", help="comma-separated sign words; skips the word2vec lookup")
    ap.add_argument("--samples", type=int, default=16, help="Cycles samples (Blender renderers)")
    ap.add_argument("--res", default="960x540", help="Blender render resolution")
    args = ap.parse_args()
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.signs:
        from .extract import VIDEO_BASE
        matches = [dict(query=s, sign=s, url=f"{VIDEO_BASE}/{s}.mp4", similarity=None)
                   for s in args.signs.split(",")]
    else:
        matches = lookup(tokens(args.text))
    print(json.dumps(matches, indent=2))
    seq, info = build_motion(matches, return_info=True)
    print(f"motion: {len(seq)} frames ({len(seq) / 30:.1f}s)")

    if args.renderer == "2d":
        from .render2d import write_video
        write_video(seq, out)
    else:
        motion_file = os.path.splitext(out)[0] + ".motion.npy"
        save_motion(motion_file, seq, info)
        blender_py = os.path.join(HERE, ".venv-blender", "bin", "python")
        script = "render_human.py" if args.renderer == "human" else "render3d.py"
        subprocess.run([blender_py, os.path.join(HERE, "asl_v2", script), motion_file, out,
                        "--samples", str(args.samples), "--res", args.res], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
