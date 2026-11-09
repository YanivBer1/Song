"""
Coarticulated phrase assembly: continue each sign from exactly where the
previous one visually ended.

Dictionary clips are isolated *citation* signs:  rest -> preparation ->
STROKE -> retraction -> rest.  Fluent signers drop the retraction and the
next preparation between words; the hand flows from the end of one stroke
straight into the start of the next (Kendon's gesture phases; Liddell &
Johnson's movement-hold model).

  1. segment_phases(): find each clip's stroke (first/last frame where the
     hands have arrived in signing space and settled), from wrist distance
     to the rest pose and wrist speed.
  2. Transitions are synthesised from boundary STATES, not frames of video:
     position and velocity at the end of stroke A and at the start of stroke
     B, joined by a quintic Hermite curve (matches position + velocity, zero
     acceleration at both ends -> no stop-and-go, no jerk spikes).
  3. Hand configuration changes *anticipate* the arrival (signers pre-shape
     the next handshape while moving): the new handshape is complete at ~75%
     of the movement; the hand rotates (slerp) rather than morphing points.
  4. A non-dominant hand that is idle for one short sign between two signs
     that use it hovers in a relaxed ready position instead of dropping to
     rest and coming back.
  5. Everything depends only on each sign's boundary states, so any word
     order / combination works without per-pair tuning.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .motion import (FACE, LH, RH, P, SIDES, _hand_turn, _orient, _travel, blend_hand, rest_arm,
                     rest_frame, retime, REST_HAND)

FPS = 30
SPEED = 1.15            # citation forms are slower than fluent signing
HOVER_MAX = 45          # frames: an idle gap up to this long keeps the weak hand hovering


def _minjerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def _wrists(clip):
    return np.stack([clip[:, LH], clip[:, RH]], 1)          # (T, 2, 3)


def segment_phases(clip, meta):
    """Return (stroke_start, stroke_end, active_hands) for a processed, untrimmed clip."""
    T = len(clip)
    active = {s: float(np.max(meta[f"weight_{s}"])) > 0.5 for s in "LR"}
    if not any(active.values()):
        return 0, T - 1, active
    rest = rest_frame(clip[0])
    d = np.zeros(T)
    for s, off in (("L", LH), ("R", RH)):
        if active[s]:
            d = np.maximum(d, np.linalg.norm(clip[:, off] - rest[off], axis=1) * meta[f"weight_{s}"])
    w = _wrists(clip)
    v = np.zeros(T)
    v[1:] = np.linalg.norm(np.diff(w, axis=0), axis=2).max(1)
    v = gaussian_filter1d(v, 1.0)
    thr = 0.6 * d.max()
    inside = np.flatnonzero(d > thr)
    first_in, last_in = int(inside[0]), int(inside[-1])
    # arrival: the hands settle after the preparation movement
    a0, a1 = first_in, min(first_in + 12, last_in)
    start = a0 if first_in <= 1 else a0 + int(np.argmin(v[a0:a1 + 1]))
    # departure: the last settle before the retraction movement
    b0, b1 = max(last_in - 12, start), last_in
    end = b1 if last_in >= T - 2 else b0 + int(np.argmin(v[b0:b1 + 1]))
    if end - start < 6:                       # very short signs: keep the whole in-space span
        start, end = first_in, last_in
    # Hold reduction (movement-hold model): citation signs freeze at their start/end; in
    # fluent signing those holds shrink to a brief contact, so the motion flows on.
    peak = v[start:end + 1].max() if end > start else 0.0
    if peak < 0.012:                          # static sign (number, letter): the hold IS the sign
        mid = (start + end) // 2
        active["static"] = True
        return max(start, mid - 3), min(end, mid + 3), active   # ~0.23 s hold
    still = v < 0.18 * peak
    keep = 2
    while start < end - 10 and still[start + keep]:
        start += 1
    while end > start + 10 and still[end - keep]:
        end -= 1
    # Coarticulation / target undershoot: fluent signers start the next movement while
    # the hand is still moving, so drop the citation form's final deceleration and
    # initial acceleration (speed < 25% of the sign's peak, at most 8 frames each).
    slow = v < 0.25 * peak
    for _ in range(8):
        if end - start > 12 and slow[end]:
            end -= 1
        if end - start > 12 and slow[start]:
            start += 1
    return start, end, active


def _hover_part(frame, side):
    """Relaxed 'ready' position for an idle hand: forearm forward at waist height."""
    s, e, w, _, off, sign = SIDES[side]
    sh = frame[P + s]
    el = sh + np.array([0.13 * sign, 0.80, 0.05])
    wr = el + np.array([-0.12 * sign, 0.22, -0.48])
    hand = wr + _orient(REST_HAND[side], wr - el)
    return el, wr, hand


def _set_side(frame, side, el, wr, hand):
    s, e, w, _, off, sign = SIDES[side]
    frame[P + e], frame[P + w] = el, wr
    frame[off:off + 21] = hand


def _velocity(clip, at_end):
    """Boundary velocity per frame, averaged over 4 frames (robust to tracker jitter)."""
    if len(clip) < 3:
        return np.zeros_like(clip[0])
    return (clip[-1] - clip[-3]) / 2 if at_end else (clip[2] - clip[0]) / 2


def hermite_transition(a, b, va, vb, n):
    """n in-between frames from state (a, va) to (b, vb). Velocities are per frame."""
    D = n + 1
    # keep boundary velocities from causing overshoot loops
    span = np.linalg.norm(b - a, axis=-1, keepdims=True)
    va, vb = va.copy(), vb.copy()
    for v in (va, vb):            # allow natural follow-through, but no wild loops
        mag = np.linalg.norm(v, axis=-1, keepdims=True) * D
        cap = 1.2 * span + 0.12
        v *= np.where(mag > cap, cap / (mag + 1e-9), 1.0)
    frames = []
    for k in range(1, n + 1):
        t = k / D
        h0 = 1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5
        h1 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
        h4 = -4 * t ** 3 + 7 * t ** 4 - 3 * t ** 5
        h5 = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
        f = h0 * a + h1 * va * D + h4 * vb * D + h5 * b
        # hands: wrist follows the Hermite path, the hand rotates + pre-shapes (anticipation)
        th = _minjerk(min(1.0, t / 0.75))
        for off in (LH, RH):
            f[off:off + 21] = f[off] + blend_hand(a[off:off + 21] - a[off], b[off:off + 21] - b[off], th)
        # face: smooth ease, no velocity carry (expressions change per sign)
        tf = _minjerk(t)
        f[FACE:] = (1 - tf) * a[FACE:] + tf * b[FACE:]
        frames.append(f)
    return np.stack(frames) if frames else np.zeros((0,) + a.shape)


def transition_frames(a, b):
    """Fluent transition length from wrist travel and hand rotation (~100-500 ms)."""
    return int(np.clip(round(3 + 5 * _travel(a, b) + 3 * _hand_turn(a, b) / np.pi), 4, 13))


def assemble(items, hold=8, speed=SPEED):
    """items: list of (processed_untrimmed_clip, meta) per word, in phrase order.
    Returns (seq, info) with info['segments'] = [(kind, start, end, word_index)]."""
    strokes, actives, static = [], [], []
    for clip, meta in items:
        s0, s1, act = segment_phases(clip, meta)
        strokes.append(retime(clip[s0:s1 + 1], speed))
        actives.append(act)
        static.append(bool(act.get("static")))

    # weak-hand hover across short idle gaps
    for i in range(1, len(strokes) - 1):
        for side in "LR":
            if (not actives[i].get(side) and actives[i - 1].get(side) and actives[i + 1].get(side)
                    and len(strokes[i]) <= HOVER_MAX):
                st = strokes[i].copy()
                for t in range(len(st)):
                    _set_side(st[t], side, *_hover_part(st[t], side))
                strokes[i] = st

    parts, segs, pos = [], [], 0

    def add(kind, arr, wi):
        nonlocal pos
        if len(arr):
            parts.append(arr)
            segs.append((kind, pos, pos + len(arr), wi))
            pos += len(arr)

    start_rest = rest_frame(strokes[0][0])
    add("hold", np.repeat(start_rest[None], hold, axis=0), -1)
    zero = np.zeros_like(start_rest)
    add("transition", hermite_transition(start_rest, strokes[0][0], zero, _velocity(strokes[0], False),
                                         max(8, transition_frames(start_rest, strokes[0][0]) + 4)), 0)
    for i, st in enumerate(strokes):
        add("stroke", st, i)
        if i + 1 < len(strokes):
            nxt = strokes[i + 1]
            add("transition", hermite_transition(st[-1], nxt[0], _velocity(st, True), _velocity(nxt, False),
                                                 transition_frames(st[-1], nxt[0])), i + 1)
    end_rest = rest_frame(strokes[-1][-1])
    add("transition", hermite_transition(strokes[-1][-1], end_rest, _velocity(strokes[-1], True), zero,
                                         max(10, transition_frames(strokes[-1][-1], end_rest) + 4)), -1)
    add("hold", np.repeat(end_rest[None], hold, axis=0), -1)
    seq = np.concatenate(parts)
    # light zero-phase smoothing (tracker jitter); hands less, to keep handshapes crisp
    seq[:, :LH] = gaussian_filter1d(seq[:, :LH], 1.0, axis=0, mode="nearest")
    seq[:, LH:FACE] = gaussian_filter1d(seq[:, LH:FACE], 0.7, axis=0, mode="nearest")
    seq[:, FACE:] = gaussian_filter1d(seq[:, FACE:], 1.0, axis=0, mode="nearest")
    return seq, {"segments": segs, "strokes": [len(s) for s in strokes], "static": static}


def junction_continuity(seq, segments, k=1.5):
    """% of stroke<->transition junctions without a velocity jump. A velocity
    discontinuity shows up as an acceleration spike, so a junction passes when its
    peak wrist acceleration is within k x the 95th percentile of acceleration inside
    the strokes themselves (real signing)."""
    w = np.stack([seq[:, LH], seq[:, RH]], 1)
    acc = np.zeros(len(seq))
    acc[1:-1] = np.linalg.norm(w[2:] - 2 * w[1:-1] + w[:-2], axis=2).max(1)
    stroke = np.zeros(len(seq), bool)
    for kind, a, b, _ in segments:
        if kind == "stroke":
            stroke[a + 2:b - 2] = True
    thr = max(k * np.percentile(acc[stroke], 95), 0.004) if stroke.any() else 0.01
    ok, n = 0, 0
    for kind, a, b, _ in segments:
        if kind != "transition":
            continue
        for j in (a, b):
            if 2 <= j <= len(seq) - 3:
                ok += acc[j - 1:j + 2].max() <= thr
                n += 1
    return 100.0 * ok / max(n, 1)


def transition_flow(seq, segments, min_ratio=0.15):
    """% of word->word transitions (rest lead-in/out excluded) in which the moving hand
    never comes to a stop: the slowest wrist speed across the junction window
    (3 frames before -> 3 after the transition) stays >= min_ratio (15%) x that transition's
    peak speed. Clip-by-clip stitching (hold, stop, go) fails this; fluent
    coarticulated signing passes."""
    w = np.stack([seq[:, LH], seq[:, RH]], 1)
    v = np.zeros((len(seq), 2))
    v[1:] = np.linalg.norm(np.diff(w, axis=0), axis=2)
    ok, n = 0, 0
    trans = [(a, b) for kind, a, b, _ in segments if kind == "transition"]
    for a, b in trans[1:-1]:                      # skip rest -> first sign and last sign -> rest
        lo, hi = max(a - 3, 1), min(b + 3, len(seq) - 1)
        hand = int(np.argmax(v[a:b].max(0))) if b > a else 1
        peak = v[a:b, hand].max() if b > a else 0
        if peak < 0.01:                           # no real movement needed between the signs
            ok += 1
        else:
            ok += v[lo:hi, hand].min() >= min_ratio * peak
        n += 1
    return 100.0 * ok / max(n, 1)


def junction_no_freeze(seq, segments, still=0.004, run=4, static=None):
    """% of word->word junctions with no visible freeze: no run of >= `run` consecutive
    frames where both wrists are essentially motionless (speed < `still` shoulder
    widths/frame ~ 1.4 mm/frame) within the window from 6 frames before the
    transition to 6 frames after it. Contact signs may decelerate onto the body;
    what reads as stop-and-go is the hand sitting frozen between words."""
    w = np.stack([seq[:, LH], seq[:, RH]], 1)
    v = np.zeros(len(seq))
    v[1:] = np.linalg.norm(np.diff(w, axis=0), axis=2).max(1)
    trans = [(a, b, wi) for kind, a, b, wi in segments if kind == "transition"][1:-1]
    if static is not None:   # a static sign's brief hold is its content, not a freeze
        trans = [(a, b, wi) for a, b, wi in trans if not (static[wi] or static[wi - 1])]
    ok = 0
    for a, b, _ in trans:
        lo, hi = max(a - 6, 1), min(b + 6, len(seq))
        frozen, longest = 0, 0
        for x in v[lo:hi] < still:
            frozen = frozen + 1 if x else 0
            longest = max(longest, frozen)
        ok += longest < run
    return 100.0 * ok / max(len(trans), 1)
