"""
Tests + metrics for asl_v2.kinematics.

    .venv/bin/python -m pytest asl_v2/tests -q          (if pytest is installed)
    .venv/bin/python asl_v2/tests/test_kinematics.py    (plain script: runs the
                                                        tests and prints the table)

Metrics per cached clip (cache/tracks/*.npz), "before" = raw MediaPipe
landmarks, "after" = fitted skeleton:
  (a) bone-length coefficient of variation (mean over the 20 bones)
  (b) % frames with any joint angle outside the range of motion
      (before: analytic inverse kinematics of the raw metric landmarks)
  (c) reprojection error of the fitted joints against confident (src>=2)
      image landmarks: px, fraction of hand size (max side of the landmark
      bounding box) and PCK@0.1
  (d) RMS angular jerk (deg/s^3) over runs of consecutive confident frames
  (e) behaviour on the NAME / DOLLARS hand-over-hand frames
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from asl_v2 import kinematics as K                      # noqa: E402

TRACKS = os.path.join(ROOT, "cache", "tracks")
_CACHE = {}


def clips():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(TRACKS, "*.npz")))


def tracks():
    if "tracks" not in _CACHE:
        _CACHE["tracks"] = {n: dict(np.load(os.path.join(TRACKS, n + ".npz"))) for n in clips()}
    return _CACHE["tracks"]


def fits():
    if "fits" not in _CACHE:
        _CACHE["fits"] = K.fit_clips(tracks(), prior=K.load_prior(), workers=4)
    return _CACHE["fits"]


def _bone_cov(J):
    L = np.stack([np.linalg.norm(J[:, a] - J[:, b], axis=1) for a, b in K.BONES], 1)
    return float((L.std(0) / L.mean(0)).mean())


def _jerk(A, mask, fps):
    """RMS third difference (deg/s^3) over runs of >= 4 consecutive masked frames."""
    out = []
    for s, e in K._runs(mask):
        if e - s >= 4:
            out.append(np.diff(np.degrees(A[s:e]), 3, axis=0) * fps ** 3)
    return float(np.sqrt(np.mean(np.concatenate(out) ** 2))) if out else np.nan


def hand_metrics(tr, fit, S):
    src = tr[f"src_{S}"].astype(int)
    conf = src >= 2
    if conf.sum() < 4:
        return None
    fps = float(tr["fps"])
    W, I = tr[f"world_{S}"].astype(float), tr[f"img_{S}"].astype(float)
    raw, _ = K.inverse_kinematics(W, fit["bones"], S)
    m = dict(n=int(conf.sum()), n_all=int((src > 0).sum()))
    m["cov_before"] = _bone_cov(W[conf])
    m["cov_after"] = _bone_cov(fit["joints"])
    m["lim_before"] = 100 * K.outside_limits(raw[conf]).any(1).mean()
    m["lim_after"] = 100 * K.outside_limits(fit["angles"], tol_deg=1e-6).any(1).mean()
    proj = fit["wrist_px"][:, None] + fit["px_per_m"][:, None, None] * fit["joints"][:, :, :2]
    e = np.linalg.norm(proj - I[:, :, :2], axis=2)
    hs = np.maximum(np.ptp(I[:, :, 0], axis=1), np.ptp(I[:, :, 1], axis=1))[:, None]
    m["err_px"], m["err_frac"], m["hs"] = e[conf], (e / hs)[conf], float(np.median(hs[conf]))
    m["err_frac3"] = (e / hs)[src == 3]
    m["err_frac_acc"] = (e / hs)[conf & (fit["quality"] >= 0.2)]     # frames the fitter did not reject
    m["rejected"] = int((conf & (fit["quality"] < 0.2)).sum())
    m["jerk_before"] = _jerk(raw, conf, fps)
    m["jerk_after"] = _jerk(fit["angles"], conf, fps)
    return m


def all_metrics():
    if "metrics" not in _CACHE:
        out = {}
        for n, tr in tracks().items():
            for S in "LR":
                m = hand_metrics(tr, fits()[n][S], S)
                if m:
                    out[(n, S)] = m
        _CACHE["metrics"] = out
    return _CACHE["metrics"]


def body_frame(n):
    """Both fitted hands of a clip in canonical units (both wrists at depth 0)."""
    tr, f = tracks()[n], fits()[n]
    sh = tr["pose"][:, [11, 12], :2]
    center = np.median(sh.mean(1), axis=0)
    scale = np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1))
    JL, uL = K.place_in_body_frame(f["L"], center, scale)
    JR, uR = K.place_in_body_frame(f["R"], center, scale)
    return JL, JR, 0.5 * (uL + uR)


def occlusion_report(n):
    tr, f = tracks()[n], fits()[n]
    occl = tr["occl"] > 0
    JL, JR, upm = body_frame(n)
    over = (tr["occl"] > 0) | K.image_overlap(JL, JR, upm)
    front = K.front_hand_from_evidence(tr["src_L"], tr["src_R"], tr["conf_L"], tr["conf_R"], over)
    contact = K.detect_contact(JL, JR, tr["occl"], units_per_m=upm)
    res = K.resolve_contacts(JL, JR, front=front, contact=contact, units_per_m=upm)
    pen0 = K.penetration(JL, JR, units_per_m=upm)
    pen1 = K.penetration(res["L"], res["R"], units_per_m=upm)
    rep = dict(n_occl=int(occl.sum()), front=front, contact=contact, pen_before=pen0, pen_after=pen1,
               upm=upm, res=res, occl=occl, over=over)
    for S in "LR":
        a = f[S]["angles"]
        step = np.degrees(np.abs(np.diff(a, axis=0))).max(1)
        o = occl[1:] | occl[:-1]
        rep[S] = dict(finite=bool(np.isfinite(f[S]["joints"][occl]).all()),
                      low_src=int(((tr[f"src_{S}"] < 2) & occl).sum()),
                      in_limits=bool(not K.outside_limits(a[occl], 1e-6).any()),
                      step_occl=float(step[o].max()) if o.any() else np.nan,
                      step_rest=float(step[~o].max()) if (~o).any() else np.nan,
                      quality_occl=float(f[S]["quality"][occl].mean()),
                      quality_rest=float(f[S]["quality"][~occl & (tr[f"src_{S}"] >= 2)].mean()))
    return rep


# ----------------------------------------------------------------- unit tests
def _random_angles(rng, n, frac=0.8):
    mid, half = (K.LIMITS_LO + K.LIMITS_HI) / 2, (K.LIMITS_HI - K.LIMITS_LO) / 2
    a = mid + frac * half * rng.uniform(-1, 1, (n, K.D))
    a[:, 20] = 0.0
    a[:, [4, 8, 12, 16]] = np.clip(a[:, [4, 8, 12, 16]], None, np.radians(60))   # abduction is ill-defined at 90 deg
    return a


def test_forward_inverse_roundtrip():
    rng = np.random.default_rng(0)
    a = _random_angles(rng, 50)
    R = K._exp(rng.normal(size=(50, 3)))
    for side in "RL":
        bones = dict(K.DEFAULT_BONES, side=side)
        Rs = R if side == "R" else R * np.outer(K._MIRROR, K._MIRROR)
        J = K.forward(a, Rs, bones)
        assert J.shape == (50, 21, 3) and np.allclose(J[:, 0], 0)
        a2, R2 = K.inverse_kinematics(J, bones, side)
        assert np.abs(a2 - a).max() < 1e-6, np.abs(a2 - a).max()
        assert np.abs(R2 - Rs).max() < 1e-6
        assert np.allclose(np.linalg.det(R2), 1)
        assert np.allclose(K.forward(a[3], Rs[3], bones), J[3])
    # a left hand is the mirror image of the right hand with the same angles
    JL = K.forward(a, None, dict(K.DEFAULT_BONES, side="L"))
    JR = K.forward(a, None, dict(K.DEFAULT_BONES, side="R"))
    assert np.allclose(JL, JR * K._MIRROR)


def test_biomechanics_of_model():
    b = K.DEFAULT_BONES
    flat = K.forward(np.zeros(K.D), None, b)
    a = np.zeros(K.D)
    a[4:8] = np.radians([60, 0, 40, 20])                      # curl the index finger
    J = K.forward(a, None, b)
    assert J[8, 2] > flat[8, 2] + 0.02                        # flexion moves the tip palmar (+z)
    n = np.cross(J[6] - J[5], J[7] - J[6])
    assert abs(n @ (J[8] - J[7])) < 1e-9                      # PIP/DIP hinges: the finger stays planar
    assert abs(n[2]) / np.linalg.norm(n) < 1e-6               # flexion axis lies in the palm plane
    a = np.zeros(K.D)
    a[5] = np.radians(15)
    assert K.forward(a, None, b)[8, 0] > flat[8, 0]           # +abduction = radial deviation
    a = np.zeros(K.D)
    a[20] = np.radians(15)
    J = K.forward(a, None, b)
    assert J[17, 2] > flat[17, 2] and J[13, 2] > flat[13, 2] and np.allclose(J[[5, 9]], flat[[5, 9]])
    x_thumb = K.THUMB_REST[:, 0]                              # saddle joint: axes oblique to the palm
    assert 0.2 < abs(x_thumb[2]) < 0.98
    assert (K.LIMITS_LO < K.LIMITS_HI).all() and len(K.ANGLE_NAMES) == K.D


def test_fit_recovers_synthetic_sequence():
    rng = np.random.default_rng(1)
    T = 40
    k = _random_angles(rng, 4, 0.6)
    u = np.linspace(0, 3, T)
    a = np.stack([np.interp(u, np.arange(4), k[:, j]) for j in range(K.D)], 1)
    R = K._exp(np.linspace(0, 1, T)[:, None] * np.array([0.6, -0.9, 0.4]) + np.array([0.2, 0.1, 2.0]))
    for side in "RL":
        bones = dict(K.DEFAULT_BONES, side=side)
        J = K.forward(a, R, bones)
        world = J + rng.normal(0, 0.0008, J.shape) + np.array([0.01, -0.02, 0.0])
        img = np.zeros((T, 21, 3))
        img[:, :, :2] = 300 * J[:, :, :2] + np.array([320.0, 180.0]) + rng.normal(0, 0.3, (T, 21, 2))
        src = np.full(T, 3)
        src[15:19] = 0                                        # a dropout ...
        world[15:19] = np.nan
        img[15:19] = np.nan
        src[25:28] = 1                                        # ... and a Holistic-only stretch
        world[25:28] = np.nan
        img[25:28, :, 2] = 300 * (J[25:28, :, 2] - J[25:28, :1, 2])
        f = K.fit_hand_sequence(img, world, np.ones(T), src, np.zeros(T), side=side, bones=bones)
        assert np.isfinite(f["angles"]).all() and np.isfinite(f["joints"]).all()
        err = np.linalg.norm(f["joints"] - J, axis=2)
        assert err[src == 3].mean() < 0.004, err[src == 3].mean()
        assert err[15:19].mean() < 0.012, err[15:19].mean()   # gap filled in angle space
        assert not K.outside_limits(f["angles"], 1e-6).any()
        assert _bone_cov(f["joints"]) < 1e-3                  # only the arching 4th/5th knuckles move
        seg = np.linalg.norm(f["joints"][:, K.CHAIN[:, 1:]] - f["joints"][:, K.CHAIN[:, :-1]], axis=-1)
        assert np.allclose(seg, bones["lengths"], atol=1e-12)  # phalanges exactly constant
        assert (f["quality"][15:19] == 0.25).all() and f["quality"][src == 3].min() > f["quality"][25:28].max()
        proj = f["wrist_px"][:, None] + f["px_per_m"][:, None, None] * f["joints"][:, :, :2]
        assert np.abs(proj - img[:, :, :2])[src == 3].mean() < 1.5
        assert K.detect_side(world) == side
        g = K.fit_hand_sequence(img, world, np.ones(T), src, side=None)
        assert g["side"] == side


def test_prior():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 3)) * [0.8, 0.4, 0.2]
    B = np.linalg.qr(rng.normal(size=(K.D, 3)))[0].T
    A = 0.3 + Z @ B + rng.normal(0, 0.02, (500, K.D))
    p = K.train_angle_prior([A[:250], A[250:]], n_components=3)
    assert p.residuals(A[:5]).shape == (5, K.D + 3)
    x = A[7]
    num = np.array([(p.neg_log_lik(x + 1e-6 * e) - p.neg_log_lik(x - 1e-6 * e)) / 2e-6 for e in np.eye(K.D)])
    assert np.allclose(num, p.grad(x), atol=1e-4)
    assert p.neg_log_lik(p.mean) < p.neg_log_lik(p.mean + 0.5 * B[0]) < p.neg_log_lik(p.mean + 0.5 * B[2])
    q = K.AnglePrior.from_dict(p.to_dict())
    assert np.allclose(q.residuals(A[:3]), p.residuals(A[:3]))
    saved = K.load_prior()
    if saved is not None:
        assert saved.mean.shape == (K.D,) and np.isfinite(saved.residuals(np.zeros(K.D))).all()
        assert (saved.mean >= K.LIMITS_LO).all() and (saved.mean <= K.LIMITS_HI).all()


def test_stabilize_handshapes():
    rng = np.random.default_rng(3)
    a = np.zeros((40, K.D))
    a[20:] = 0.8
    a[18:22] = np.linspace(0, 0.8, 6)[1:5, None]
    noisy = a + rng.normal(0, 0.03, a.shape)
    w = np.ones(40)
    w[5:9] = 0.05
    noisy[5:9] += 0.05
    out, static = K.stabilize_handshapes(noisy, w)
    assert static[:15].all() and static[25:].all()
    assert np.abs(out - a)[5:9].mean() < 0.5 * np.abs(noisy - a)[5:9].mean()
    assert np.abs(out - a).mean() < np.abs(noisy - a).mean()


def test_resolve_contacts_synthetic():
    b = K.DEFAULT_BONES
    upm = 1.0
    flat = K.forward(np.zeros(K.D), None, b)
    T = 6
    A = np.repeat(flat[None], T, 0)
    Bh = A + np.array([0.01, 0.0, 0.0])
    Bh[:, :, 2] += np.array([0.0, 0.003, -0.004, 0.02, 0.2, 0.0])[:, None]   # last: far apart sideways
    Bh[5, :, 0] += 0.5
    assert (K.penetration(A, Bh, units_per_m=upm)[:4] > 0).all()
    front = np.array([1, 1, 1, 1, 1, 0])                      # right hand (Bh) in front
    contact = np.array([0, 0, 0, 0, 1, 0])
    r = K.resolve_contacts(A, Bh, front=front, contact=contact, units_per_m=upm, smooth=0, max_snap=1.0)
    pen = K.penetration(r["L"], r["R"], units_per_m=upm)
    assert (pen < 1e-9).all(), pen
    assert np.abs(pen[:5]).max() < 1e-6                       # touching: no gap, no penetration
    assert (r["R"][:5, :, 2].mean(1) < r["L"][:5, :, 2].mean(1)).all()        # right hand ended up in front
    assert np.allclose(r["L"][:, :, :2], A[:, :, :2]) and r["dz_L"][5] == 0 and r["dz_R"][5] == 0
    assert np.allclose(r["dz_L"], -r["dz_R"])
    # no evidence: keep the existing order, move as little as possible
    r0 = K.resolve_contacts(A[1], Bh[1], units_per_m=upm, smooth=0)
    assert K.penetration(r0["L"], r0["R"], units_per_m=upm)[0] < 1e-9
    forced = K.resolve_contacts(A[1], Bh[1], front=-r0["front"], units_per_m=upm, smooth=0)
    assert abs(r0["dz_L"][0]) <= abs(forced["dz_L"][0]) + 1e-12
    # temporal smoothing never re-introduces penetration
    r2 = K.resolve_contacts(A, Bh, front=front, units_per_m=upm, smooth=2)
    assert (K.penetration(r2["L"], r2["R"], units_per_m=upm) < 1e-9).all()
    ev = K.front_hand_from_evidence([3, 3, 3, 3], [1, 1, 0, 3], [1, 1, 1, 1], [.5, .5, 0, 1], [1, 1, 1, 0], bridge=0)
    assert list(ev) == [-1, -1, -1, 0]


def test_keep_in_front_synthetic():
    upm = 1.0
    flat = K.forward(np.zeros(K.D), None, K.DEFAULT_BONES)
    torso = dict(type="ellipsoid", c=np.array([0.0, 0.05, 0.3]), radii=np.array([0.2, 0.3, 0.12]))
    neck = dict(type="capsule", a=np.array([0.0, -0.3, 0.3]), b=np.array([0.0, -0.2, 0.3]), r=0.05)
    inside = flat + np.array([0.0, 0.0, 0.25])
    J, dz = K.keep_in_front(inside, [torso, neck], units_per_m=upm, smooth=0)
    assert dz < 0
    S, r = K.hand_spheres(J, units_per_m=upm)
    d = np.sqrt((((S - torso["c"]) / (torso["radii"] + r[:, None])) ** 2).sum(1))
    assert d.min() > 1 - 1e-6 and d.min() < 1.02               # touching the chest, not floating
    J2, dz2 = K.keep_in_front(flat - np.array([0, 0, 0.3]), [torso, neck], units_per_m=upm, smooth=0)
    assert dz2 == 0 and np.allclose(J2, flat - np.array([0, 0, 0.3]))
    J3, dz3 = K.keep_in_front(np.repeat(inside[None], 4, 0), [torso, neck], units_per_m=upm)
    assert J3.shape == (4, 21, 3) and (dz3 <= dz + 1e-9).all()
    P = np.zeros((33, 3))
    P[11], P[12], P[7], P[8] = [0.5, 0, 0], [-0.5, 0, 0], [0.2, -0.6, 0], [-0.2, -0.6, 0]
    assert len(K.body_proxies(P)) == 3
    _, dz4 = K.keep_in_front(flat * 2.6 + [0, 0.5, 0.1], K.body_proxies(P))
    assert dz4 < 0


# ------------------------------------------------------------ data-set tests
def test_bone_lengths_constant():
    for (n, S), m in all_metrics().items():
        assert m["cov_after"] < 1e-3, (n, S, m["cov_after"])   # palm arch moves knuckles 13/17 slightly
        assert m["cov_before"] > 10 * m["cov_after"]


def test_joint_limits_respected():
    for n, f in fits().items():
        for S in "LR":
            a = f[S]["angles"]
            a = a[np.isfinite(a).all(1)]
            assert not K.outside_limits(a, 1e-6).any(), (n, S)


def test_outputs_complete():
    for n, tr in tracks().items():
        for S in "LR":
            f, seen = fits()[n][S], tr[f"src_{S}"] > 0
            if not seen.any():
                assert np.isnan(f["angles"]).all() and (f["quality"] == 0).all()
                continue
            for k in ("angles", "joints", "orient", "wrist_px", "px_per_m"):   # gaps filled everywhere
                assert np.isfinite(f[k]).all(), (n, S, k)
            assert np.allclose(np.linalg.det(f["orient"]), 1, atol=1e-6)
            assert np.allclose(K.forward(f["angles"], f["orient"], f["bones"]), f["joints"], atol=1e-9)
            assert (f["quality"] >= 0).all() and (f["quality"] <= 1).all()
            lo = tr[f"src_{S}"] == 1
            if lo.any() and (tr[f"src_{S}"] == 3).any():       # Holistic-only frames are down-weighted
                assert f["quality"][lo].mean() < f["quality"][tr[f"src_{S}"] == 3].mean()


def test_reprojection():
    M = all_metrics()
    best = np.concatenate([m["err_frac3"] for m in M.values()])
    acc = np.concatenate([m["err_frac_acc"] for m in M.values()])
    every = np.concatenate([m["err_frac"] for m in M.values()])
    assert (acc < 0.1).mean() >= 0.95, (acc < 0.1).mean()      # PCK@0.1 on confident, non-rejected frames
    assert (best < 0.1).mean() >= 0.95, (best < 0.1).mean()    # ... and on the best (src == 3) frames
    assert (every < 0.1).mean() >= 0.90


def test_jerk_reduced():
    for (n, S), m in all_metrics().items():
        if np.isfinite(m["jerk_before"]):
            assert m["jerk_after"] < 0.5 * m["jerk_before"], (n, S, m["jerk_before"], m["jerk_after"])


def test_occlusion_name_dollars():
    for n in ("name", "dollars"):
        if n not in tracks():
            continue
        r = occlusion_report(n)
        assert r["n_occl"] > 0
        for S in "LR":
            assert r[S]["finite"] and r[S]["in_limits"]
            assert r[S]["step_occl"] < 45.0                    # no handshape pops while occluded (deg / frame)
            assert r[S]["quality_occl"] < r[S]["quality_rest"]  # occluded frames are down-weighted
        assert np.abs(np.diff(r["res"]["dz_L"])).max() * 1000 / r["upm"] < 40.0   # no depth pops (mm / frame)
        assert (r["pen_after"] < 1e-6).all()                   # hands never interpenetrate ...
        c = r["contact"] & r["res"]["overlap"]
        if c.any():                                            # ... and stay in touch where they were touching
            assert np.nanmedian(r["res"]["gap"][c]) * 1000 / r["upm"] < 10.0
        assert (r["front"][r["over"]] != 0).any()


# ----------------------------------------------------------------- reporting
def print_table():
    M = all_metrics()
    names = sorted({n for n, _ in M})
    print("\nPer clip (hands pooled). CoV = bone-length coefficient of variation; lim = % frames with a joint "
          "outside its range;\nreproj = fitted joints vs confident (src>=2) image landmarks; hs = hand size "
          "(bbox max side); PCK3 = src==3 only;\nPCKacc = excluding confident frames the fitter rejected as "
          "glitches (quality<0.2, count in 'rej'); jerk in 1e3 deg/s^3.\n")
    hdr = ("clip", "frames", "CoV bef", "CoV aft", "lim bef%", "lim aft%", "px mean", "px p95", "frac mean",
           "frac p95", "PCK@.1", "PCK3", "PCKacc", "rej", "jerk bef", "jerk aft")
    print(" ".join(f"{h:>9}" for h in hdr))
    pool = {k: [] for k in ("err_px", "err_frac", "err_frac3", "err_frac_acc")}
    tot = []
    for n in names + ["ALL"]:
        ms = [m for (c, _), m in M.items() if c == n or n == "ALL"]
        w = np.array([m["n"] for m in ms], float)
        avg = lambda k: float(np.nansum([m[k] * m["n"] for m in ms]) / w.sum())
        cat = lambda k: np.concatenate([m[k] for m in ms])
        px, fr, f3, fa = cat("err_px"), cat("err_frac"), cat("err_frac3"), cat("err_frac_acc")
        pck = lambda x: 100 * (x < 0.1).mean() if x.size else np.nan
        row = (n, int(w.sum()), avg("cov_before"), avg("cov_after"), avg("lim_before"), avg("lim_after"),
               px.mean(), np.percentile(px, 95), fr.mean(), np.percentile(fr, 95), pck(fr), pck(f3), pck(fa),
               sum(m["rejected"] for m in ms), avg("jerk_before") / 1e3, avg("jerk_after") / 1e3)
        fmt = ["{:>9}", "{:>9d}", "{:>9.4f}", "{:>9.1e}", "{:>9.1f}", "{:>9.1f}", "{:>9.2f}", "{:>9.2f}",
               "{:>9.3f}", "{:>9.3f}", "{:>9.1f}", "{:>9.1f}", "{:>9.1f}", "{:>9d}", "{:>9.0f}", "{:>9.0f}"]
        print(" ".join(f.format(v) for f, v in zip(fmt, row)))
    print("\n(e) hand-over-hand frames")
    for n in ("name", "dollars"):
        if n not in tracks():
            continue
        r = occlusion_report(n)
        o = r["occl"]
        fr = {1: "R", -1: "L", 0: "?"}
        print(f"  {n.upper()}: {r['n_occl']} occluded frames; front hand from evidence: "
              f"{''.join(fr[int(v)] for v in r['front'][r['over']])} ({int(r['over'].sum())} image-overlap frames); symmetric handshape tie: {fits()[n]['symmetric']}")
        for S in "LR":
            s = r[S]
            print(f"    {S}: low-confidence (src<2) occl frames {s['low_src']}, all finite {s['finite']}, in limits "
                  f"{s['in_limits']}, max angle step {s['step_occl']:.1f} deg/frame (rest of clip "
                  f"{s['step_rest']:.1f}), quality {s['quality_occl']:.2f} vs {s['quality_rest']:.2f}")
        mm = 1000 / r["upm"]
        both = np.isfinite(r["pen_before"]) & r["over"]
        cg = np.r_[r["res"]["gap"][r["contact"] & r["res"]["overlap"]], 0.0]
        print(f"    interpenetration in overlap frames: before max {r['pen_before'][both].max() * mm:.1f} mm "
              f"({(r['pen_before'][both] > 0).sum()} frames) -> after max {max(r['pen_after'][both].max(), 0) * mm:.2f} mm; "
              f"depth gap left in contact frames median {np.median(cg) * mm:.1f} / max {cg.max() * mm:.1f} mm "
              f"({int((r['contact'] & r['res']['overlap']).sum())} contact frames); depth shift max {np.abs(r['res']['dz_L']).max() * mm:.0f} mm per hand")
    p = K.load_prior()
    if p is not None:
        d = np.load(K.PRIOR_PATH, allow_pickle=True)
        if "report_heldout" in d.files:
            tr_, te_ = d["report_train"], d["report_heldout"]
            print(f"\nangle prior ({p.k} PCA comps, ASL Citizen): train NLL {tr_[0]:.2f} nats/frame (n={int(tr_[2])}), "
                  f"held-out NLL {te_[0]:.2f} (n={int(te_[2])}, {int(d['report_heldout_files'])} clips), isotropic baseline "
                  f"{float(d['report_heldout_isotropic_nll']):.2f}; held-out PCA reconstruction {te_[1]:.1f} deg RMS")


if __name__ == "__main__":
    import time
    import traceback
    t0 = time.time()
    fits()
    print(f"fitted {len(clips())} clips in {time.time() - t0:.1f}s")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    print_table()
    print(f"\n{failed} failed, total {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)
