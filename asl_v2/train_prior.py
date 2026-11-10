"""
Train a hand-pose prior in the avatar's own joint-angle space.

  python -m asl_v2.train_prior collect   # fit every tracked sign, save finger angles
  python -m asl_v2.train_prior train     # PCA + Gaussian prior -> models/avatar_hand_prior.npz

Data: the avatar-space fits (asl_v2.handfit) of every frame of every tracked dictionary
sign (both hands; left-hand abduction/thumb signs are comparable because handfit's
angles are defined per hand with flexion positive). The prior is used by handfit as a
soft regulariser, so under-determined frames settle on handshapes signers actually use
instead of wandering between equivalent solutions.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "cache", "prior", "avatar_angles.npz")
OUT = os.path.join(ROOT, "asl_v2", "models", "avatar_hand_prior.npz")


def collect():
    from .generate import tracked_keypoints
    from .handfit import _fit_side, _to_bl
    from .motion import LH, RH, process_clip
    words = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "cache", "tracks")) if f.endswith(".npz"))
    X, meta = [], []
    for i, w in enumerate(words):
        try:
            clip, m = process_clip(tracked_keypoints(w), return_meta=True, trim=False)
        except Exception as e:  # noqa: BLE001
            print("skip", w, e, flush=True)
            continue
        mask = np.ones(len(clip), bool)
        for s, off in (("l", LH), ("r", RH)):
            wgt = m[f"weight_{s.upper()}"]
            if wgt.max() < 0.5:
                continue
            _, x, _, rms = _fit_side((s, _to_bl(clip[:, off:off + 21] - clip[:, off:off + 1]), mask))
            keep = (wgt > 0.8) & (rms < 0.05)
            X.append(x[keep, 3:25])
            meta += [(i, s == "r")] * int(keep.sum())
        if i % 10 == 0:
            print(f"{i + 1}/{len(words)} signs, {sum(len(a) for a in X)} poses", flush=True)
            np.savez_compressed(DATA, X=np.concatenate(X), meta=np.array(meta), words=np.array(words))
    np.savez_compressed(DATA, X=np.concatenate(X), meta=np.array(meta), words=np.array(words))
    print("collected", sum(len(a) for a in X), "poses")


def train(n_comp=12, holdout=0.1):
    d = np.load(DATA)
    X, meta = d["X"], d["meta"]
    signs = np.unique(meta[:, 0])
    rng = np.random.default_rng(0)
    test_signs = rng.choice(signs, max(1, int(len(signs) * holdout)), replace=False)
    te = np.isin(meta[:, 0], test_signs)
    mu = X[~te].mean(0)
    U, S, Vt = np.linalg.svd(X[~te] - mu, full_matrices=False)
    var = S ** 2 / (len(X[~te]) - 1)
    comps, lam = Vt[:n_comp], var[:n_comp]
    resid = float(var[n_comp:].mean())

    def recon_err(A):
        z = (A - mu) @ comps.T
        return np.degrees(np.sqrt((((A - mu) - z @ comps) ** 2).mean()))

    print(f"{len(X)} poses from {len(signs)} signs; {n_comp} components explain "
          f"{100 * lam.sum() / var.sum():.1f}% of variance; reconstruction error: train {recon_err(X[~te]):.1f} deg, "
          f"held-out signs {recon_err(X[te]):.1f} deg")
    np.savez(OUT, mean=mu, comps=comps, lam=lam, resid=resid)


if __name__ == "__main__":
    {"collect": collect, "train": train}[sys.argv[1]]()
