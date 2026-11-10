"""Minimal numpy MANO (right hand) forward pass + safe loader for MANO_RIGHT.pkl.

The pickle holds chumpy objects; we do not install/execute chumpy. A restricted
unpickler only allows numpy / scipy.sparse reconstruction and replaces every
chumpy class by a stub that keeps the raw array.
"""
import pickle
import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("scipy.sparse.csc", "csc_matrix"), ("scipy.sparse._csc", "csc_matrix"),
    ("copy_reg", "_reconstructor"), ("copyreg", "_reconstructor"), ("__builtin__", "object"),
    ("builtins", "object"),
}


class _Stub:
    """Stands in for chumpy.Ch & co; keeps whatever state the pickle carries."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"state": st})

    def arr(self):
        d = self.__dict__
        for k in ("x", "a"):
            if k in d:
                v = d[k]
                return v.arr() if isinstance(v, _Stub) else np.asarray(v)
        raise KeyError(list(d))


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        if module in ("__builtin__", "builtins") and name in ("set", "object"):
            return {"set": set, "object": object}[name]
        if module in ("scipy.sparse.csc", "scipy.sparse._csc") and name == "csc_matrix":
            import scipy.sparse
            return scipy.sparse.csc_matrix
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def _arr(v):
    if isinstance(v, _Stub):
        return np.asarray(v.arr(), np.float64)
    if hasattr(v, "toarray"):
        return np.asarray(v.toarray(), np.float64)
    return np.asarray(v)


TIPS = [744, 320, 443, 554, 671]  # smplx vertex_ids["mano"] thumb..pinky tips (as WiLoR-mini)
# 16 MANO joints + 5 tips -> OpenPose/MediaPipe 21 order
TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class Mano:
    def __init__(self, path):
        with open(path, "rb") as f:
            d = _Safe(f, encoding="latin1").load()
        self.v_template = _arr(d["v_template"])            # (778,3)
        self.shapedirs = _arr(d["shapedirs"])[:, :, :10]   # (778,3,10)
        self.posedirs = _arr(d["posedirs"])                # (778,3,135)
        self.J_regressor = _arr(d["J_regressor"])          # (16,778)
        self.weights = _arr(d["weights"])                  # (778,16)
        self.parents = np.asarray(d["kintree_table"])[0].astype(int)
        self.parents[0] = -1
        self.faces = np.asarray(d["f"]).astype(int)

    def __call__(self, rotmats, betas):
        """rotmats (16,3,3) [global + 15 fingers], betas (10,) -> verts (778,3), joints21 (21,3)
        in OpenPose/MediaPipe order, metres, MANO frame (root-relative NOT applied)."""
        v = self.v_template + self.shapedirs @ betas
        J = self.J_regressor @ v
        pose_feat = (rotmats[1:] - np.eye(3)).reshape(-1)
        v = v + self.posedirs @ pose_feat
        G = np.zeros((16, 4, 4))
        for i in range(16):
            T = np.eye(4)
            T[:3, :3] = rotmats[i]
            p = self.parents[i]
            T[:3, 3] = J[i] - (J[p] if p >= 0 else 0)
            G[i] = T if p < 0 else G[p] @ T
        Jt = G[:, :3, 3].copy()
        A = G.copy()
        A[:, :3, 3] -= np.einsum("nij,nj->ni", G[:, :3, :3], J)
        Tv = np.einsum("vn,nij->vij", self.weights, A)
        verts = np.einsum("vij,vj->vi", Tv[:, :3, :3], v) + Tv[:, :3, 3]
        j21 = np.concatenate([Jt, verts[TIPS]])[TO_OPENPOSE]
        return verts, j21


def rot6d_to_rotmat(x):
    """HaMeR/WiLoR convention: x (...,6) -> reshape(2,3).T, Gram-Schmidt on columns."""
    x = np.asarray(x, np.float64).reshape(-1, 2, 3).transpose(0, 2, 1)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], -1)


if __name__ == "__main__":
    import sys
    m = Mano(sys.argv[1])
    v, j = m(np.tile(np.eye(3), (16, 1, 1)), np.zeros(10))
    print(v.shape, j.shape, m.parents)
    print("hand length wrist->middle tip (m):", np.linalg.norm(j[12] - j[0]))
    print(np.round(j, 3))
