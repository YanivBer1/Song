"""Full WiLoR on CPU: ONNX ViT-H backbone (onnxruntime) + RefineNet head (torch, own re-implementation
of the published architecture) + numpy MANO.  No GitHub code is imported.

Weights (Hugging Face warmshao/WiLoR-mini, pretrained_models/):
  wilor_vit.onnx   -> converted to fp32 by fp16_to_fp32.py
  wilor_final.ckpt -> only the refine_net.* tensors are used (extracted once to refine_net.pt)
  MANO_RIGHT.pkl

    w = WiLoR()
    out = w(frame_bgr, [cx, cy, size], is_right=True)
    out["kp2d"]  (21,2) px in the frame          out["kp3d"] (21,3) m, wrist-relative, camera axes (x right, y down, z away)
    out["img"]   (21,3) MediaPipe-style landmarks: x,y px in the frame, z in the same px scale relative to the wrist
                 (negative = towards the camera) -> drop-in for the cached img_L / img_R rows
    out["R"]     (3,3) wrist/global orientation  out["rot"]  (16,3,3) MANO joint rotations   out["betas"] (10,)
    out["coarse"] same fields from the backbone-only (un-refined) estimate
"""
import math
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from mano_np import Mano, rot6d_to_rotmat

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "wilor")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FOCAL = 5000.0


def _deconv(dims):
    L = []
    for a, b in zip(dims[:-1], dims[1:]):
        L += [nn.ConvTranspose2d(a, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*L)


class DeConvNet(nn.Module):
    def __init__(self, d=1280):
        super().__init__()
        self.first_conv = nn.Sequential(nn.Conv2d(d, d // 2, 1))
        self.deconv = nn.ModuleList([_deconv([d // 2, d // 4]), _deconv([d // 2, d // 4, d // 8])])

    def forward(self, x):
        x = self.first_conv(x)
        return [self.deconv[1](x), self.deconv[0](x), x]  # high -> low resolution


class RefineNet(nn.Module):
    def __init__(self, d=1280):
        super().__init__()
        self.deconv = DeConvNet(d)
        n = d // 8 + d // 4 + d // 2
        self.dec_pose, self.dec_cam, self.dec_shape = nn.Linear(n, 96), nn.Linear(n, 3), nn.Linear(n, 10)

    def forward(self, img_feat, verts, cam, pose6d, betas):
        feats = []
        for f in self.deconv(img_feat):
            size = f.shape[2]
            t = torch.stack([cam[:, 1], cam[:, 2], 2 * FOCAL / (size * cam[:, 0] + 1e-9)], -1)
            p = verts + t[:, None]
            xy = (FOCAL / size) * p[..., :2] / p[..., 2:3]      # as published: no principal point added
            h, w = f.shape[2:]
            grid = torch.stack([xy[..., 0] / (w - 1) * 2 - 1, xy[..., 1] / (h - 1) * 2 - 1], 2)[:, :, None]
            s = F.grid_sample(f, grid, align_corners=True)[..., 0]  # B,C,V
            feats.append(s.max(2).values)
        v = torch.cat(feats, -1)
        return pose6d + self.dec_pose(v), betas + self.dec_shape(v), cam + self.dec_cam(v)


def extract_refine(ckpt=os.path.join(W, "wilor_final.ckpt"), dst=os.path.join(W, "refine_net.pt")):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    out = {k[len("refine_net."):]: v.float() for k, v in sd.items() if k.startswith("refine_net.")}
    torch.save(out, dst)
    return sorted(out)


class WiLoR:
    def __init__(self, threads=2, rescale=2.0, refine=True):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        torch.set_num_threads(threads)
        self.sess = ort.InferenceSession(os.path.join(W, "wilor_vit_fp32.onnx"), so, providers=["CPUExecutionProvider"])
        self.mano = Mano(os.path.join(W, "MANO_RIGHT.pkl"))
        self.rescale = rescale
        self.refine = None
        if refine:
            self.refine = RefineNet().eval()
            self.refine.load_state_dict(torch.load(os.path.join(W, "refine_net.pt"), weights_only=True), strict=True)
        self.last_ms = 0.0

    def _decode(self, pose6d, betas, cam, M, is_right):
        rot = rot6d_to_rotmat(pose6d)
        verts, j = self.mano(rot, np.asarray(betas, np.float64))
        sc, tx, ty = [float(v) for v in cam]
        p = j + np.array([tx, ty, 2 * FOCAL / (256.0 * sc + 1e-9)])
        uv = FOCAL * p[:, :2] / p[:, 2:3] + 128.0
        Mi = cv2.invertAffineTransform(M)
        kp2d = uv @ Mi[:, :2].T + Mi[:, 2]
        kp3d = j - j[0]
        R = rot[0]
        if not is_right:
            S = np.diag([-1.0, 1, 1])
            kp3d, R = kp3d @ S, S @ R @ S
        return dict(kp2d=kp2d.astype(np.float32), kp3d=kp3d.astype(np.float32), R=R, rot=rot,
                    betas=np.asarray(betas), cam=np.asarray(cam)), verts

    def __call__(self, frame, box, is_right=True):
        cx, cy, s = [float(v) for v in box]
        size = s * self.rescale
        k = 256.0 / size
        if is_right:
            M = np.array([[k, 0, 128 - k * cx], [0, k, 128 - k * cy]], np.float32)
        else:  # mirror: the network only knows right hands
            M = np.array([[-k, 0, 127 + k * cx], [0, k, 128 - k * cy]], np.float32)
        src = frame
        if size / 256 / 2 > 1.1:
            src = cv2.GaussianBlur(frame, (0, 0), (size / 256 / 2 - 1) / 2)
        patch = cv2.warpAffine(src, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        x = (patch[:, 32:-32, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        t0 = time.perf_counter()
        pose6d, betas, cam, feat = self.sess.run(None, {"imgs": x})
        coarse, verts = self._decode(pose6d[0], betas[0], cam[0], M, is_right)
        out = coarse
        if self.refine is not None:
            with torch.no_grad():
                p, b, c = self.refine(torch.from_numpy(feat.astype(np.float32)),
                                      torch.from_numpy(verts[None].astype(np.float32)),
                                      torch.from_numpy(cam), torch.from_numpy(pose6d), torch.from_numpy(betas))
            out, _ = self._decode(p[0].numpy(), b[0].numpy(), c[0].numpy(), M, is_right)
            out["coarse"] = coarse
        self.last_ms = (time.perf_counter() - t0) * 1000
        ppm = float(out["cam"][0]) * size / 2.0          # frame pixels per metre at the hand
        out["img"] = np.concatenate([out["kp2d"], out["kp3d"][:, 2:3] * ppm], 1).astype(np.float32)
        out["px_per_m"] = ppm
        out["patch"] = patch
        return out


def palm_frame(kp3d, is_right=True):
    """Hand orientation from 21 joints (camera axes). Returns (3,3) with columns
    [across: index-MCP -> pinky-MCP, along: wrist -> middle-MCP, out: normal pointing OUT OF THE PALM]."""
    y = kp3d[9] - kp3d[0]
    y = y / np.linalg.norm(y)
    x = kp3d[17] - kp3d[5]
    x = x - y * (x @ y)
    x = x / np.linalg.norm(x)
    n = np.cross(x, y) * (-1.0 if is_right else 1.0)
    return np.stack([x, y, n], 1)


def hand21(frame_bgr, bbox_xyxy, is_right=True, _m=[]):
    """Minimal entry point: BGR frame + tight hand bbox (x1,y1,x2,y2) around the visible hand ->
    (kp2d (21,2) px, kp3d (21,3) m wrist-relative camera axes, R (3,3) wrist orientation). Joint order = MediaPipe."""
    if not _m:
        _m.append(WiLoR())
    x1, y1, x2, y2 = bbox_xyxy
    o = _m[0](frame_bgr, [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.25], is_right)
    return o["kp2d"], o["kp3d"], o["R"]


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["extract"]:
        for k in extract_refine():
            print(k)
