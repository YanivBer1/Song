#!/usr/bin/env bash
# Optional stage: 3D hand reconstruction with WiLoR (research licence: CC-BY-NC-ND weights,
# MANO hand model licence). Own environment so the main pipeline stays light.
# Only PyPI wheels and weight files are fetched; the model head and the MANO layer are
# re-implemented in asl_v2/wilor_full.py and asl_v2/mano_np.py.
#
#   scripts/setup_wilor.sh
#   .venv-wilor/bin/python asl_v2/track_wilor.py <word> [<word> ...]   # -> cache/wilor/<word>.npz
#
# About 2.2 s per hand per frame on 2 CPU threads (a 75-frame clip: 3-6 minutes).
# Signs that have a cache/wilor (or asl_v2/data/wilor) file use it automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv -q -p python3.11 .venv-wilor
uv pip install -p .venv-wilor/bin/python numpy==1.26.4 onnxruntime onnx opencv-python-headless scipy rtmlib
uv pip install -p .venv-wilor/bin/python torch --index-url https://download.pytorch.org/whl/cpu

W=asl_v2/models/wilor
HF=https://huggingface.co/warmshao/WiLoR-mini/resolve/main/pretrained_models
mkdir -p "$W"
if [ ! -f "$W/wilor_vit_fp32.onnx" ] || [ ! -f "$W/refine_net.pt" ]; then
  for f in wilor_vit.onnx wilor_final.ckpt MANO_RIGHT.pkl; do
    [ -f "$W/$f" ] || curl -sSfL -o "$W/$f" "$HF/$f"
  done
  .venv-wilor/bin/python -I scripts/wilor_fp16_to_fp32.py "$W/wilor_vit.onnx" "$W/wilor_vit_fp32.onnx"   # fp16 graph -> fp32 (CPU)
  .venv-wilor/bin/python asl_v2/wilor_full.py extract                                                      # refinement head only (33 MB)
  rm -f "$W/wilor_vit.onnx" "$W/wilor_final.ckpt"                                                          # frees 3.8 GB
fi
echo "wilor ready"
