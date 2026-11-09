#!/usr/bin/env bash
# Set up the v2 pipeline: Python envs, word2vec model, MakeHuman CC0 data/assets.
# Only MakeHuman *data files* are used; no MakeHuman/MPFB code is executed.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv -q -p python3.11 .venv
uv pip install -p .venv/bin/python -r upstream/requirements.txt gradio_client scipy
uv venv -q -p python3.11 .venv-blender
uv pip install -p .venv-blender/bin/python "bpy==5.0.1" numpy

mkdir -p gensim-data
[ -f gensim-data/GoogleNews-vectors-negative300.bin.gz ] || curl -L -o gensim-data/GoogleNews-vectors-negative300.bin.gz \
  https://github.com/piskvorky/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz

mkdir -p third_party
if [ ! -d third_party/mpfb2 ]; then
  git clone --filter=blob:none https://github.com/makehumancommunity/mpfb2 third_party/mpfb2
  git -C third_party/mpfb2 checkout 6ff1eb3
fi
if [ ! -d third_party/mh_assets/skins ]; then
  mkdir -p third_party/mh_assets
  curl -L -o /tmp/mh_assets.zip https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
  unzip -q /tmp/mh_assets.zip -d third_party/mh_assets && rm /tmp/mh_assets.zip
fi
echo "ready"
