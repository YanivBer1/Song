# AI-SL (text → ASL video), run locally

A self-hosted copy of the Hugging Face Space
[`deenasun/ai-sl-api`](https://huggingface.co/spaces/deenasun/ai-sl-api)
(Berkeley AI Hackathon 2025, Apache-2.0), changed so it runs with **no external
secrets**. The Space's code is vendored unmodified in `upstream/`.

## How the pipeline works

There is no generative video model. The pipeline is retrieval plus pose re-rendering:

1. The sentence is lower-cased, stripped of punctuation and split into words.
   On the Space's text path the words are not converted to ASL gloss (only
   uploaded documents go through the Claude glosser).
2. Each word is embedded with **word2vec-google-news-300** and matched to the
   nearest of ~11.4k sign words from a dictionary (`upstream/videos_rows.csv`).
3. The matching sign clips are downloaded from the dataset's public bucket.
4. **MediaPipe** pulls body and hand keypoints out of each clip. The keypoints
   are drawn as a cartoon avatar, with interpolated transitions between signs,
   and saved as one MP4.

## What is different from the Space

| Space | Here (`local_backend.py`) |
|---|---|
| Supabase pgvector `match_vector` RPC | In-memory cosine search over the same CSV, embedded with the same word2vec model (8,273 of 11,420 words are in word2vec's vocabulary) |
| Upload result to Cloudflare R2 | Skipped; the file stays local |
| OpenCV `avc1` writer (falls back to `mp4v` with pip OpenCV) | Output re-encoded to H.264 with ffmpeg so browsers can play it |

## v2: better motion + realistic avatar (`asl_v2/`)

v2 fixes hands popping in and out, and adds a realistic human renderer.
See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the root-cause analysis, the
survey of open-source alternatives and the design.

- **Both hands are present in every frame.** A hand that isn't signing rests
  relaxed at the side from the start. (In the source clips it hangs below the
  frame, so the Space never drew it.)
- **Consistent body.** MediaPipe Holistic tracking and per-clip normalization,
  plus gap filling and eased, closest-frame transitions.
- **Realistic renderer.** A CC0 MakeHuman character with a full finger rig,
  driven by the sign keypoints (hand twist, IK arms with body contact) and
  rendered in Blender Cycles on CPU.

```bash
scripts/setup_v2.sh                       # venvs, word2vec, MakeHuman CC0 assets
.venv/bin/python -m asl_v2.generate "Hi, my name is John" -o outputs/v2/john_2d.mp4               # seconds
.venv/bin/python -m asl_v2.generate "Hi, my name is John" --renderer human -o outputs/v2/john.mp4  # ~7 s/frame
```

The v2 outputs are in `outputs/v2/`: `*_2d.mp4` (illustrated) and
`*_realistic.mp4` (MakeHuman avatar).

## Current pipeline (v6)

```bash
# text -> realistic avatar video (digits and $/% are spelled out: "$7 billion" -> seven billion dollars)
.venv/bin/python -m asl_v2.generate "Hi, my name is John" --renderer human -o outputs/v6/john.mp4
# optional 3D hand stage (WiLoR) for chosen signs; used automatically once cached
scripts/setup_wilor.sh && .venv-wilor/bin/python asl_v2/track_wilor.py help small
# 24-phrase test on the posed avatar          # visual QA sheets (source next to avatar) for every sign
.venv/bin/python -m asl_v2.eval_suite --tag run   ;   .venv/bin/python -m asl_v2.qa_sheets prepare
```

No generative video model is involved: recorded sign clips are tracked (MediaPipe + RTMPose,
fused per frame; WiLoR 3D hand mesh where available), a MakeHuman character's hand skeleton
is fitted to the tracked landmarks under finger/wrist/forearm joint limits
(`asl_v2/handfit.py`), signs are joined with minimum-jerk hand paths, and Blender Cycles
renders the frames (`asl_v2/render_fast.py`, about 3 s per 1280x720 frame on 4 CPU cores).
Method, measured results and known weak cases: [`docs/QUALITY.md`](docs/QUALITY.md).
Videos: `outputs/v6/`.

## v3: frame-by-frame tracking, anatomical solver, seamless phrases

See [`docs/QUALITY.md`](docs/QUALITY.md) for the method, the audits and the measured
results. In short:

- **Word orders:** 99.93% of all 9,801 ordered sign pairs join without a freeze,
  velocity jump or loop.
- **Joins and anatomy:** 100% velocity-continuous, freeze-free joins; 100% of finger
  joints within human range.
- **Contact and coverage:** 99.6% of frames have no hand inside the body, and 100% of
  hands are on target.
- **Handshape:** 83.6% on signing frames. This is the remaining gap.

The per-frame analysis of every source frame is in [`docs/analysis/`](docs/analysis/).

## Setup (v1, the original Space)

```bash
uv venv -p python3.11 .venv
uv pip install -p .venv/bin/python -r upstream/requirements.txt gradio_client
mkdir -p gensim-data
curl -L -o gensim-data/GoogleNews-vectors-negative300.bin.gz \
  https://github.com/piskvorky/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz
```

The model download is 1.7 GB. Loading it needs about 4 GB of RAM. No GPU is needed.

## Usage

```bash
# Text → MP4 (~50 s on 4 CPUs, most of it spent loading word2vec)
.venv/bin/python run_local.py generate "Hi, my name is John" -o outputs/john.mp4

# The Space's Gradio UI and /predict API on :7860
.venv/bin/python run_local.py serve
```

## Sample outputs (`outputs/`)

| Input | Matched signs |
|---|---|
| Hi, my name is John | hi · my · name · is · john |
| He won $7 billion | he · win · **pm** · billion |
| He won seven billion dollars | he · win · seven · billion · dollars |

`$7` comes out wrong because the Space drops `$` and then embeds `7` with
word2vec. The nearest sign to that vector is "pm". The Space would do the same.
Spelling out numbers avoids this.
