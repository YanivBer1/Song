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

## Setup

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
