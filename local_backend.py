"""
Local, self-contained replacements for the cloud services the
deenasun/ai-sl-api Hugging Face Space depends on.

The Space looks up each word with a Supabase pgvector RPC (`match_vector`)
and uploads the stitched result to Cloudflare R2. Neither is reachable without
the Space owner's secrets, so here:

  * Supabase -> an in-memory cosine-similarity index built from the Space's
    own `videos_rows.csv` (word -> public sign-video URL), embedded with the
    same word2vec-google-news-300 model the Space uses.
  * R2 upload -> skipped; the stitched video stays on local disk.

Everything else (word2vec encoding, MediaPipe keypoint extraction, skeleton
rendering and stitching) runs unmodified from `upstream/`.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "upstream", "videos_rows.csv")


class LocalVectorIndex:
    """Mimics the Supabase `match_vector` RPC: nearest sign by cosine similarity."""

    def __init__(self, kv, csv_path=CSV_PATH):
        df = pd.read_csv(csv_path)
        words, urls, vecs = [], [], []
        for word, url in zip(df["word"], df["video_url"]):
            word = str(word)
            if word in kv.key_to_index:
                words.append(word)
                urls.append(url)
                vecs.append(kv[word])
        mat = np.asarray(vecs, dtype=np.float32)
        self.matrix = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        self.words = words
        self.urls = urls
        print(f"Local vector index: {len(words)}/{len(df)} sign videos embedded")

    def match(self, query_embedding, match_count=1):
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / np.linalg.norm(q)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:match_count]
        return [
            {"word": self.words[i], "video_url": self.urls[i], "similarity": float(sims[i])}
            for i in top
        ]


def install(app_module):
    """Patch the imported upstream `app` module to use local services."""
    vectorizer = app_module.vectorizer
    index = LocalVectorIndex(vectorizer.kv)

    async def vector_query_local(query):
        embedding = vectorizer.encode(query)
        if embedding is None:
            return {"match": False,
                    "error": f"'{query}' not in vocabulary and no similar words found"}
        best = index.match(embedding, match_count=1)[0]
        return {"match": True, "query": query, "matching_word": best["word"],
                "video_url": best["video_url"], "similarity": best["similarity"]}

    vectorizer.vector_query_from_supabase = vector_query_local
    app_module.upload_video_to_r2 = lambda path, bucket_name=None: None
    return index
