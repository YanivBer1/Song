import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import os
from supabase import acreate_client, AsyncClient
from dotenv import load_dotenv

class Vectorizer:
    """
    A class to:
        - Generate embeddings of words
        - Query for words from Supabase database based on vector similarity
        - Return matching ASL videos for words
    """

    def load_kv(self, model_name='word2vec-google-news-300'):
        """
        Returns a KeyedVector object loaded from gensim
        """
        model_path = os.path.join(os.getcwd(), 'gensim-data', 'GoogleNews-vectors-negative300.bin.gz')
        try:
            print(f"Loading model from {model_path}")
            kv = KeyedVectors.load_word2vec_format(model_path, binary=True)
            print("Word2Vec model loaded successfully as KeyedVectors object.")
            return kv
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Trying to download...")
            kv = gensim.downloader.load(model_name)  # returns a keyedvector
            print("Word2Vec model loaded successfully as KeyedVectors object.")
            return kv
        except Exception as e:
            print(f"Unable to load embedding model from gensim: {e}")
            return None

    async def initialize_supabase(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        supabase: AsyncClient = await acreate_client(url, key)
        return supabase

    def __init__(self):
        load_dotenv()
        self.kv = self.load_kv()
        self.supabase = None  # Will be initialized when needed

    async def ensure_supabase_initialized(self):
        """Ensure Supabase client is initialized"""
        if self.supabase is None:
            self.supabase = await self.initialize_supabase()

    def encode(self, word):
        print(f"encoding {word}")
        if self.kv is None:
            print("KeyedVectors not loaded")
            return None
        if word in self.kv.key_to_index:
            return self.kv[word]
        else:
            print(f"Error: {word} is not in the KeyedVector's vocabulary")
            # Try to find closest match
            try:
                closest_matches = self.kv.most_similar(word, topn=3)
                if closest_matches:
                    closest_word = closest_matches[0][0]
                    print(f"Using closest match '{closest_word}' for '{word}'")
                    return self.kv[closest_word]
                else:
                    print(f"No similar words found for '{word}'")
            except Exception as e:
                print(f"Error finding similar words: {e}")
            return None
    
    def encode_and_format(self, word):
        """
        Apply encoding function to each word.
        Prettify the encoding to match expected format for Supabase vectors
        """
        enc = self.encode(word)
        return "[" + ",".join(map(str, enc.tolist())) + "]" if enc is not None else None

    async def vector_query_from_supabase(self, query):
        try:
            await self.ensure_supabase_initialized()
            query_embedding = self.encode(query)
            
            if query_embedding is None:
                return {
                    "match": False,
                    "error": f"'{query}' not in vocabulary and no similar words found"
                }

            query_embedding = query_embedding.tolist()

            if self.supabase is not None:
                result = await self.supabase.rpc(
                    "match_vector",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.0,
                        "match_count": 1
                    }
                ).execute()

                data = result.data
                if data:
                    match = data[0]
                    return {
                        "match": True,
                        "query": query,
                        "matching_word": match["word"],
                        "video_url": match["video_url"],
                        "similarity": match["similarity"]
                    }
                else:
                    return {"match": False}
            else:
                return {"match": False, "error": "Supabase not initialized"}

        except Exception as e:
            print(f"RPC call failed: {e}")
            return {"match": False, "error": str(e)}


def load_filtered_kv(model_name='word2vec-google-news-300', vocab=None):
    """
    Returns a KeyedVector object whose vocabulary
    consists of the words in vocab
    """
    if vocab is None:
        vocab = []
    try:
        # gensim.downloader.load returns a KeyedVector
        original_kv = gensim.downloader.load(model_name)
        if vocab:
            filtered_key2vec_map = {}
            for key in vocab:
                if key in original_kv.key_to_index:
                    filtered_key2vec_map[key] = original_kv[key]

            new_kv = gensim.models.KeyedVectors(
                vector_size=original_kv.vector_size)
            new_kv.add_vectors(list(filtered_key2vec_map.keys()),
                               np.array(list(filtered_key2vec_map.values())))
            return original_kv
        else:
            return original_kv
    except Exception as e:
        print(f"Unable to load embedding model from gensim: {e}")
        return None


async def main():
    vectorizer = Vectorizer()

    # Test exact word match
    vector = vectorizer.encode("test")
    print(vector)
    
    # Test words not in vocabulary with closest match fallback
    result = await vectorizer.vector_query_from_supabase("dog")
    print(result)
    result = await vectorizer.vector_query_from_supabase("cat")
    print(result)

    # read word list
    # df = pd.read_csv('videos_rows.csv')
    
    # # Add embeddings column - apply encode to each word
    # df['embedding'] = df['word'].apply(vectorizer.encode_and_format)

    # # Drop any rows that don't have an embedding
    # df = df.dropna(subset=['embedding'])
    # print(df.head())

    # df.to_csv("vectors.csv", index=False, columns=["word", "video_url", "embedding"], header=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
