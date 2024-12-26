import os
import numpy as np


def load_embedding_cache(embedding_cache_file='embedding_cache1.npy'):
    """
    Load cached embeddings from a file.
    """
    embeddings = {}
    image_paths = []

    try:
        if os.path.exists(embedding_cache_file):
            data = np.load(embedding_cache_file, allow_pickle=True).item()
            embeddings = data.get("embeddings", {})
            image_paths = data.get("image_paths", [])
            print(f"Loaded {len(embeddings)} cached embeddings")
        else:
            print("No cache found. Starting fresh.")
    except Exception as e:
        print(f"Error loading cache: {e}")

    return embeddings, image_paths

embeddings, image_paths = load_embedding_cache('/hackathon/results/embedding_cache1.npy')
print(f"Loaded {len(embeddings)} embeddings and {len(image_paths)} image paths.")
