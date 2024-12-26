import faiss
import numpy as np

def initialize_faiss_index(embeddings, image_paths):
    """
    Initialize FAISS index and add embeddings to it.
    """
    try:
        dim = 512  # CLIP embedding dimension
        index = faiss.IndexFlatIP(dim)

        if len(embeddings) > 0:
            embeddings_list = [embeddings[path] for path in image_paths]
            embeddings_array = np.array(embeddings_list).astype(np.float32)
            index.add(embeddings_array)
            print(f"Added {len(embeddings)} embeddings to FAISS index")
        return index
    except Exception as e:
        print(f"Error initializing FAISS index: {e}")
        raise

embeddings = {'/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/1.jpg': [0.1, 0.2, 0.3]}  # Dummy data
image_paths = ['/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/1.jpg']
index = initialize_faiss_index(embeddings, image_paths)
assert index.ntotal > 0, "FAISS index initialization failed"
print(f"FAISS index initialized with {index.ntotal} items.")