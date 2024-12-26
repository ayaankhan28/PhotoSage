import numpy as np

def save_embedding_cache(embeddings, image_paths, embedding_cache_file='embedding_cache1.npy'):
    try:
        np.save(embedding_cache_file, {"embeddings": embeddings, "image_paths": image_paths})
        print("Saved embeddings to cache.")
    except Exception as e:
        print(f"Error saving cache: {e}")


embeddings = {'/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/1.jpg': [0.1, 0.2, 0.3]}  # Dummy data
image_paths = ['/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/2.jpg']
save_embedding_cache(embeddings, image_paths, 'embedding_cache1.npy')
print("Embeddings saved successfully.")
