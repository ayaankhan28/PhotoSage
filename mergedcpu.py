import os
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Dict
import faiss
import clip


class MobileImageSearch:
    def __init__(self,
                 images_folder='images',
                 embedding_cache_file='embedding_cache.npy',
                 similarity_threshold=0.7,
                 batch_size=16,
                 max_index_size=10000):
        """
        Mobile Optimized Image Search Engine with improved macOS compatibility
        """
        self.device = "cpu"  # Force CPU usage for stability
        print(f"Using device: {self.device}")

        self.images_folder = images_folder
        self.embedding_cache_file = embedding_cache_file
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_index_size = max_index_size

        self.model, self.transform = self._initialize_model()
        self.embeddings = {}
        self.image_paths = []
        self.index = None

        # Load or initialize embeddings
        self._load_or_create_embedding_cache()

        # Initialize FAISS index after loading cache
        self._initialize_faiss_index()

    def _initialize_model(self):
        """
        Load the CLIP model and preprocessing pipeline
        """
        print("Initializing CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        return model, transform

    def _initialize_faiss_index(self):
        """
        Initialize FAISS index with improved error handling
        """
        print("Initializing FAISS index...")
        try:
            dim = 512  # CLIP embedding dimension
            self.index = faiss.IndexFlatIP(dim)

            if len(self.embeddings) > 0:
                # Convert embeddings dictionary to a list while maintaining order
                embeddings_list = [self.embeddings[path] for path in self.image_paths]
                embeddings_array = np.array(embeddings_list).astype(np.float32)
                self.index.add(embeddings_array)
                print(f"Added {len(self.embeddings)} embeddings to FAISS index")
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            raise

    def _load_or_create_embedding_cache(self):
        """
        Load cached embeddings with error handling
        """
        try:
            if os.path.exists(self.embedding_cache_file):
                data = np.load(self.embedding_cache_file, allow_pickle=True).item()
                self.embeddings = data.get("embeddings", {})
                self.image_paths = data.get("image_paths", [])
                print(f"Loaded {len(self.embeddings)} cached embeddings")
            else:
                print("No cache found. Starting fresh.")
                self.embeddings = {}
                self.image_paths = []
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.embeddings = {}
            self.image_paths = []

    def _save_embedding_cache(self):
        """
        Save embeddings cache with error handling
        """
        try:
            np.save(self.embedding_cache_file, {
                "embeddings": self.embeddings,
                "image_paths": self.image_paths
            })
            print("Saved embeddings to cache.")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _process_single_image(self, image_path):
        """
        Process a single image with error handling
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor).cpu().numpy().flatten()

            embedding /= np.linalg.norm(embedding)
            return image_path, embedding
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def generate_embeddings(self, new_image_paths: List[str]):
        """
        Generate embeddings with improved batch processing
        """
        print(f"Generating embeddings for {len(new_image_paths)} new images...")

        processed = 0
        new_embeddings = {}

        for i in range(0, len(new_image_paths), self.batch_size):
            batch_paths = new_image_paths[i:i + self.batch_size]

            for path in batch_paths:
                result = self._process_single_image(path)
                if result:
                    path, embedding = result
                    new_embeddings[path] = embedding
                    processed += 1

            if processed % 400 == 0 and processed > 0:
                print(f"Processed {processed} images...")

        # Update embeddings and paths
        self.embeddings.update(new_embeddings)
        self.image_paths.extend(list(new_embeddings.keys()))

        # Rebuild the FAISS index from scratch
        self._initialize_faiss_index()

        self._save_embedding_cache()
        print(f"Successfully processed {len(new_embeddings)} images")

    def index_images(self):
        """
        Index all images in the folder
        """
        print(f"Indexing images from folder: {self.images_folder}")

        try:
            all_image_paths = []
            valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

            for root, _, files in os.walk(self.images_folder):
                for file in files:
                    if os.path.splitext(file.lower())[1] in valid_extensions:
                        all_image_paths.append(os.path.join(root, file))

            new_image_paths = [path for path in all_image_paths if path not in self.embeddings]
            print(f"Found {len(new_image_paths)} new images to index.")

            if new_image_paths:
                self.generate_embeddings(new_image_paths)

        except Exception as e:
            print(f"Error during indexing: {e}")

    def search_images(self, query: str) -> List[str]:
        """
        Search for images with improved error handling
        """
        try:
            print(f"Searching for images matching query: '{query}'...")
            text_tokens = clip.tokenize([query]).to(self.device)

            with torch.no_grad():
                text_embedding = self.model.encode_text(text_tokens).cpu().numpy().flatten()

            text_embedding /= np.linalg.norm(text_embedding)
            text_embedding = text_embedding.astype(np.float32)

            # Get number of images to search through
            k = min(15, self.index.ntotal)
            if k == 0:
                print("No images in the index to search.")
                return []

            distances, indices = self.index.search(text_embedding.reshape(1, -1), k)

            # Filter results based on similarity threshold
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if dist >= self.similarity_threshold:
                    results.append(self.image_paths[idx])

            print(f"Found {len(results)} matching results.")
            return results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def get_stats(self):
        """
        Return stats about current embeddings and index
        """
        return {
            "total_images": len(self.embeddings),
            "indexed_images": self.index.ntotal if self.index else 0
        }


def main():
    """
    Example usage of the MobileImageSearch class
    """
    search_engine = MobileImageSearch(
        images_folder="/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage",
        embedding_cache_file="embedding_cache1.npy",
        similarity_threshold=0.25,  # Lowered threshold for more results
        batch_size=16
    )

    # Index all images
    search_engine.index_images()

    # Display statistics
    stats = search_engine.get_stats()
    print("\nCurrent Statistics:")
    print(f"Total images in database: {stats['total_images']}")
    print(f"Total indexed images: {stats['indexed_images']}")

    # Example searches
    example_queries = [
        "A girl with red light effect",
    ]

    for query in example_queries:
        print(f"\nSearching for: {query}")
        results = search_engine.search_images(query)

        if results:
            print("Matching images:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result}")
        else:
            print("No matching images found.")


if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        execution_time = time.time() - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")