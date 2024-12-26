import os
import numpy as np
import time
import torch
from torchvision import transforms
from PIL import Image
import faiss
import clip
from typing import List

class ImageSearchUpdater:
    def __init__(self,
                 images_folder='images',
                 embedding_cache_file='embedding_cache.npy',
                 batch_size=16):
        """
        Initialize the Image Search Updater.
        """
        self.device = "cpu"  # Force CPU usage for stability
        print(f"Using device: {self.device}")

        self.images_folder = images_folder
        self.embedding_cache_file = embedding_cache_file
        self.batch_size = batch_size

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
        Generate embeddings for new images and update FAISS index
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

    def update_index_with_new_images(self):
        """
        Check if new images have been added, process them and update the FAISS index
        """
        print(f"Checking for new images in folder: {self.images_folder}")

        all_image_paths = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

        for root, _, files in os.walk(self.images_folder):
            for file in files:
                if os.path.splitext(file.lower())[1] in valid_extensions:
                    all_image_paths.append(os.path.join(root, file))

        new_image_paths = [path for path in all_image_paths if path not in self.embeddings]
        print(f"Found {len(new_image_paths)} new images to process.")

        if new_image_paths:
            self.generate_embeddings(new_image_paths)
        else:
            print("No new images to process.")

if __name__ == "__main__":
    try:
        start_time = time.time()

        # Initialize ImageSearchUpdater
        search_updater = ImageSearchUpdater(
            images_folder='/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage',  # Replace with your images folder path
            embedding_cache_file='embedding_cache.npy',
            batch_size=16
        )

        # Check for new images and update the FAISS index if new ones are found
        search_updater.update_index_with_new_images()

        execution_time = time.time() - start_time
        print(f"\nTotal execution time: {execution_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
