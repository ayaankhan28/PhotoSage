from PIL import Image
import torch
import numpy as np

def process_single_image(image_path, model, transform, device='cpu'):
    """
    Process a single image and return its embedding.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor).cpu().numpy().flatten()

        embedding /= np.linalg.norm(embedding)
        return image_path, embedding
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

from test_1 import initialize_model
model, transform = initialize_model('cpu')
image_path = "/hackathon/results/Storage/1.jpg"  # Ensure you have this image available
result = process_single_image(image_path, model, transform)
assert result is not None, "Processing image failed"
print(f"Processed image: {result[0]} {result[1]}")