import torch
import clip
from torchvision import transforms

def initialize_model(device='cpu'):
    """
    Load the CLIP model and preprocessing pipeline.
    """
    print("Initializing CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return model, transform

device = 'cpu'
model, transform = initialize_model(device)
assert model is not None, "Model initialization failed"
assert transform is not None, "Transform initialization failed"
print("Model and transform initialized successfully.")
