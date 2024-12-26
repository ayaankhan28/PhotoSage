from test_4 import process_single_image
from test_1 import initialize_model
def generate_embeddings(new_image_paths, batch_size, model, transform, device='cpu'):
    """
    Generate embeddings for a list of images.
    """
    print(f"Generating embeddings for {len(new_image_paths)} new images...")

    processed = 0
    new_embeddings = {}

    for i in range(0, len(new_image_paths), batch_size):
        batch_paths = new_image_paths[i:i + batch_size]

        for path in batch_paths:
            result = process_single_image(path, model, transform, device)
            if result:
                path, embedding = result
                new_embeddings[path] = embedding
                processed += 1

        if processed % 400 == 0 and processed > 0:
            print(f"Processed {processed} images...")

    print(f"Successfully processed {len(new_embeddings)} images")
    return new_embeddings


model, transform = initialize_model('cpu')
new_image_paths = ["/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/1.jpg", "/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage/2.jpg"]  # Ensure these images exist
embeddings = generate_embeddings(new_image_paths, batch_size=2, model=model, transform=transform)
assert len(embeddings) == len(new_image_paths), "Generated embeddings mismatch"
print("Embeddings generated successfully.")