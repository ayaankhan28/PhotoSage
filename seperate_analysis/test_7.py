import os
import imghdr
import time

# Define the path to start the search from (e.g., home directory)
start_path = os.path.expanduser("~")
output_file = "found_images.txt"

def find_images(directory, output_file):
    image_extensions = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}
    image_paths = []
    total_images = 0

    with open(output_file, "w") as file:
        for root, _, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)

                try:
                    # Check if the file is an image using imghdr
                    if imghdr.what(file_path) in image_extensions:
                        image_paths.append(file_path)
                        file.write(file_path + "\n")
                        total_images += 1
                except (OSError, PermissionError):
                    # Skip files that cannot be accessed
                    continue

    return total_images

def main():
    print("Searching for images. This might take a while...")
    a = time.time()
    total_images = find_images(start_path, output_file)
    print(f"Search complete. Total images found: {total_images} in {time.time()-a} secs")
    print(f"Paths of images have been saved to '{output_file}'.")

if __name__ == "__main__":
    main()