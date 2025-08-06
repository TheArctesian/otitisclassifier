import os
from PIL import Image
import numpy as np
from skimage import exposure

src_dir = "data/raw"
out_dir = "data/processed"
target_size = (500, 500)  # Change this as needed

def convert_and_standardize(image_path, save_path):
    with Image.open(image_path) as im:
        im = im.convert('RGB')
        im = im.resize(target_size, Image.LANCZOS)
        img_array = np.array(im).astype(np.uint8)
    
    # Apply CLAHE to each color channel independently
    clahe_img = np.zeros_like(img_array)
    for c in range(3):
        clahe_channel = exposure.equalize_adapthist(img_array[..., c], clip_limit=0.03)
        clahe_img[..., c] = np.clip(clahe_channel * 255, 0, 255)
    clahe_img = clahe_img.astype(np.uint8)

    # Standardize to zero mean and unit variance
    img_float = clahe_img.astype(np.float32) / 255.0
    mean = img_float.mean(axis=(0, 1), keepdims=True)
    std = img_float.std(axis=(0, 1), keepdims=True) + 1e-7
    img_standardized = (img_float - mean) / std

    # Rescale for JPEG saving (so the output is visible as a standard image)
    img_rescaled = np.clip((img_standardized * 64) + 128, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(img_rescaled)
    img_out.save(save_path, "JPEG", quality=95)

# Recursively process and preserve folder structure
if __name__ == "__main__":
    valid_exts = (".tif", ".tiff", ".png", ".jpg")
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(out_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in files:
            if file_name.lower().endswith(valid_exts):
                src_path = os.path.join(root, file_name)
                base_name = os.path.splitext(file_name)[0]
                out_path = os.path.join(target_dir, base_name + ".jpg")
                # Process ONLY if the output file does NOT already exist
                if not os.path.exists(out_path):
                    convert_and_standardize(src_path, out_path)