import cv2
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
# Where your current high-res photos are
DATASET_DIR = Path("dataset/known_faces") 

def add_noise(image):
    """Adds 'grain' to the image to simulate a cheap sensor."""
    row, col, ch = image.shape
    mean = 0
    sigma = 15 # Intensity of the noise
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    # Clip values to stay between 0-255
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def darken_image(image):
    """Reduces brightness to simulate a dark room."""
    # Convert to HSV, lower the 'Value' (Brightness), convert back
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Reduce brightness by 40%
    v = cv2.multiply(v, 0.6).astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def blur_image(image):
    """Simulates a camera out of focus."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def process_dataset():
    print(f"[INFO] Augmenting data in {DATASET_DIR}...")
    
    # Walk through every person's folder
    for folder in DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        print(f"Processing: {folder.name}")
        
        # Grab all images in that folder
        # We wrap in list() so we don't loop over the new files we create
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        for img_path in images:
            # Skip images that are already augmented
            if "aug_" in img_path.name:
                continue

            # Load the original high-quality image
            original = cv2.imread(str(img_path))
            if original is None:
                continue

            # 1. Create Dark Version
            dark = darken_image(original)
            cv2.imwrite(str(folder / f"aug_dark_{img_path.name}"), dark)

            # 2. Create Noisy Version
            noisy = add_noise(original)
            cv2.imwrite(str(folder / f"aug_noise_{img_path.name}"), noisy)

            # 3. Create Blurry Version
            blur = blur_image(original)
            cv2.imwrite(str(folder / f"aug_blur_{img_path.name}"), blur)

    print("[INFO] Augmentation Complete! Re-run train_faces.py now.")

if __name__ == "__main__":
    process_dataset()