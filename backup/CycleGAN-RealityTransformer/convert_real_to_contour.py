import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def convert(img_path, save_img_path):

    # Load image
    img = cv2.imread(str(img_path))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    result = np.full_like(gray, 255)
    cv2.drawContours(result, contours, -1, (0, 0, 0), 1)

    # Display image
    cv2.imwrite(str(save_img_path), result)


src_root = Path("data/sketch2render/render_chair_images")
tgt_root = Path("data/sketch2render/sketch_chair_boundary_images")
tgt_root.mkdir(parents=True, exist_ok=True)

for img_name in tqdm(os.listdir(src_root)):
    src_img_path = src_root / img_name
    tgt_img_path = tgt_root / img_name
    convert(src_img_path, tgt_img_path)
