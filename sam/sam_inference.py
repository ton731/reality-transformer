from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2


print("***** Loading Segment Anything model......")
sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)



def inference(image_rgb: np.array):
    masks = mask_generator.generate(image_rgb)
    return masks



if __name__ == "__main__":
    test_img_path = "fake_interior_design.png"
    image = cv2.imread(test_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("input image:", image.shape)
    results = inference(image)
    print("result:", results)