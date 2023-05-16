from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SkecthRenderDataset(Dataset):
    def __init__(self, root_A, root_B, img_size, logger):
        self.root_A = root_A
        self.root_B = root_B

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)
        logger.info(f"A data: {self.A_len}")
        logger.info(f"B data: {self.B_len}")

        self.transform = A.Compose(
            [
                A.Resize(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )


    def __len__(self):
        return self.length_dataset


    def __getitem__(self, index):
        A_img = self.A_images[index % self.A_len]   # since len of animation and reality is not same
        B_img = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        # transform
        augmentations = self.transform(image=A_img, image0=B_img)
        A_img = augmentations["image"]
        B_img = augmentations["image0"]

        return A_img, B_img


