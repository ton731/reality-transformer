from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RealityAnimationDataset(Dataset):
    def __init__(self, root_reality, root_animation, img_size):
        self.root_animation = root_animation
        self.root_reality = root_reality

        self.animation_images = os.listdir(root_animation)
        self.reality_images = os.listdir(root_reality)
        self.length_dataset = max(len(self.animation_images), len(self.reality_images))
        self.animation_len = len(self.animation_images)
        self.reality_len = len(self.reality_images)

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
        animation_img = self.animation_images[index % self.animation_len]   # since len of animation and reality is not same
        reality_img = self.reality_images[index % self.reality_len]

        animation_path = os.path.join(self.root_animation, animation_img)
        reality_path = os.path.join(self.root_reality, reality_img)

        animation_img = np.array(Image.open(animation_path).convert("RGB"))
        reality_img = np.array(Image.open(reality_path).convert("RGB"))

        # transform
        augmentations = self.transform(image=animation_img, image0=reality_img)
        animation_img = augmentations["image"]
        reality_img = augmentations["image0"]

        return animation_img, reality_img


