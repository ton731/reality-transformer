from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


print("***** Loading ControlNet model......")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model('controlnet/models/cldm_v15.yaml').to(device)
model.load_state_dict(load_state_dict('checkpoints/control_sd15_scribble.pth', location=device))
ddim_sampler = DDIMSampler(model)


def inference(input_image=None, 
              prompt='',
              a_prompt='best quality, extremely detailed', 
              n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 
              num_samples=1, 
              image_resolution=256, 
              ddim_steps=5, 
              guess_mode=False, 
              strength=1.0, 
              scale=9.0, 
              seed=1234567, 
              eta=0.0):
    
    assert input_image.shape[2] == 3, "image should be in numpy with shape [h, w, 3]!"

    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        # control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.from_numpy(detected_map.copy()).float() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    return results



if __name__ == "__main__":
    test_img_path = "test_imgs/room.png"
    image = cv2.imread(test_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("input image:", image.shape)
    results = inference(input_image=image)
    print("result num:", len(results))
    print("result image:", results[0].shape)
