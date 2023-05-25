import sys
sys.path.append("./controlnet")
sys.path.append("./sam")

from controlnet import controlnet_inference
from sam import sam_inference

import numpy as np
import cv2
import matplotlib.pyplot as plt


# user upload a sketch
sketch_image = cv2.imread("sketch.png")


# render with ControlNet
render_images = controlnet_inference.inference(input_image=sketch_image)
print("generated render:", render_images[0].shape)
plt.imshow(render_images[0])
plt.savefig("results/render.png")
# plt.show()


# get the furnuture segments
segments = sam_inference.inference(render_images[0])
print("SAM segments:", segments)



