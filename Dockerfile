FROM python:3.8-alpine

WORKDIR /sketch2render2market

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY controlnet /sketch2render2market
COPY sam /sketch2render2market

# download checkpoints
# RUN curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth --output checkpoints/sam_vit_b_01ec64.pth
# RUN curl https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth --output checkpoints/control_sd15_scribble.pth

CMD 
