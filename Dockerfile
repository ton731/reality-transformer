# FROM python:3.8-slim
FROM pytorch/pytorch:latest

RUN apt-get -y update
RUN apt-get -y install git
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /sketch2render2market

COPY controlnet /sketch2render2market/controlnet
COPY sam /sketch2render2market/sam
COPY sketch.png /sketch2render2market/sketch.png
COPY fake_app.py /sketch2render2market/fake_app.py
COPY test.py /sketch2render2market/test.py

# download checkpoints
# RUN curl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth --output checkpoints/sam_vit_b_01ec64.pth
# RUN curl https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth --output checkpoints/control_sd15_scribble.pth

CMD [ "python", "test.py" ]
