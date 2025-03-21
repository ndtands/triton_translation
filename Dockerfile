FROM nvcr.io/nvidia/tritonserver:23.02-py3

ENV PYTHONIOENCODING "UTF-8"

ENV PYTORCH_NVFUSER_DISABLE "fallback"

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libmagic-dev


RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . /workspace