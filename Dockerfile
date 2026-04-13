FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
    python3.12 python3.12-dev \
    build-essential ninja-build git curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN python3.12 -m pip install --no-cache-dir hatchling ninja setuptools wheel

RUN python3.12 -m pip install --no-cache-dir --no-build-isolation \
    nemotron-ocr==1.0.1 \
    "fastapi>=0.100.0" "uvicorn[standard]" python-multipart pillow

WORKDIR /workspace
COPY server.py /workspace/server.py

EXPOSE 8001

CMD ["python3.12", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
