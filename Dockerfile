FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

RUN pip install --no-cache-dir hatchling ninja setuptools && \
    pip install --no-cache-dir nemotron-ocr==1.0.1 && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow

COPY server.py /workspace/server.py

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
