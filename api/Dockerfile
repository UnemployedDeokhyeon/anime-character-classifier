FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/

ENV ONNX_PATH=checkpoints/model.onnx
ENV INDEX_PATH=checkpoints/index.faiss
ENV LABELS_PATH=checkpoints/labels.npy

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
