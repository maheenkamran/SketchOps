FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

ARG CLIP_MODEL=google/siglip-base-patch16-224
ARG CLIP_REVISION=7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
AutoImageProcessor.from_pretrained('${CLIP_MODEL}', revision='${CLIP_REVISION}'); \
AutoModel.from_pretrained('${CLIP_MODEL}', revision='${CLIP_REVISION}')"
ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["bash", "run_pipeline.sh"]
