FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ffmpeg for pitch-preserving speed + audio encoding; libgomp1/libsndfile1 for torch/soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        libgomp1 \
        libsndfile1 \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py start.sh status.sh ./

EXPOSE 8998

CMD ["python", "app.py"]
