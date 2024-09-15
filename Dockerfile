FROM python:3.12

RUN apt-get update && apt-get install -y \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV MODEL=jinaai/jina-clip-v1

ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

RUN pip install --no-cache-dir huggingface_hub[cli]
RUN huggingface-cli download $MODEL

ENV HF_HUB_OFFLINE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]