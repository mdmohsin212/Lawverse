FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/huggingface /tmp/lawverse_data/gdown_cache && chmod -R 777 /tmp

ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV HF_DATASETS_CACHE=/tmp/huggingface/datasets
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface/hub
ENV GDOWN_CACHE=/tmp/lawverse_data/gdown_cache

COPY . .

EXPOSE 10000

CMD ["python", "app.py"]