FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /.cache/gdown && chmod -R 777 /.cache
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
RUN mkdir -p /app/api/instance && chmod -R 777 /app/api/instance

RUN mkdir -p /app/.cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

ENV GDOWN_CACHE=/tmp/lawverse_data/gdown_cache

COPY . .

EXPOSE 10000

CMD ["python", "-m", "api.app"]