FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
RUN mkdir -p /app/api/instance && chmod -R 777 /app/api/instance

RUN mkdir -p /app/.cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

COPY . .

EXPOSE 7860

CMD ["python", "-m", "api.app"]