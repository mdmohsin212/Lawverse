FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /app/.cache \
    && chmod -R 777 /app/.cache

ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

COPY . .

EXPOSE 10000

CMD ["python", "app.py"]