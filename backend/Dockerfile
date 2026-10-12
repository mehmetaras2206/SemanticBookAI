FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models FASTEMBED_CACHE_PATH=/models/fastembed
WORKDIR /srv/backend

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY backend/start.sh ./start.sh
COPY data/books.yaml /srv/data/books.yaml
RUN chmod +x start.sh

ENV DATA_DIR=/srv/data QDRANT_PATH=/srv/data/qdrant
EXPOSE 8000
CMD ["./start.sh"]
