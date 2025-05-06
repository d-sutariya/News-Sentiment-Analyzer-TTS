FROM python:3.10-slim-bullseye AS builder

RUN groupadd -r non_root_users

RUN useradd -r user1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* 
    
# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY /app.py /app/


RUN mkdir audio_data
RUN chown -R user1:non_root_users /app

USER user1

HEALTHCHECK CMD nc -z localhost 8501 || exit 1

ENTRYPOINT ["streamlit","run","app.py","--server.port","8501"]