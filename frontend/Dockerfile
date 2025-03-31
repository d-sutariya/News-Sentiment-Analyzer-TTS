FROM python:3.10-slim-bullseye AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* 
    
# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY /app.py /app/

# FROM python:3.10-slim-bullseye
# WORKDIR /app

# COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# COPY --from=builder /app/* .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
