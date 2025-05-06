FROM python:3.10-slim-bullseye AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt .

RUN pip install  --no-cache-dir -r requirements.txt 

COPY * /app/

FROM python:3.10-slim-bullseye

RUN groupadd -r non_root_users

RUN useradd -r -g non_root_users user1

WORKDIR /app
RUN mkdir audio_data
RUN  chown -R user1:non_root_users /app

COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app/* .

USER user1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:5000/

CMD ["python","api.py"]
