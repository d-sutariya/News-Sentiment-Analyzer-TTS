FROM python:3.10-slim-bullseye AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt .

RUN pip install  --no-cache-dir -r requirements.txt 

COPY * /app/

FROM python:3.10-slim-bullseye
WORKDIR /app

COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app/* .

CMD [ "python" , "api.py" ]
