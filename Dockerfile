# Use an official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /code/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make sure logs directory exists inside the container
RUN mkdir -p /code/logs

# Expose port (Streamlit by default runs on 8501)
EXPOSE 8501

# Start both frontend (streamlit) and backend (flask) using supervisord or a custom script
# For simplicity, here’s an example using a simple script to start both
COPY start.sh /code/start.sh
RUN chmod +x /code/start.sh

CMD ["bash", "start.sh"]
