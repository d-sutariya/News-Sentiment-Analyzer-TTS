# Use an official Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory inside the container
WORKDIR /app

# Install system dependencies and Supervisor
RUN apt-get update && \
    apt-get install -y supervisor ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies with correct ownership
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY --chown=user . /app/

# Ensure the logs directory exists inside the container
RUN mkdir -p /app/logs

# Copy supervisord.conf into the container (adjust paths if needed)
COPY --chown=user supervisord.conf /etc/supervisord.conf

# Switch to the non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port used by your app (adjust if needed)
EXPOSE 8501

# Command to run Supervisor, which starts both your backend and frontend as defined in supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
