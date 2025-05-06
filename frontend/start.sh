#!/bin/bash

# Ensure audio_data exists and has proper ownership
mkdir -p /app/audio_data
chown -R user1:non_root_users /app/audio_data

echo "Starting Streamlit App as user: $(whoami)"
echo "audio_data directory:"
ls -ld /app/audio_data

# Run streamlit app
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
