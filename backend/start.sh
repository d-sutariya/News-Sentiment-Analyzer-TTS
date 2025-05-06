#!/bin/bash

# Ensure the audio_data directory exists and has the correct permissions
mkdir -p /app/audio_data
chown -R user1:non_root_users /app/audio_data

# Optional: Debug output
echo "Starting backend as user: $(whoami)"
echo "audio_data owner:"
ls -ld /app/audio_data

# Run your backend app
exec python api.py
