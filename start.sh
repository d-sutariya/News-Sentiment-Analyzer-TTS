#!/bin/bash

# Start the backend API
python api.py &

# Start the frontend Streamlit app
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
