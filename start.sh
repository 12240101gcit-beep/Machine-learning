#!/bin/bash

echo "🚀 Starting Docker..."
docker compose up -d

echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 3

echo "🎵 Starting Streamlit app..."
streamlit run streamlit_app.py
