#!/bin/bash
# Shell script to build and run Streamlit app in Docker

echo "1. Cleaning up existing container if any..."
docker stop sentiment-analysis-web 2>/dev/null || true
docker rm sentiment-analysis-web 2>/dev/null || true

echo "2. Building Docker image..."
docker build -t sentiment-analysis-app .

echo "3. Running Docker container on http://localhost:8501..."
docker run -d \
  -p 8501:8501 \
  --name sentiment-analysis-web \
  --restart always \
  sentiment-analysis-app

echo "Success! The application is running at http://localhost:8501"
