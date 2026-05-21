# PowerShell script to build and run Streamlit app in Docker

Write-Host "1. Cleaning up existing container if any..." -ForegroundColor Yellow
docker stop sentiment-analysis-web 2>$null
docker rm sentiment-analysis-web 2>$null

Write-Host "2. Building Docker image..." -ForegroundColor Cyan
docker build -t sentiment-analysis-app .

Write-Host "3. Running Docker container on http://localhost:8501..." -ForegroundColor Green
docker run -d `
  -p 8501:8501 `
  --name sentiment-analysis-web `
  --restart always `
  sentiment-analysis-app

Write-Host "Success! The application is running at http://localhost:8501" -ForegroundColor Green
