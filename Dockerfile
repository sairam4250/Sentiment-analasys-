# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to speed up runtime and support read-only file systems
RUN python -m nltk.downloader stopwords wordnet

# Copy the rest of the application files
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define healthcheck using Python to ensure the service is running
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
