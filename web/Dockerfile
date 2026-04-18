FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence-transformers and git (needed for pip git installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install web requirements (includes query-classifier from GitHub)
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

# Copy the web app
COPY web/ /app/web/

EXPOSE 8000

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
