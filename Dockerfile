FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train inside the image so the container ships with a ready-to-serve model
RUN python train.py

# 8501 = Streamlit UI, 8000 = Prometheus /metrics
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "run.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
