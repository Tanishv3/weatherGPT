FROM python:3.12-slim

LABEL org.opencontainers.image.title="WeatherGPT" org.opencontainers.image.version="2.1-india-first-dynamic"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
