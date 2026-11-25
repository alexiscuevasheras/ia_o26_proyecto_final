# Sirve la demo estática; los modelos llegan por el volumen ./modelos de compose.yaml.
FROM python:3.12-slim
WORKDIR /app
COPY main.py .
COPY web/ web/
EXPOSE 8000
CMD ["python", "main.py"]
