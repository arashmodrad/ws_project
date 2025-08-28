FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/open-webui/pipelines.git .

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the pipeline server runs on
EXPOSE 9099

CMD ["python", "main.py"]