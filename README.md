# ws_project
writing styler


# Pull llama3.2:latest
```bash
docker run --rm -v /mnt/e/ai/ws_project/ollama_models:/root/.ollama ollama/ollama ollama pull llama3.2:latest
```

# Pull gpt-oss:latest
```bash
docker run --rm -v /mnt/e/ai/ws_project/ollama_models:/root/.ollama ollama/ollama ollama pull gpt-oss:latest
```

# Build the image
```bash
cd /mnt/e/ai/ws_project/webui_stack/
docker-compose up -d --build

docker-compose restart pipelines
```