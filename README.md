# Installation
```
brew install tesseract
```

# Run command
```
chainlit run app.py --watch
```


Note
- Cannot use chroma db because it doesn't support complex metadata that docling provided
- Milvus
  - Download the configuration file: 
  ```
  wget https://github.com/milvus-io/milvus/releases/download/v2.6.7/milvus-standalone-docker-compose.yml -O docker-compose.yml
  ```
  - This is the link to access the web-ui: http://127.0.0.1:9091/webui/
  - Index: https://milvus.io/docs/index.md?tab=floating
  

