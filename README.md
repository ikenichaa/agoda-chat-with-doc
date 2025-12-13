# Chat with your document

## 1. Project Overview
This project is a web application that allows users to upload multiple text documents and ask document-related questions of the system. The system is able to answer user queries and provide an excerpt from the paragraph that contains the answer.

## 2. System Architecture
To enable knowledge retrieval from input documents via natural language queries, RAG must be used. RAG is the concept in which the system receives the user's query and retrieves similar documents from a vector database. Then, the top-k most similar documents to the query are passed to the prompt so the LLM can augment the user request with additional context. Finally, the LLM generates the answer and returns the response to the user.

Below is the summary of the process:
```
User uploads → chunk → embed → store → retrieve → LLM → answer
```

The system architecture of this project is divided into two phases: document storage and user retrieval. Below, I outline the high-level design of this application. The reasoning behind each tool will be discussed later in the tech stack section.

### 2.1 Document Storing
![Document Storing](/images/document_storing.png)
In this first stage, the users are asked to input text documents (maximum 3) through the UI to start a conversation with the bot. After that, Docling is used to parse and chunk the input documents, including metadata such as the file name so the user can trace the source later. The resulting chunks are then passed to the embedding model, sentence-transformers/all-MiniLM-L6-v2, which converts them into vectors. The first stage is complete once the vectors, along with their metadata, are successfully stored in Milvus, the vector database.

### 2.2 User Retrieval

After the document content is successfully stored inside the vector database, it is now time for the user to ask questions.
![Document Storing](/images/user_retrieval.png)
Users can input a query into the system. Then, using the LangChain Retriever, the query is automatically embedded using the same embedding model that was used during the document storage phase. Using the same model ensures that the input query is embedded in a compatible way, which guarantees correct similarity scoring. After that, the top-k retrieved chunks are inserted into the prompt along with the question. The responses are formatted using Pydantic to ensure that the LLM output always includes both the answer to the question and the excerpt the answer is based on, so the user can recheck the source.


## 3. Tech Stack
- UI: Chainlit
- Embedding model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- LLM Orchestration: LangChain
  - Retriever https://docs.langchain.com/oss/python/integrations/retrievers
- Vector Database: Milvus

## 4. How to run the project

### Option 1: Run with Docker (Recommended)

**Prerequisites:**
- Docker and Docker Compose installed

**Steps:**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agoda-chat-with-doc
   ```

2. **Create .env file**
   ```bash
   cp .env.example .env
   # Add your GOOGLE_API_KEY to .env
   ```
   
   Get your free API key from [Google AI Studio](https://aistudio.google.com/api-keys)

3. **Start all services**
   ```bash
   docker compose up -d
   ```

4. **Access the application**
   - App: http://localhost:8000
   - Milvus UI: http://localhost:9091/webui/

5. **View logs**
   ```bash
   docker compose logs -f app
   ```

6. **Stop services**
   ```bash
   docker compose down
   ```

### Option 2: Run Locally (Development)

**Prerequisites:**
- Python 3.11+
- Docker (for Milvus only)
- uv package manager

**Steps:**

1. **Install dependencies**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

2. **Start Milvus**
   ```bash
   docker compose up -d etcd minio standalone
   ```

3. **Create .env file**
   ```bash
   cp .env.example .env
   # Add your GOOGLE_API_KEY
   ```

4. **Run the app**
   ```bash
   chainlit run app.py --watch
   ```

5. **Access**: http://localhost:8000

## 5. Deployment

### Docker Image
The application is containerized and can be deployed to any platform supporting Docker:

**Build image:**
```bash
docker build -t agoda-chat-app .
```

**Run container:**
```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e MILVUS_URI=http://milvus:19530 \
  agoda-chat-app
```

### CI/CD Pipeline
GitHub Actions workflow included (`.github/workflows/ci-cd.yml`):
- ✅ Automated testing on push
- ✅ Docker image build and push to GitHub Container Registry
- ✅ Production deployment (manual approval required)

### Cloud Deployment Options

**Google Cloud Run:**
```bash
gcloud run deploy agoda-chat-app \
  --image gcr.io/your-project/agoda-chat-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**AWS ECS/Fargate:**
```bash
# Use provided Dockerfile
aws ecs create-service --cluster your-cluster \
  --service-name agoda-chat-app \
  --task-definition agoda-chat-app
```

**Azure Container Apps:**
```bash
az containerapp create \
  --name agoda-chat-app \
  --resource-group your-rg \
  --image your-registry/agoda-chat-app
```

### Quick Deploy Script
```bash
./deploy.sh
```

## 6. Limitation and future development




# TO BE DELETED
```
brew install tesseract
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
  

