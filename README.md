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

## 4. How to run the project locally
**Prerequisite:**
- Having Docker and uv installed in your local computer

**Sync the package**
```
uv sync
```

**Activate the virtual env**
```
source .venv/bin/activate
```

**Start Milvus with Docker**
```
docker compose up -d
```
When the docker starts, you can access Milvus web UI here: http://127.0.0.1:9091/webui/

**Add the env into .env file**
- GEMINI_API_KEY

Noted that you can create a project and get the GEMINI_API_KEY for free with limited number of query from [here](https://aistudio.google.com/api-keys).

**Start the chainlit application**
```
chainlit run app.py --watch
```

## 5. Deployment plan

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
  

