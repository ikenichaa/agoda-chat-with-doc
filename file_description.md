### File Descriptions

#### Core Application Files

- **`app.py`**: The main application file that defines the Chainlit UI interface. Contains two primary handlers:

  - `@cl.on_chat_start`: Initializes the chat session, handles file uploads, processes PDFs, and sets up the RAG system
  - `@cl.on_message`: Handles user queries, executes the RAG pipeline, and returns answers with source citations

- **`config.py`**: Centralized configuration management including:

  - Model settings (embedding model, LLM model)
  - Vector database connection parameters
  - Retrieval settings (top-k)
  - RAG prompt templates

- **`indexing.py`**: Handles the document ingestion pipeline:

  - `load_docs_and_chunk()`: Uses Docling to parse PDFs and chunk them with HybridChunker
  - `parse_and_chunk_files()`: Processes multiple uploaded files asynchronously
  - `ingest_to_vectorstore()`: Embeds chunks and stores them in Milvus vector database

- **`retrieval.py`**: Implements the RAG retrieval logic:

  - `format_docs_with_sources()`: Formats retrieved documents with source labels
  - `format_sources_for_display()`: Creates markdown-formatted source citations for UI
  - `get_rag_response()`: Executes the complete RAG pipeline (retrieve → format → generate)

- **`schema.py`**: Defines Pydantic models for structured outputs:

  - `SourceDocument`: Represents a document fragment with file name and content
  - `StructuredAnswer`: Ensures LLM responses always include answer and source citations

- **`error_handler.py`**: Provides centralized error handling with:
  - Categorized error messages (validation, connection, generic)
  - Logging integration
  - User-friendly error display in Chainlit UI

#### Configuration Files

- **`pyproject.toml`**: Python project metadata and dependencies (managed by `uv`)
- **`docker-compose.yml`**: Defines services (app, Milvus, etcd, MinIO) for containerized deployment
- **`Dockerfile`**: Container image for the Chainlit application
- **`makefile`**: Convenient commands for Docker operations (`make up`, `make down`, `make logs`)
- **`.env.example`**: Template for required environment variables (GOOGLE_API_KEY, etc.)
