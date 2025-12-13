# Tests

This directory contains unit tests for the Agoda Chat with Doc project.

## Setup

Install test dependencies:

```bash
uv sync --extra dev
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run with verbose output:

```bash
uv run pytest -v
```

Run specific test file:

```bash
uv run pytest tests/test_indexing.py
```

Run specific test class:

```bash
uv run pytest tests/test_indexing.py::TestLoadDocsAndChunk
```

Run with coverage:

```bash
uv run pytest --cov=. --cov-report=html
```

## Test Structure

- **`test_indexing.py`** - Tests for document parsing and vector store ingestion
  - `TestLoadDocsAndChunk` - Tests for loading and chunking documents
  - `TestParseAndChunkFiles` - Tests for async file processing
  - `TestIngestToVectorstore` - Tests for Milvus vector store ingestion

- **`test_retrieval.py`** - Tests for RAG retrieval and response generation
  - `TestFormatDocsWithSources` - Tests for document formatting
  - `TestFormatSourcesForDisplay` - Tests for source citation formatting
  - `TestGetRagResponse` - Tests for end-to-end RAG pipeline

- **`conftest.py`** - Shared pytest fixtures and configuration

## Test Coverage

The tests cover:

✅ **Document Processing**
- Loading and chunking PDFs with Docling
- Handling empty documents
- Error handling for corrupted files
- Multi-file processing with partial failures

✅ **Vector Store Operations**
- Successful ingestion to Milvus
- Connection error handling
- Empty chunk validation

✅ **RAG Pipeline**
- Document retrieval
- Response generation with LLM
- Source citation formatting
- Error handling for retriever and LLM failures

## Mocking Strategy

The tests use extensive mocking to avoid:
- Actual PDF file processing (slow)
- Real Milvus database connections
- Real LLM API calls (expensive)
- Chainlit UI interactions

Key mocked components:
- `DoclingLoader` - Document loading
- `Milvus.from_documents` - Vector store creation
- `HuggingFaceEmbeddings` - Embedding model
- `cl.Message` - Chainlit UI messages
- LLM chain - Response generation

## Writing New Tests

When adding new tests:

1. Use appropriate fixtures from `conftest.py`
2. Mock external dependencies (APIs, databases, file I/O)
3. Test both success and failure scenarios
4. Use descriptive test names following pattern: `test_<function>_<scenario>`
5. Add docstrings explaining what each test validates

Example:

```python
@pytest.mark.asyncio
async def test_my_function_success(self):
    """Test successful execution of my_function."""
    # Arrange
    mock_input = MagicMock()
    
    # Act
    result = await my_function(mock_input)
    
    # Assert
    assert result is not None
    mock_input.method.assert_called_once()
```
