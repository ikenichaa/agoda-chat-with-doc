import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from langchain_core.documents import Document

from indexing import (
    load_docs_and_chunk,
    parse_and_chunk_files,
    ingest_to_vectorstore,
    _log_sample_chunks
)


class TestLoadDocsAndChunk:
    """Test suite for load_docs_and_chunk function."""
    
    @patch('indexing.DoclingLoader')
    def test_load_docs_success(self, mock_loader_class):
        """Test successful document loading and chunking."""
        # Arrange
        mock_docs = [
            Document(page_content="Chunk 1", metadata={"page": 1}),
            Document(page_content="Chunk 2", metadata={"page": 2}),
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_docs
        mock_loader_class.return_value = mock_loader
        
        # Act
        result = load_docs_and_chunk("/path/to/test.pdf")
        
        # Assert
        assert len(result) == 2
        assert result[0].page_content == "Chunk 1"
        assert result[1].page_content == "Chunk 2"
        mock_loader_class.assert_called_once()
        mock_loader.load.assert_called_once()
    
    @patch('indexing.DoclingLoader')
    def test_load_docs_empty_result(self, mock_loader_class):
        """Test handling of documents with no extractable content."""
        # Arrange
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_class.return_value = mock_loader
        
        # Act
        result = load_docs_and_chunk("/path/to/empty.pdf")
        
        # Assert
        assert result == []
    
    @patch('indexing.DoclingLoader')
    def test_load_docs_exception(self, mock_loader_class):
        """Test exception handling during document loading."""
        # Arrange
        mock_loader_class.side_effect = Exception("Document loading failed")
        
        # Act & Assert
        with pytest.raises(Exception, match="Document loading failed"):
            load_docs_and_chunk("/path/to/broken.pdf")


class TestParseAndChunkFiles:
    """Test suite for parse_and_chunk_files async function."""
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    async def test_parse_single_file_success(self, mock_make_async, mock_message_class):
        """Test parsing a single file successfully.
        
        Note: One file can produce multiple chunks after processing.
        This test simulates a single PDF file being split into 2 chunks.
        """
        # Arrange
        mock_file = MagicMock()
        mock_file.name = "test.pdf"
        mock_file.path = "/tmp/test.pdf"
        
        # Simulate that the PDF file gets chunked into 2 Document objects
        mock_docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # Mock load_docs_and_chunk to return 2 chunks from 1 file
        mock_load_func = AsyncMock(return_value=mock_docs)
        mock_make_async.return_value = mock_load_func
        
        # Act
        result = await parse_and_chunk_files([mock_file])
        
        # Assert
        # We expect 2 chunks because the single file was chunked into 2 pieces
        assert len(result) == 2
        # Both chunks should have the same source file name
        assert result[0].metadata['source'] == "test.pdf"
        assert result[0].metadata['original_path'] == "/tmp/test.pdf"
        assert result[1].metadata['source'] == "test.pdf"
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    async def test_parse_multiple_files(self, mock_make_async, mock_message_class):
        """Test parsing multiple files successfully."""
        # Arrange
        mock_file1 = MagicMock()
        mock_file1.name = "doc1.pdf"
        mock_file1.path = "/tmp/doc1.pdf"
        
        mock_file2 = MagicMock()
        mock_file2.name = "doc2.pdf"
        mock_file2.path = "/tmp/doc2.pdf"
        
        mock_docs1 = [Document(page_content="Doc1 Content", metadata={})]
        mock_docs2 = [Document(page_content="Doc2 Content", metadata={})]
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # Mock load_docs_and_chunk to return different docs
        mock_load_func = AsyncMock(side_effect=[mock_docs1, mock_docs2])
        mock_make_async.return_value = mock_load_func
        
        # Act
        result = await parse_and_chunk_files([mock_file1, mock_file2])
        
        # Assert
        assert len(result) == 2
        assert result[0].metadata['source'] == "doc1.pdf"
        assert result[1].metadata['source'] == "doc2.pdf"
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    async def test_parse_partial_failure(self, mock_make_async, mock_message_class):
        """Test handling when some files fail but others succeed."""
        # Arrange
        mock_file1 = MagicMock()
        mock_file1.name = "good.pdf"
        mock_file1.path = "/tmp/good.pdf"
        
        mock_file2 = MagicMock()
        mock_file2.name = "bad.pdf"
        mock_file2.path = "/tmp/bad.pdf"
        
        mock_docs = [Document(page_content="Good content", metadata={})]
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # First file succeeds, second fails
        mock_load_func = AsyncMock(side_effect=[mock_docs, Exception("Parse error")])
        mock_make_async.return_value = mock_load_func
        
        # Act
        result = await parse_and_chunk_files([mock_file1, mock_file2])
        
        # Assert - Should still return the successful file's chunks
        assert len(result) == 1
        assert result[0].metadata['source'] == "good.pdf"
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    async def test_parse_all_files_fail(self, mock_make_async, mock_message_class):
        """Test handling when all files fail to parse."""
        # Arrange
        mock_file = MagicMock()
        mock_file.name = "bad.pdf"
        mock_file.path = "/tmp/bad.pdf"
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # Mock load_docs_and_chunk to fail
        mock_load_func = AsyncMock(side_effect=Exception("All files failed"))
        mock_make_async.return_value = mock_load_func
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to extract content from all files"):
            await parse_and_chunk_files([mock_file])


class TestIngestToVectorstore:
    """Test suite for ingest_to_vectorstore function."""
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    @patch('indexing.HuggingFaceEmbeddings')
    async def test_ingest_success(self, mock_embeddings_class, mock_make_async, mock_message_class):
        """Test successful ingestion into vector store."""
        # Arrange
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={"source": "test.pdf"}),
            Document(page_content="Chunk 2", metadata={"source": "test.pdf"}),
        ]
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # Mock embeddings
        mock_embedding = MagicMock()
        mock_embeddings_class.return_value = mock_embedding
        
        # Mock Milvus.from_documents
        mock_vectorstore = MagicMock()
        mock_from_docs = AsyncMock(return_value=mock_vectorstore)
        mock_make_async.return_value = mock_from_docs
        
        # Act
        result = await ingest_to_vectorstore(mock_chunks)
        
        # Assert
        assert result == mock_vectorstore
        mock_embeddings_class.assert_called_once()
        mock_from_docs.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest_empty_chunks(self):
        """Test handling of empty chunks list."""
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot ingest empty chunks list"):
            await ingest_to_vectorstore([])
    
    @pytest.mark.asyncio
    @patch('indexing.cl.Message')
    @patch('indexing.cl.make_async')
    @patch('indexing.HuggingFaceEmbeddings')
    async def test_ingest_connection_error(self, mock_embeddings_class, mock_make_async, mock_message_class):
        """Test handling of Milvus connection failure."""
        # Arrange
        mock_chunks = [Document(page_content="Test", metadata={})]
        
        # Mock cl.Message
        mock_msg = AsyncMock()
        mock_msg.send = AsyncMock(return_value=mock_msg)
        mock_msg.update = AsyncMock()
        mock_message_class.return_value = mock_msg
        
        # Mock embeddings
        mock_embeddings_class.return_value = MagicMock()
        
        # Mock Milvus.from_documents to raise ConnectionError
        mock_from_docs = AsyncMock(side_effect=ConnectionError("Cannot connect to Milvus"))
        mock_make_async.return_value = mock_from_docs
        
        # Act & Assert
        with pytest.raises(ConnectionError, match="Failed to connect to Milvus"):
            await ingest_to_vectorstore(mock_chunks)


class TestLogSampleChunks:
    """Test suite for _log_sample_chunks helper function."""
    
    def test_log_sample_chunks(self, caplog):
        """Test logging of sample chunks."""
        # Arrange
        chunks = [
            Document(page_content="Short chunk", metadata={"source": "test.pdf"}),
            Document(page_content="A" * 100, metadata={"source": "test.pdf"}),
        ]
        
        # Act
        with caplog.at_level("DEBUG"):
            _log_sample_chunks(chunks, sample_size=2)
        
        # Assert
        assert "Sample of first 2 chunks" in caplog.text
        assert "Short chunk" in caplog.text
        assert "test.pdf" in caplog.text
