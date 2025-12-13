import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document

from retrieval import (
    format_docs_with_sources,
    format_sources_for_display,
    get_rag_response
)
from schema import SourceDocument, StructuredAnswer


class TestFormatDocsWithSources:
    """Test suite for format_docs_with_sources function."""
    
    def test_format_single_doc(self):
        """Test formatting a single document."""
        # Arrange
        docs = [
            Document(
                page_content="This is the content",
                metadata={"source": "test.pdf"}
            )
        ]
        
        # Act
        result = format_docs_with_sources(docs)
        
        # Assert
        assert "[1]" in result
        assert "test.pdf" in result
        assert "This is the content" in result
    
    def test_format_multiple_docs(self):
        """Test formatting multiple documents."""
        # Arrange
        docs = [
            Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
            Document(page_content="Content 3", metadata={"source": "doc1.pdf"}),
        ]
        
        # Act
        result = format_docs_with_sources(docs)
        
        # Assert
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result
        assert "Content 1" in result
        assert "Content 2" in result
        assert "Content 3" in result
        assert "---" in result  # Separator between docs
    
    def test_format_doc_without_source_metadata(self):
        """Test formatting document without source metadata."""
        # Arrange
        docs = [
            Document(page_content="Content", metadata={})
        ]
        
        # Act
        result = format_docs_with_sources(docs)
        
        # Assert
        assert "Unknown Source" in result
        assert "Content" in result
    
    def test_format_empty_list(self):
        """Test formatting an empty document list."""
        # Act
        result = format_docs_with_sources([])
        
        # Assert
        assert result == ""


class TestFormatSourcesForDisplay:
    """Test suite for format_sources_for_display function."""
    
    def test_format_single_source(self):
        """Test formatting a single source citation."""
        # Arrange
        sources = [
            SourceDocument(
                file_name="test.pdf",
                chunk_content="This is an excerpt from the document."
            )
        ]
        
        # Act
        result = format_sources_for_display(sources)
        
        # Assert
        assert result is not None
        assert "**Sources Cited**" in result
        assert "test.pdf" in result
        assert "This is an excerpt" in result
        assert "**Excerpt 1:**" in result
    
    def test_format_multiple_sources_same_file(self):
        """Test formatting multiple excerpts from the same file."""
        # Arrange
        sources = [
            SourceDocument(file_name="doc.pdf", chunk_content="Excerpt 1"),
            SourceDocument(file_name="doc.pdf", chunk_content="Excerpt 2"),
        ]
        
        # Act
        result = format_sources_for_display(sources)
        
        # Assert
        assert "doc.pdf" in result
        assert "**Excerpt 1:**" in result
        assert "**Excerpt 2:**" in result
        assert result.count("doc.pdf") == 1  # File name should appear once
    
    def test_format_multiple_sources_different_files(self):
        """Test formatting excerpts from different files."""
        # Arrange
        sources = [
            SourceDocument(file_name="doc1.pdf", chunk_content="Content from doc1"),
            SourceDocument(file_name="doc2.pdf", chunk_content="Content from doc2"),
        ]
        
        # Act
        result = format_sources_for_display(sources)
        
        # Assert
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result
        assert "Content from doc1" in result
        assert "Content from doc2" in result
    
    def test_format_empty_sources(self):
        """Test formatting when no sources are cited."""
        # Act
        result = format_sources_for_display([])
        
        # Assert
        assert result is None
    
    def test_format_sources_with_whitespace(self):
        """Test that source content whitespace is properly stripped."""
        # Arrange
        sources = [
            SourceDocument(
                file_name="test.pdf",
                chunk_content="  \n  Content with whitespace  \n  "
            )
        ]
        
        # Act
        result = format_sources_for_display(sources)
        
        # Assert
        assert "Content with whitespace" in result
        # Check that leading/trailing whitespace is handled


class TestGetRagResponse:
    """Test suite for get_rag_response async function."""
    
    @pytest.mark.asyncio
    async def test_get_rag_success(self):
        """Test successful RAG response generation."""
        # Arrange
        question = "What is the refund policy?"
        
        mock_retrieved_docs = [
            Document(
                page_content="Refunds are processed within 5 business days.",
                metadata={"source": "policy.pdf"}
            ),
            Document(
                page_content="Full refunds available within 30 days of purchase.",
                metadata={"source": "policy.pdf"}
            ),
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_retrieved_docs
        
        mock_structured_response = StructuredAnswer(
            answer="Refunds are processed within 5 business days and full refunds are available within 30 days of purchase.",
            sources_cited=[
                SourceDocument(
                    file_name="policy.pdf",
                    chunk_content="Refunds are processed within 5 business days."
                )
            ]
        )
        
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = mock_structured_response
        
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        # Mock the chain construction
        with patch('retrieval.RunnablePassthrough') as mock_runnable:
            mock_runnable.assign.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            
            # Act
            result = await get_rag_response(question, mock_retriever, mock_llm)
        
        # Assert
        assert result["answer"] == mock_structured_response.answer
        assert len(result["sources_cited"]) == 1
        assert result["sources_cited"][0].file_name == "policy.pdf"
        mock_retriever.invoke.assert_called_once_with(question)
    
    @pytest.mark.asyncio
    async def test_get_rag_no_documents_retrieved(self):
        """Test handling when no documents are retrieved."""
        # Arrange
        question = "What is the weather today?"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []  # No docs retrieved
        
        mock_llm = MagicMock()
        
        # Act
        result = await get_rag_response(question, mock_retriever, mock_llm)
        
        # Assert
        assert "couldn't find any relevant information" in result["answer"]
        assert result["sources_cited"] == []
    
    @pytest.mark.asyncio
    async def test_get_rag_empty_question(self):
        """Test handling of empty question."""
        # Arrange
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await get_rag_response("", mock_retriever, mock_llm)
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await get_rag_response("   ", mock_retriever, mock_llm)
    
    @pytest.mark.asyncio
    async def test_get_rag_retriever_exception(self):
        """Test handling when retriever raises an exception."""
        # Arrange
        question = "Test question"
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = Exception("Retriever failed")
        
        mock_llm = MagicMock()
        
        # Act & Assert
        with pytest.raises(Exception, match="Retriever failed"):
            await get_rag_response(question, mock_retriever, mock_llm)
    
    @pytest.mark.asyncio
    async def test_get_rag_llm_exception(self):
        """Test handling when LLM invocation fails."""
        # Arrange
        question = "Test question"
        
        mock_retrieved_docs = [
            Document(page_content="Content", metadata={"source": "test.pdf"})
        ]
        
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_retrieved_docs
        
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("LLM invocation failed")
        
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        # Mock the chain construction
        with patch('retrieval.RunnablePassthrough') as mock_runnable:
            mock_runnable.assign.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            
            # Act & Assert
            with pytest.raises(Exception, match="LLM invocation failed"):
                await get_rag_response(question, mock_retriever, mock_llm)
