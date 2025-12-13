"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file path for testing."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 mock content")
    return str(pdf_file)


@pytest.fixture
def mock_chunks():
    """Provide sample document chunks for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="This is the first chunk of content.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="This is the second chunk with more information.",
            metadata={"source": "test.pdf", "page": 2}
        ),
        Document(
            page_content="Final chunk with concluding information.",
            metadata={"source": "test.pdf", "page": 3}
        ),
    ]
