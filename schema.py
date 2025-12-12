from pydantic import BaseModel, Field
from typing import List

class SourceDocument(BaseModel):
    """A document fragment used as a source for the answer."""
    file_name: str = Field(description="The source file name, e.g., 'policy.pdf'.")
    chunk_content: str = Field(description="The direct text excerpt taken from the file that supports the answer.")

class StructuredAnswer(BaseModel):
    """The final structured response from the LLM."""
    answer: str = Field(description="The complete, conversational answer to the user's question, strictly based on the context. Use the full 'I cannot answer...' phrase for failure cases.")
    sources_cited: List[SourceDocument] = Field(description="A list of ALL source documents used to formulate the answer. This list MUST be empty if the answer field indicates a failure (e.g., 'I cannot answer...').")