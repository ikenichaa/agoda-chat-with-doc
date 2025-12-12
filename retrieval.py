import logging
from typing import Dict, List

from langchain_core.runnables import RunnablePassthrough

from config import RAG_PROMPT
from schema import StructuredAnswer


def format_docs_with_sources(docs):
    """
    format_docs_with_sources prepares the retrieved documents into a formatted string with source labels
    
    :param docs: List of document chunks to format
    ex: [Document(page_content='...', metadata={'source': 'doc1.pdf'}), ...]
    """
    formatted_context = []
    for i, d in enumerate(docs):
        # Create a unique label for the source (e.g., [1], [2])
        source_label = f"[{i+1}]"
        
        # Get the file name from metadata
        # Common key is 'source', 'file_name', or 'id' depending on your loader
        file_name = d.metadata.get('source', 'Unknown Source')
        
        # Combine the source info and the content
        formatted_context.append(f"{source_label} (File: {file_name}):\n{d.page_content}")
    
    # Join all formatted chunks with a clear separator
    return "\n\n---\n\n".join(formatted_context)


def format_sources_for_display(sources_cited):
    """Format cited sources into a readable markdown string for UI display.
    
    Args:
        sources_cited: List of SourceDocument objects
        
    Returns:
        Formatted markdown string with sources grouped by file, or None if empty
    """
    if not sources_cited:
        return None
    
    # Aggregate excerpts by file name
    file_excerpts: Dict[str, List[str]] = {}
    for source in sources_cited:
        if source.file_name not in file_excerpts:
            file_excerpts[source.file_name] = []
        file_excerpts[source.file_name].append(source.chunk_content)
    
    # Build markdown string
    sources_text = "**Sources Cited**:\n\n"
    for file_name, contents in file_excerpts.items():
        sources_text += f"**File:** `{file_name}`\n"
        for i, content in enumerate(contents, start=1):
            sources_text += f"> **Excerpt {i}:** {content.strip()}\n"
        sources_text += "\n"
    
    return sources_text


async def get_rag_response(question: str, retriever, llm) -> Dict[str, any]:
    """Execute the RAG pipeline: retrieve relevant documents, generate structured answer.
    
    Args:
        question: User's question
        retriever: Vector store retriever instance
        llm: Language model instance
        
    Returns:
        Dictionary with 'answer' and 'sources_cited' keys
        
    Raises:
        ValueError: If no documents retrieved or question is empty
        Exception: If LLM invocation fails
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    logging.info(f"Processing question: {question[:100]}...")
    
    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            logging.warning("No documents retrieved from vector store")
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources_cited": []
            }
        
        # Step 2: Format documents into context string
        context_string = format_docs_with_sources(retrieved_docs)
        logging.info(f"Retrieved {len(retrieved_docs)} relevant document chunks")
        logging.debug(f"Context string length: {len(context_string)} characters")
        
        # Step 3: Create structured LLM and RAG chain
        structured_llm = llm.with_structured_output(StructuredAnswer)
        rag_chain = (
            RunnablePassthrough.assign(context=lambda x: context_string)
            | RAG_PROMPT
            | structured_llm
        )
        
        # Step 4: Invoke the chain to get structured output
        logging.info("Invoking LLM for structured response...")
        structured_response: StructuredAnswer = await rag_chain.ainvoke({"question": question})
        
        logging.info(f"LLM response received with {len(structured_response.sources_cited)} sources cited")
        logging.debug(f"Answer preview: {structured_response.answer[:100]}...")
        
        # Step 5: Return as dictionary
        return {
            "answer": structured_response.answer,
            "sources_cited": structured_response.sources_cited
        }
        
    except Exception as e:
        logging.error(f"RAG pipeline failed for question '{question[:50]}...': {str(e)}", exc_info=True)
        raise