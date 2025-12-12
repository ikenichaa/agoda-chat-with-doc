import logging

import chainlit as cl
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from config import COLLECTION_NAME, EMBED_MODEL_ID, MILVUS_INDEX_PARAMS, MILVUS_URI


def load_docs_and_chunk(path: str):
    """Load and chunk a document using Docling with HybridChunker.
    
    Args:
        path: Path to the document file
        
    Returns:
        List of chunked document objects
    """
    loader = DoclingLoader(path, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))
    return loader.load()


async def parse_and_chunk_files(files):
    """Parse and chunk uploaded PDF files into document chunks.
    
    Args:
        files: List of uploaded file objects from Chainlit
        
    Returns:
        List of all document chunks across all files
        
    Raises:
        ValueError: If no valid chunks could be extracted from any file
    """
    step_msg = await cl.Message(content="⏳ Step 1/2: Parsing and Chunking PDFs…").send()
    
    all_chunks = []
    failed_files = []
    
    for file in files:
        file_msg = await cl.Message(content=f"⏳ Parsing and chunking **{file.name}**…").send()
        
        try:
            # Load and chunk document
            docs = await cl.make_async(load_docs_and_chunk)(file.path)
            
            if not docs:
                logging.warning(f"No content extracted from {file.name}")
                failed_files.append(file.name)
                file_msg.content = f"⚠️ **{file.name}**: No content extracted"
                await file_msg.update()
                continue
            
            # Update metadata with original filename
            for doc in docs:
                doc.metadata['source'] = file.name
                doc.metadata['original_path'] = file.path
            
            all_chunks.extend(docs)
            
            # Update file processing status
            file_msg.content = f"✅ Parsed **{file.name}** → {len(docs)} chunks"
            await file_msg.update()
            logging.info(f"Successfully processed {file.name}: {len(docs)} chunks")
            
        except Exception as e:
            logging.error(f"Failed to process {file.name}: {str(e)}", exc_info=True)
            failed_files.append(file.name)
            file_msg.content = f"❌ **{file.name}**: Failed to process"
            await file_msg.update()
    
    # Check if we got any chunks
    if not all_chunks:
        error_msg = f"Failed to extract content from all files: {', '.join(failed_files)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Update overall progress
    status = f"✅ Step 1/2: Parsing complete. Total chunks: {len(all_chunks)}"
    if failed_files:
        status += f" (Failed: {', '.join(failed_files)})"
    step_msg.content = status
    await step_msg.update()
    
    # Log sample chunks for debugging
    _log_sample_chunks(all_chunks)
    
    return all_chunks


def _log_sample_chunks(chunks, sample_size=3):
    """Log sample chunks for debugging purposes."""
    logging.debug(f"--- Sample of first {sample_size} chunks (Total: {len(chunks)}) ---")
    for i, chunk in enumerate(chunks[:sample_size], start=1):
        content_preview = chunk.page_content[:50] + '...' if len(chunk.page_content) > 50 else chunk.page_content
        logging.debug(f"Chunk {i}: {repr(content_preview)}")
        logging.debug(f"  Source: {chunk.metadata.get('source', 'N/A')}")


async def ingest_to_vectorstore(chunks):
    """Ingest document chunks into Milvus vector store.
    
    Args:
        chunks: List of document chunks to ingest
        
    Returns:
        Configured Milvus vector store instance
        
    Raises:
        ConnectionError: If cannot connect to Milvus
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Cannot ingest empty chunks list")
    
    step_msg = await cl.Message(content="⏳ Step 2/2: Ingesting chunks into vector store…").send()
    
    try:
        # Initialize embedding model
        logging.info(f"Initializing embedding model: {EMBED_MODEL_ID}")
        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
        
        # Create vector store
        logging.info(f"Connecting to Milvus at {MILVUS_URI}")
        vectorstore = await cl.make_async(Milvus.from_documents)(
            documents=chunks,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params=MILVUS_INDEX_PARAMS,
            drop_old=True,
        )
        
        step_msg.content = f"✅ Step 2/2: Ingested {len(chunks)} chunks successfully."
        await step_msg.update()
        logging.info(f"Successfully created vector store with {len(chunks)} chunks")
        
        return vectorstore
        
    except ConnectionError as e:
        error_msg = f"Failed to connect to Milvus at {MILVUS_URI}: {str(e)}"
        logging.error(error_msg, exc_info=True)
        step_msg.content = "❌ Step 2/2: Failed to connect to vector store."
        await step_msg.update()
        raise ConnectionError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Failed to ingest chunks to vector store: {str(e)}"
        logging.error(error_msg, exc_info=True)
        step_msg.content = "❌ Step 2/2: Failed to ingest chunks."
        await step_msg.update()
        raise