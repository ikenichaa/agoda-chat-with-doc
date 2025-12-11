import chainlit as cl
import logging

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_docling import DoclingLoader
from langchain_milvus import Milvus
from docling.chunking import HybridChunker

logging.basicConfig(level=logging.DEBUG)


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MILVUS_URI = "http://localhost:19530" 
COLLECTION_NAME = "docling_rag_index"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)


def load_docs_and_chunk(path: str):
    """
    load_docs_and_chunk use docling to load and chunk documents to get the docling docs
    
    :param path: path to the document
    :type path: str
    """
    loader = DoclingLoader(path, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))
    return loader.load()


async def parse_and_chunk_files(files):
    """
    Parse and chunk PDF files into document chunks.
    
    :param files: List of uploaded file objects
    :return: List of all document chunks across all files
    """
    step_msg = await cl.Message(content="⏳ Step 1/2: Parsing and Chunking PDFs…").send()
    
    all_chunks = []
    for f in files:
        parse_msg = await cl.Message(content=f"⏳ Parsing and chunking **{f.name}**…").send()
        
        docs = await cl.make_async(load_docs_and_chunk)(f.path)
        all_chunks.extend(docs)

        parse_msg.content = f"✅ Finish parsing and chunking **{f.name}** → {len(docs)} document chunks" 
        await parse_msg.update()
    
    step_msg.content = f"✅ Step 1/2: Parsing and chunking complete. Total chunks: {len(all_chunks)}"
    await step_msg.update()
    
    # Log samples
    logging.debug(f"--- Sample of first 3 chunks (Total: {len(all_chunks)}) ---")
    for i, d in enumerate(all_chunks[:3]):
        logging.debug(f"Chunk {i+1}: {repr(d.page_content[:50] + '...')}") 
        logging.debug(f"Source: {d.metadata.get('source', 'N/A')}")
    
    return all_chunks


async def ingest_to_vectorstore(chunks):
    """
    Ingest document chunks into Milvus vector store.
    
    :param chunks: List of document chunks to ingest
    :return: Configured vector store instance
    """
    step_msg = await cl.Message(content="⏳ Step 2/2: Ingesting chunks into vector store…").send()
    
    milvus_conn_args = {"uri": MILVUS_URI}
    
    vectorstore = await cl.make_async(Milvus.from_documents)(
        documents=chunks,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_args=milvus_conn_args,
        index_params={"index_type": "IVF_FLAT", "metric_type": "L2"},
        drop_old=True,
    )
    
    step_msg.content = f"✅ Step 2/2: Successfully ingested {len(chunks)} chunks into vector store."
    await step_msg.update()
    
    return vectorstore


@cl.on_chat_start
async def start():
    """
    This function is called when the chat starts. It prompts the user to upload PDF files,
    processes the files, and sets up the vector store and LLM in the user session.
    """
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload 1 to 3 PDF files to begin!", accept=["application/pdf"], max_files=3
        ).send()

    processing_msg = await cl.Message(content=f"⏳ Received {len(files)} file(s). Processing…").send()

    # Step 1: Parse and chunk
    all_chunks = await parse_and_chunk_files(files)

    # Step 2: Ingest to vector store
    vector_store = await ingest_to_vectorstore(all_chunks)

    processing_msg.content = "✅ All processing complete!"
    await processing_msg.update()

    # Store the retrievers in the Chainlit user session
    cl.user_session.set("vector_store", vector_store.as_retriever(search_kwargs={"k": 5}))
    
    # Initialize the LLM and store in user session
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    cl.user_session.set("llm", llm)
    
    # Notify user that setup is complete
    await cl.Message(content="Setup complete! Ready to chat.").send()

def log_retrieved_docs(retrieved_docs):
    """
    Log the content and metadata of retrieved document chunks.
    
    :param retrieved_docs: List of retrieved document chunks
    """
    logging.debug("--- Retrieved Document Chunks (Context) ---")
    for i, doc in enumerate(retrieved_docs):
        # Print the source file path (from metadata) and a snippet of the content
        source_file = doc.metadata.get('source', 'N/A')
        
        # Strip out complex metadata to make the output clean and readable
        metadata_keys = ", ".join([
            f"{k}:{v}" for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool)) and k != 'source'
        ])
        
        logging.debug("-" * 40)
        logging.debug(f"Chunk {i + 1}:")
        logging.debug(f"  Source File: {source_file}")
        if metadata_keys:
            logging.debug(f"  Metadata: {metadata_keys}")
        logging.debug(f"  Content Snippet: {doc.page_content[:200]}...")

@cl.on_message
async def on_message(msg: cl.Message):
    print("The user sent:", msg.content)

    llm = cl.user_session.get("llm")
    if llm is None:
        await cl.Message(content="LLM not initialized. Please wait for setup to complete.").send()
        return
    
    vector_store = cl.user_session.get("vector_store")
    retrieved_docs = vector_store.invoke(msg.content)

    logging.info(f"\n✅ Retrieved {len(retrieved_docs)} relevant document chunks.")
    log_retrieved_docs(retrieved_docs)

    # Further processing with LLM and retrieved_docs can be done here
    llm = cl.user_session.get("llm")

        

