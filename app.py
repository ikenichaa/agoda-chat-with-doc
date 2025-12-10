import asyncio
from importlib.metadata import files
import chainlit as cl
import logging

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_docling import DoclingLoader
from langchain_community.vectorstores import Chroma
from docling.chunking import HybridChunker

logging.basicConfig(level=logging.INFO)


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)


def load_docs_and_chunk(path: str):
    loader = DoclingLoader(path, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))
    return loader.load()


async def process_files_stepwise(files):
    # STEP 1: Parse and chunk PDFs
    step1_msg = await cl.Message(content="⏳ Step 1/2: Parsing and Chunking PDFs…").send()
    
    # Initialize list to hold results *per file* (for source tracking)
    parsed_files_data = [] 
    
    # Initialize list to hold all chunks *combined* (for batch ingestion)
    all_chunks = []
    
    for f in files:
        await cl.Message(content=f"⏳ Parsing and chunking **{f.name}**…").send()
        
        # Load and chunk the file. This should return a list of LangChain Documents.
        # Ensure load_docs_and_chunk handles the DoclingLoader with HybridChunker.
        docs = await cl.make_async(load_docs_and_chunk)(f.path)  # Offload blocking work
        
        # Store results for source tracking
        parsed_files_data.append((f, docs))
        
        # Aggregate all chunks into the master list
        all_chunks.extend(docs)
        
        await cl.Message(content=f"✅ Parsed **{f.name}** → {len(docs)} document chunks").send()
    
    # Final update on the step
    step1_msg.content = f"✅ Step 1/2: Parsing and chunking complete. Total chunks: {len(all_chunks)}"
    await step1_msg.update()

    # Log a few examples from the master list
    logging.info(f"--- Sample of first 3 chunks (Total: {len(all_chunks)}) ---")
    for i, d in enumerate(all_chunks[:3]):
        # Using repr(d.page_content) is cleaner for logging strings
        logging.info(f"Chunk {i+1}: {repr(d.page_content[:50] + '...')}") 
        logging.info(f"Source: {d.metadata.get('source', 'N/A')}")
    
    # At this point, you can use:
    # 1. `all_chunks` for batch ingestion into a single vector store.
    # 2. `parsed_files_data` if you need to access chunks grouped by their original file.

    # STEP 2: Ingest all chunks into a single Chroma collection
    step2_msg = await cl.Message(content="⏳ Step 2/2: Ingesting chunks into vector store…").send()
    chroma_global = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name="index_global"
    )
    
    step2_msg.content = f"✅ Step 2/2: Ingesting chunks into vector store completed."
    await step2_msg.update()
    
    return chroma_global




@cl.on_chat_start
async def start():
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload 1 to 3 PDF files to begin!", accept=["application/pdf"], max_files=3
        ).send()

    await cl.Message(content=f"⏳ Received {len(files)} file(s). Processing…").send()

    # Kick off processing
    processing_task = asyncio.create_task(process_files_stepwise(files))


    # Wait for processing to finish first, then for the user to click
    chroma_global = await processing_task

    # Store the retrievers in the Chainlit user session
    cl.user_session.set("retriever_global", chroma_global.as_retriever())
    
    # Initialize your LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    cl.user_session.set("llm", llm)
    
    await cl.Message(content="Setup complete! Ready to chat.").send()



@cl.on_message
async def on_message(msg: cl.Message):
    print("The user sent:", msg.content)

    llm = cl.user_session.get("llm")
    if llm is None:
        await cl.Message(content="LLM not initialized. Please wait for setup to complete.").send()
        return
    
    # 1. Load the necessary components from the session
    retriever_global = cl.user_session.get("retriever_global")
    
    # 2. **RETRIEVAL STEP:** Fetch context using the user's query
    # In a simple RAG, we use the global retriever
    retrieved_docs = retriever_global.invoke(msg.content)
    
    # 3. **RAG LOGIC:** Format the prompt (this is often done with a LangChain chain)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_template = (
        "You are an expert assistant. Use the following context to answer the user's question. "
        "If you don't know the answer, state that you cannot find it in the provided documents.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {msg.content}"
    )

    # 4. **GENERATION STEP:** Invoke the LLM with the augmented prompt
    response = llm.invoke(prompt_template)
    
    # 5. Send the response
    await cl.Message(content=response.content).send()
