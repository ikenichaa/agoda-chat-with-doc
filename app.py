import logging

import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL_NAME, RETRIEVAL_TOP_K
from indexing import ingest_to_vectorstore, parse_and_chunk_files
from retrieval import format_sources_for_display, get_rag_response

logging.basicConfig(level=logging.DEBUG)


@cl.on_chat_start
async def start():
    """Initialize chat session: upload files, process documents, and setup RAG system."""
    
    # Request file upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload 1 to 3 PDF files to begin!",
            accept=["application/pdf"],
            max_files=3
        ).send()

    # Process files
    processing_msg = await cl.Message(
        content=f"⏳ Received {len(files)} file(s). Processing…"
    ).send()

    all_chunks = await parse_and_chunk_files(files)
    vector_store = await ingest_to_vectorstore(all_chunks)

    processing_msg.content = "✅ All processing complete!"
    await processing_msg.update()

    # Initialize and store session components
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
    
    cl.user_session.set("vector_store", retriever)
    cl.user_session.set("llm", llm)
    
    await cl.Message(content="Setup complete! Ready to chat.").send()


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages, execute RAG pipeline, and send response."""
    
    # Validate session components
    llm = cl.user_session.get("llm")
    retriever = cl.user_session.get("vector_store")
    
    if llm is None or retriever is None:
        await cl.Message(content="System not initialized. Please wait for setup to complete.").send()
        return
    
    await cl.Message(content="Thinking...").send()
    
    try:
        # Execute RAG pipeline
        response = await get_rag_response(msg.content, retriever, llm)
        
        # Send answer
        await cl.Message(content=f"**Answer**:\n{response['answer']}").send()
        
        # Send sources if available
        sources_text = format_sources_for_display(response['sources_cited'])
        if sources_text:
            await cl.Message(content=sources_text).send()
    
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()


        

