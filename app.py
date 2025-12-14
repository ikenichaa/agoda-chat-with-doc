import logging
import sys

# Configure logging BEFORE any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Force reconfiguration even if already configured
)

import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL_NAME, RETRIEVAL_TOP_K
from error_handler import ErrorHandler
from indexing import ingest_to_vectorstore, parse_and_chunk_files
from retrieval import format_sources_for_display, get_rag_response


@cl.on_chat_start
async def start():
    """Initialize chat session: upload files, process documents, and setup RAG system."""
    
    # Request file upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload 1 to 3 documents to begin!\n\n**Supported formats:** PDF, Word (.docx)\n\n**Noted:** Running for the first time may take a while",
            accept=[
                "application/pdf",                                                      # PDF
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
            ],
            max_files=3
        ).send()

    logging.info(f"Received {len(files)} file(s) for processing")
    
    processing_msg = await cl.Message(
        content=f"⏳ Received {len(files)} file(s). Processing…"
    ).send()

    try:
        # Process files
        all_chunks = await parse_and_chunk_files(files)
        vector_store = await ingest_to_vectorstore(all_chunks)

        processing_msg.content = "✅ All processing complete!"
        await processing_msg.update()

        # Initialize and store session components
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
        
        cl.user_session.set("vector_store", retriever)
        cl.user_session.set("llm", llm)
        
        logging.info("Chat session initialized successfully")
        await cl.Message(content="Setup complete! Ready to chat.").send()
        
    except (ValueError, ConnectionError, Exception) as e:
        await ErrorHandler.handle_error(e, "setup", show_details=isinstance(e, ValueError))
        processing_msg.content = ErrorHandler.get_error_message(e)
        await processing_msg.update()


@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages, execute RAG pipeline, and send response."""
    
    # Validate session components
    llm = cl.user_session.get("llm")
    retriever = cl.user_session.get("vector_store")
    
    if llm is None or retriever is None:
        logging.warning("User attempted to send message before initialization")
        await cl.Message(content="⚠️ System not initialized. Please wait for setup to complete.").send()
        return
    
    if not msg.content or not msg.content.strip():
        await cl.Message(content="⚠️ Please provide a question.").send()
        return
    
    logging.info(f"Processing user message: {msg.content[:100]}...")
    await cl.Message(content="🤔 Thinking...").send()
    
    try:
        # Execute RAG pipeline
        response = await get_rag_response(msg.content, retriever, llm)
        
        # Send answer
        await cl.Message(content=f"**Answer**:\n{response['answer']}").send()
        
        # Send sources if available
        sources_text = format_sources_for_display(response['sources_cited'])
        if sources_text:
            await cl.Message(content=sources_text).send()
        
        logging.info("Response sent successfully")
    
    except (ValueError, ConnectionError, Exception) as e:
        await ErrorHandler.handle_error(e, "message processing", show_details=isinstance(e, ValueError))


        

