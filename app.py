import asyncio
import chainlit as cl
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

logging.basicConfig(level=logging.INFO)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


async def process_file_background(f):
    # This runs in the background after on_chat_start returns
    try:
        loader = DoclingLoader(f.path)
        async_load = cl.make_async(loader.load)
        docs = await async_load()
        logging.info(f"Docs({f.name}): {len(docs)} documents")
        await cl.Message(
            content=f"✅ Finished processing **{f.name}**.\n• Extracted documents: **{len(docs)}**"
        ).send()
    except Exception as e:
        logging.exception("Error processing file in background")
        await cl.Message(
            content=f"❌ Failed processing **{f.name}**: `{e}`"
        ).send()


@cl.on_chat_start
async def start():
    files = None
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload 1 to 3 PDF files to begin!", accept=["application/pdf"], max_files=3
        ).send()

    
    logging.info(f"Processing {len(files)} uploaded files.")
    for f in files:
        logging.info(f"Queued file: {f.name}")
        await cl.Message(content=f"✅ Received file: {f.name}\n⏳ Processing in background…").send()
        # Schedule without awaiting (fire-and-forget)
        asyncio.create_task(process_file_background(f))

    # Optionally send a final note right away
    await cl.Message("👍 You can continue chatting—I'll notify you as each file finishes.").send()


    #     # Step 2: Chunk with HybridChunker
    #     chunker = HybridChunker()
    #     chunks = chunker.split_documents(docs)

    #     # Step 3: Display chunks and metadata
    #     for i, chunk in enumerate(chunks[:5]):  # show first 5 chunks for brevity
    #         await cl.Message(
    #             content=f"**Chunk {i+1}:**\n{chunk.page_content}\n\n**Metadata:** {chunk.metadata}"
    #         ).send()

    # await cl.Message("✨ All files processed and chunked!").send()


@cl.on_message
async def on_message(msg: cl.Message):
    print("The user sent:", msg.content)

    response = llm.invoke(msg.content)
    await cl.Message(content=f"You said: {msg.content}\nLLM responded: {response}").send()
