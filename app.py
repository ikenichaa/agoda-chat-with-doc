import asyncio
from importlib.metadata import files
import chainlit as cl
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

logging.basicConfig(level=logging.INFO)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")



def load_docs_sync(path: str):
    loader = DoclingLoader(path)
    return loader.load()



async def process_files_stepwise(files):
    # Example: three steps with per-file work & progress messages
    # STEP 1: Parse PDFs
    step1_msg = await cl.Message(content="⏳ Step 1/3: Parsing PDFs…").send()
    parsed = []
    for f in files:
        await cl.Message(content=f"⏳ Parsing **{f.name}**…").send()
        docs = await cl.make_async(load_docs_sync)(f.path)  # offload blocking work
        parsed.append((f, docs))
        # await cl.Message(content=f"✅ Parsed **{f.name}**").send()
        await cl.Message(content=f"✅ Parsed **{f.name}** → {len(docs)} document(s)").send()
    
    
    step1_msg.content = "✅ Step 1/3: Parsing complete."
    await step1_msg.update()


    # STEP 2: Chunking (illustrative)
    # step2_msg = await cl.Message(content="✂️ Step 2/3: Chunking documents…").send()
    # chunked = []
    # for f, docs in parsed:
    #     # … your chunking logic …
    #     chunked.append((f, docs))
    #     await cl.Message(content=f"✅ Chunked **{f.name}**").send()
    # await step2_msg.update(content="✅ Step 2/3: Chunking complete.")

    # STEP 3: Embeddings / Index
    # step3_msg = await cl.Message(content="🧠 Step 3/3: Creating embeddings & index…").send()
    # # … create embeddings/vector store asynchronously …
    # await cl.Message(content="✅ Index ready. You can start asking questions!").send()
    # await step3_msg.update(content="✅ Step 3/3 complete.")




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
    await processing_task
    await cl.Message(content="✅ Processing complete.").send()

    await cl.Message(content="You can now start chatting with your documents!").send()



@cl.on_message
async def on_message(msg: cl.Message):
    print("The user sent:", msg.content)

    response = llm.invoke(msg.content)
    await cl.Message(content=f"You said: {msg.content}\nLLM responded: {response}").send()
