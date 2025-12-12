import logging

from pydantic import BaseModel, Field
from typing import List, Dict, Any

import chainlit as cl
from docling.chunking import HybridChunker

from langchain_docling import DoclingLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus

logging.basicConfig(level=logging.DEBUG)


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MILVUS_URI = "http://localhost:19530" 
COLLECTION_NAME = "docling_rag_index"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

class SourceDocument(BaseModel):
    """A document fragment used as a source for the answer."""
    file_name: str = Field(description="The source file name, e.g., 'policy.pdf'.")
    chunk_content: str = Field(description="The direct text excerpt taken from the file that supports the answer.")

class StructuredAnswer(BaseModel):
    """The final structured response from the LLM."""
    answer: str = Field(description="The complete, conversational answer to the user's question, strictly based on the context. Use the full 'I cannot answer...' phrase for failure cases.")
    sources_cited: List[SourceDocument] = Field(description="A list of ALL source documents used to formulate the answer. This list MUST be empty if the answer field indicates a failure (e.g., 'I cannot answer...').")


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
        
        # Update metadata to use the original filename instead of temp path
        for doc in docs:
            doc.metadata['source'] = f.name
            doc.metadata['original_path'] = f.path  # Keep temp path if needed for debugging
        
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

# Assuming 'retrieved_docs' is the list of documents from your retriever
# e.g., retrieved_docs = [Document(page_content='...', metadata={'source': 'doc1.pdf'}), ...]

def format_docs_with_sources(docs):
    formatted_context = []
    for i, d in enumerate(docs):
        # Create a unique label for the source (e.g., [Source 1], [Source 2])
        source_label = f"[{i+1}]"
        
        # Get the file name from metadata
        # Common key is 'source', 'file_name', or 'id' depending on your loader
        file_name = d.metadata.get('source', 'Unknown Source')
        
        # Combine the source info and the content
        formatted_context.append(f"{source_label} (File: {file_name}):\n{d.page_content}")
    
    # Join all formatted chunks with a clear separator
    return "\n\n---\n\n".join(formatted_context)


def prompt_with_context(user_query, context_string):
    prompt_template = """
        SYSTEM INSTRUCTION:
        You are an intelligent, helpful, and strictly factual question-answering assistant.
        Your sole task is to answer the user's question ONLY based on the provided CONTEXT.

        RULES:
        1.  **Strictly Factual:** You must use the CONTEXT below to form your answer. Only use external knowledge if you are certain it is accurate and relevant.
        2.  **Citation Required:** For every piece of information you provide, you **MUST** include the corresponding source label (e.g., [Source 1]) at the end of the sentence or paragraph where the information was drawn from.
        3.  **Unrelated/Insufficient Context:**
            a. If the question is clearly **unrelated** to the provided CONTEXT, state: "The question is unrelated to the provided documents."
            b. If the question is related, but the CONTEXT **does not contain enough information** to give a complete or accurate answer, state: "I cannot answer this question based solely on the provided documents."
        4.  **Final Summary:** After your main answer, provide a final section titled "Sources Cited" listing the full file name and source excerpt for every source label you used.

        CONTEXT:
        ---
        {context} <-- This now includes the File Name information
        ---

        --- FEW-SHOT EXAMPLES ---
        # Example 1: Successful Answer (Uses multiple sources)
        USER QUESTION:
        What is the procedure for requesting travel reimbursement and how much is the daily meal allowance?
        ---
        ASSISTANT RESPONSE:
        The procedure for requesting travel reimbursement involves submitting a completed T&E form to the finance department within 10 days of the trip's completion [Source 1]. The daily meal allowance for domestic travel is set at $50 per day [Source 2].

        Sources Cited:
        File name: Policy_Travel.pdf
        Source 1: The procedure for travel reimbursement requires submission of a T&E form within 10 days.
        Source 2: The maximum daily meal allowance for all domestic travel is $50.

        File name: Rates_2024.csv
        Source 3: The daily meal allowance for domestic travel is $50.

        # Example 2: Insufficient Context (Related question, but missing details)
        USER QUESTION:
        Who is the current head of the marketing department?
        ---
        ASSISTANT RESPONSE:
        I cannot answer this question based solely on the provided documents.

        # Example 3: Unrelated Question (Question is outside the scope of the documents)
        USER QUESTION:
        What is the capital of France?
        ---
        ASSISTANT RESPONSE:
        The question is unrelated to the provided documents.

        --- END OF EXAMPLES ---

        USER QUESTION:
        {question}
        ---
        
        ASSISTANT RESPONSE:
    """

    return prompt_template.format(
        context=context_string,
        question=user_query
    )

# 2. Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strictly factual question-answering assistant. Your sole task is to generate a JSON object to answer the user's question ONLY based on the provided CONTEXT.\n\n"
                "RULES:\n"
                "1. Answer the question ONLY using the CONTEXT. Use external knowledge only to accommodate the answer such as giving recommendation.\n"
                "2. If the question is clearly UNRELATED to the context, set the 'answer' field to: 'The question is unrelated to the provided documents.' and set 'sources_cited' to an empty list.\n"
                "3. If the question is RELATED but there is no CONTEXT to answer, set the 'answer' field to: 'I cannot answer this question based solely on the provided documents.' and set 'sources_cited' to an empty list.\n"
                "4. For every claim in the answer, include the supporting file name and excerpt in the 'sources_cited' list.\n\n"
                "CONTEXT:\n---\n{context}\n---\n"
            ),
        ),
        ("human", "USER QUESTION: {question}"),
    ]
)

@cl.on_message
async def on_message(msg: cl.Message):
    print("The user sent:", msg.content)

    llm = cl.user_session.get("llm")
    if llm is None:
        await cl.Message(content="LLM not initialized. Please wait for setup to complete.").send()
        return
    
    vector_store = cl.user_session.get("vector_store")
    retrieved_docs = vector_store.invoke(msg.content)

    context_string = format_docs_with_sources(retrieved_docs)

    logging.info(f"\n✅ Retrieved {len(retrieved_docs)} relevant document chunks.")
    logging.info(f"--- Formatted Context String ---\n{context_string}\n")
    # log_retrieved_docs(retrieved_docs)

    # # Further processing with LLM and retrieved_docs can be done here
    # llm = cl.user_session.get("llm")

    # prompt = prompt_with_context(msg.content, context_string)
    # # Create an empty message object to stream into
    # msg = cl.Message(content="")

    # # Stream the chunks from the LLM
    # async for chunk in llm.astream(prompt):
    #     # Append the chunk content to the message object and update the UI
    #     await msg.stream_token(chunk.content) 

    # # Once the stream is complete, finalize the message in the UI
    # await msg.send()

    structured_llm = llm.with_structured_output(StructuredAnswer)
    # 2. Define the LCEL Chain
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: context_string)
        | RAG_PROMPT
        | structured_llm # Returns a StructuredAnswer Pydantic object
    )

    # 3. Invoke the chain to get the structured output (this is where the delay occurs)
    # Since we are not using .astream, we get the validated object directly.
    # We use cl.defer to update the UI while waiting for the LLM.
    await cl.Message(content="Thinking...").send()
    
    try:
        structured_response: StructuredAnswer = await rag_chain.ainvoke(
            {"question": msg.content}
        )
    except Exception as e:
        await cl.Message(content=f"An error occurred during structured output: {e}").send()
        return

    # 4. Format the Pydantic object into the final streaming text
    
    # Separate the message into the answer (streamed) and sources (element)
    answer = structured_response.answer
    sources_cited = structured_response.sources_cited
    logging.info(f"\n✅ Structured Response Received:\ {structured_response}")

    # Format the Answer
    await cl.Message(content=f"**Answer:**\n{answer}").send()
    
    # 5. Handle Sources for the Side Panel (Traceability)
    if sources_cited:
        # A. Aggregate chunks by file name
        file_excerpts: Dict[str, List[str]] = {}
        for source in sources_cited:
            if source.file_name not in file_excerpts:
                file_excerpts[source.file_name] = []
            file_excerpts[source.file_name].append(source.chunk_content)

        # B. Build the final sources markdown string
        sources_text = "### Sources Cited\n\n"
        
        for file_name, contents in file_excerpts.items():
            sources_text += f"**File:** `{file_name}`\n"
            for i, content in enumerate(contents):
                # Using a markdown blockquote (>) is excellent for excerpts
                sources_text += f"> **Excerpt {i+1}:** {content.strip()}\n"
            sources_text += "\n"

        # 3. SEND MESSAGE 2: The Sources
        # We send a second message containing ONLY the source markdown.
        # Note: You could use a different author name (e.g., "Source Bot") here if desired.
        await cl.Message(
            content=sources_text,
            author="Sources" # Optional: gives the source message a distinct author name
        ).send()


        

