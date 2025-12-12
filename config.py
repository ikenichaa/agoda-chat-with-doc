# config.py
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Model Settings
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite")

# Vector Store Settings
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "docling_rag_index"
MILVUS_INDEX_PARAMS = {"index_type": "IVF_FLAT", "metric_type": "L2"}

# Retrieval Settings
RETRIEVAL_TOP_K = 5

# Prompts
RAG_PROMPT = ChatPromptTemplate.from_messages([
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
])