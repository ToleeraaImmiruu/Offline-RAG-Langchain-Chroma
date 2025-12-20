# retrieval_pipeline_offline.py

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


# 1️⃣ Load vector store

persistent_directory = "db/chroma_db"

# Use offline embedding model (MiniLM)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load Chroma vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# 2️⃣ Create retriever

retriever = db.as_retriever(search_kwargs={"k": 3})


# 3️⃣ Query examples

queries = [
    "Who founded Google?",
    "What products does Tesla make?",
    "What are the main products of Apple?",
]


# 4️⃣ Retrieve and display results

for query in queries:
    relevant_docs = retriever.invoke(query)
    print(f"\nUser Query: {query}")
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs):
        print(
            f"Document {i + 1} (Source: {doc.metadata.get('source', 'N/A')}):")
        print(doc.page_content[:500])  # preview first 500 chars
        print("-" * 80)
