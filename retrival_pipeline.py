# retrieval_pipeline_offline.py

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

load_dotenv()

# ===============================
# 1️⃣ Load vector store (OFFLINE)
# ===============================
persistent_directory = "db/chroma_db"

# Use offline embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # small, fast embeddings
)

# Load Chroma vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model
)

# ===============================
# 2️⃣ Create retriever
# ===============================
retriever = db.as_retriever(search_kwargs={"k": 3})

# ===============================
# 3️⃣ Queries
# ===============================
queries = [
    "Who founded Google?",
    "What products does Google offer?",
]

# ===============================
# 4️⃣ Offline text generation model
# ===============================
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",  # works fully offline
    device=-1  # CPU, set to 0 for GPU
)

# ===============================
# 5️⃣ Retrieve documents and generate answers
# ===============================
for query in queries:
    # Retrieve top-k relevant documents
    relevant_docs = retriever.invoke(query)

    print(f"\nUser Query: {query}")
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i + 1} (Source: {doc.metadata.get('source', 'N/A')}):")
        print(doc.page_content[:500])  # show first 500 characters
        print("-" * 80)

    # Combine documents into a single input for the model
    combined_input = f"""
Based on the following documents, answer this question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Provide a clear answer using ONLY the information from these documents.
If the answer is not found, say: "I don't have enough information to answer the question."
"""

    # Generate offline answer
    response = generator(
        combined_input,
        max_length=300,
        do_sample=False
    )[0]['generated_text']

    print("\n--- Generated Response ---")
    print(response)
    print("="*100)
