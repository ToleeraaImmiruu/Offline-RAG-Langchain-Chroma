import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from transformers import pipeline

# ===============================
# Load env (not required but OK)
# ===============================
load_dotenv()

# ===============================
# Load Chroma Vector Store
# ===============================
PERSIST_DIR = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ===============================
# Load OFFLINE LLM (FLAN-T5)
# ===============================
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

# ===============================
# Chat History
# ===============================
chat_history = []

# ===============================
# Helper: run LLM
# ===============================
def run_llm(prompt: str) -> str:
    return llm(prompt)[0]["generated_text"].strip()

# ===============================
# Ask Question (History Aware)
# ===============================
def ask_question(user_question: str):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1️⃣ Rewrite question if history exists
    if chat_history:
        history_text = "\n".join(
            [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
            for m in chat_history]
        )

        rewrite_prompt = f"""
Given the following conversation history and a follow-up question,
rewrite the question so it is a standalone question.

Conversation history:
{history_text}

Follow-up question:
{user_question}

Standalone question:
"""
        search_question = run_llm(rewrite_prompt)
        print(f"🔍 Rewritten question: {search_question}")
    else:
        search_question = user_question

    # Step 2️⃣ Retrieve documents
    docs = retriever.invoke(search_question)

    print(f"📄 Found {len(docs)} relevant documents")

    # Step 3️⃣ Build RAG prompt
    context = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the information in the documents.

Documents:
{context}

Question:
{user_question}

If the answer is not found, say:
"I don't have enough information to answer the question."
"""

    answer = run_llm(rag_prompt)

    # Step 4️⃣ Save to history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    # Step 5️⃣ Print answer
    print("\n--- Answer ---")
    print(answer)

# ===============================
# Chat Loop
# ===============================
def start_chat():
    print("🤖 Ask me questions! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            print("👋 Goodbye!")
            break
        ask_question(question)


if __name__ == "__main__":
    start_chat()
