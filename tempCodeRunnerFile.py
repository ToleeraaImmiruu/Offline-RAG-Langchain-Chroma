import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_chroma import Chroma
from langchain.embeddings import GoogleGeminiEmbeddings

from dotenv import load_dotenv

# Gemini import (used later)
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # LOADER OF THE FLE FROM THE ANOTHER ONE 
def load_documents(docs_path="docs"):
    """Load all .txt documents from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. "
            "Please create it and add your company files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. Please add your documents."
        )

    # Preview first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


# CHUCKING THE LOADED ONE SPLITTING THEM IN TO MANY CHUCKS 
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split the documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\nChunk {i + 1}")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content preview:")
            print(chunk.page_content[:200])
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

# CREATING THE VECTOR FOR THE CHUNK 


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using Gemini embeddings"""
    print("Creating embeddings and storing in ChromaDB...")

    # Initialize Gemini embeddings
    embedding_model = GoogleGeminiEmbeddings(
        model="embedding-gecko-001",
        api_key=GEMINI_API_KEY
    )

    # Create the Chroma vector store
    print("-- Creating vector store --")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore



def main():
    print("Main Function")

    documents = load_documents("docs")
    print(f"\nTotal documents loaded: {len(documents)}")
    chunks= split_documents(documents)
    vectorstore=create_vector_store(chunks)


if __name__ == "__main__":
    main()
