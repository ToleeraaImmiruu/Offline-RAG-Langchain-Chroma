import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# Load documents
# --------------------------------------------------
def load_documents(docs_path="docs"):
    print(f"Loading documents from '{docs_path}'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError("No .txt files found")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)}")
        print(f"Preview: {doc.page_content[:100]}...")

    return documents


# --------------------------------------------------
# Split documents
# --------------------------------------------------
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    print("Splitting documents into chunks...")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n"
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


# --------------------------------------------------
# Create Chroma vector store (LOCAL EMBEDDINGS)
# --------------------------------------------------
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating vector store (LOCAL embeddings)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Vector store created successfully ✅")
    return vectorstore

# Main

def main():
    print("Starting ingestion pipeline (LOCAL embeddings)...\n")

    documents = load_documents("docs")
    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("\nIngestion finished successfully 🚀")


if __name__ == "__main__":
    main()

