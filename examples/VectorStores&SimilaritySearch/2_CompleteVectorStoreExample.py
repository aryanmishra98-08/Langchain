import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

_CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "chroma_db")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # path to the text file to index
CHUNK_SIZE = 1000    # characters per chunk
CHUNK_OVERLAP = 200  # overlap between consecutive chunks
SEARCH_K = 3         # default number of results to return
DEMO_QUERY = "What were the main causes of the French Revolution?"  # demo search query
# ──────────────────────────────────────────────────────────────────────────────


class DocumentVectorStore:
    """
    Manages document vectorization and retrieval.
    """

    def __init__(self, persist_directory: str = _CHROMA_DIR):
        self.persist_directory = persist_directory
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        )
        self.vectorstore: Chroma | None = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Processing {len(chunks)} chunks...")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

        print(f"✓ Vector store created with {len(chunks)} chunks")
        return self.vectorstore

    def load_existing(self) -> Chroma:
        """Load existing vector store from disk."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        print("✓ Loaded existing vector store")
        return self.vectorstore

    def search(self, query: str, k: int = 3, method: str = "similarity"):
        """Search vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        if method == "similarity":
            return self.vectorstore.similarity_search(query, k=k)
        elif method == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=k*5
            )
        elif method == "similarity_score":
            return self.vectorstore.similarity_search_with_score(query, k=k)

        raise ValueError(
            "Invalid search method. Use 'similarity', 'mmr', or 'similarity_score'."
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} new chunks")


# Usage example
if __name__ == "__main__":
    # Load documents
    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    # Create vector store
    vs = DocumentVectorStore()
    vs.create_from_documents(documents)

    # Search
    results = vs.search(DEMO_QUERY, k=SEARCH_K)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200])