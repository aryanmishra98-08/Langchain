from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentVectorStore:
    """
    Manages document vectorization and retrieval
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_from_documents(self, documents: list):
        """
        Create vector store from documents
        """
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Processing {len(chunks)} chunks...")

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"✓ Vector store created with {len(chunks)} chunks")
        return self.vectorstore

    def load_existing(self):
        """
        Load existing vector store from disk
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"✓ Loaded existing vector store")
        return self.vectorstore

    def search(self, query: str, k: int = 3, method: str = "similarity"):
        """
        Search vector store
        """
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

    def add_documents(self, documents: list):
        """
        Add new documents to existing vector store
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)
        print(f"✓ Added {len(chunks)} new chunks")


# Usage example
if __name__ == "__main__":
    # Load documents
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "TheFrenchRevolution.txt"
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    # Create vector store
    vs = DocumentVectorStore()
    vs.create_from_documents(documents)

    # Search
    query = "What were the main causes of the French Revolution?"
    results = vs.search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200])
