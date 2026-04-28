import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # path to the text file to index
CHROMA_DIR = _BASE_DIR / "chroma_db"   # directory for the persistent ChromaDB store
FILE_ENCODING = "utf-8"                # file encoding
CHUNK_SIZE = 1000                      # characters per chunk
CHUNK_OVERLAP = 200                    # overlap between consecutive chunks
COLLECTION_NAME = "the_french_revolution"  # ChromaDB collection name

# Demo queries
SIMILARITY_QUERY = "What caused the French Revolution?"          # similarity search query
SIMILARITY_K = 3                                                  # results to return
SCORES_QUERY = "What happened during the Reign of Terror?"       # scored search query
SCORES_K = 3                                                      # results to return
MMR_K = 4           # final results for MMR search
MMR_FETCH_K = 20    # candidate pool size for MMR
MMR_LAMBDA_MULT = 0.5  # 0 = max diversity, 1 = max relevance
# ──────────────────────────────────────────────────────────────────────────────

# Load a single text file
loader = TextLoader(str(DATA_FILE), encoding=FILE_ENCODING)
documents = loader.load()

# Split into retrieval-friendly chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
)

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),  # Persist to disk
    collection_name=COLLECTION_NAME,
)

doc_count = len(vectorstore.get()["ids"])
print(f"✓ Created vector store with {doc_count} documents")

# Similarity Search
results = vectorstore.similarity_search(
    SIMILARITY_QUERY,
    k=SIMILARITY_K,
)

print(f"Found {len(results)} relevant documents:\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")

# Similarity Search with Scores
results_with_scores = vectorstore.similarity_search_with_score(
    SCORES_QUERY,
    k=SCORES_K,
)

print("Results with similarity scores:\n")
for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Maximum Marginal Relevance (MMR)
results_mmr = vectorstore.max_marginal_relevance_search(
    SCORES_QUERY,
    k=MMR_K,
    fetch_k=MMR_FETCH_K,
    lambda_mult=MMR_LAMBDA_MULT,
)

print(f"MMR returned {len(results_mmr)} diverse results")