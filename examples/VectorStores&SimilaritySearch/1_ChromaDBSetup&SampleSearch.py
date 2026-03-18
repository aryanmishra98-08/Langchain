from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


# Load a single text file
base_dir = Path(__file__).resolve().parents[1]
file_path = base_dir / "data" / "TheFrenchRevolution.txt"
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

# Split into retrieval-friendly chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # Persist to disk
    collection_name="the_french_revolution"
)

print(
    f"✓ Created vector store with {vectorstore._collection.count()} documents")

# Similarity Search
# Basic similarity search
query = "What caused the French Revolution?"
results = vectorstore.similarity_search(
    query,
    k=3  # Return top 3 most similar documents
)

print(f"Found {len(results)} relevant documents:\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}\n")

# Similarity Search with Scores
# Get similarity scores
query = "What happened during the Reign of Terror?"
results_with_scores = vectorstore.similarity_search_with_score(
    query,
    k=3
)

print("Results with similarity scores:\n")
for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Maximum Marginal Relevance (MMR)
# MMR: Balances relevance with diversity
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=4,
    fetch_k=20,  # Fetch more candidates
    lambda_mult=0.5  # 0=max diversity, 1=max relevance
)

print(f"MMR returned {len(results_mmr)} diverse results")
