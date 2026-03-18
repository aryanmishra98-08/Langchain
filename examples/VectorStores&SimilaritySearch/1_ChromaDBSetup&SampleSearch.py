from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

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
    collection_name="my_documents"
)

print(
    f"✓ Created vector store with {vectorstore._collection.count()} documents")

# Similarity Search
# Basic similarity search
query = "What is machine learning?"
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
query = "Explain neural networks"
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
