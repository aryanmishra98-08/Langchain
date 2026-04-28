import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # document to load and index
FILE_ENCODING = "utf-8"    # file encoding
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
BASE_RETRIEVER_K = 2       # number of documents per generated sub-query
LLM_TEMPERATURE = 0        # LLM temperature (0 = deterministic)
DEMO_QUESTION = "What were the main causes of the French Revolution?"  # demo question
_CHROMA_DIR = str(_BASE_DIR / "chroma_db")  # ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────────────

# Setup
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=LLM_TEMPERATURE,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
)
vectorstore = Chroma(
    persist_directory=_CHROMA_DIR,
    embedding_function=embeddings,
)

# Ingest data into the vector store
loader = TextLoader(str(DATA_FILE), encoding=FILE_ENCODING)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
vectorstore.add_documents(chunks)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": BASE_RETRIEVER_K})

# Multi-query retriever generates multiple search queries
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# Single user query generates multiple searches
unique_docs = multi_query_retriever.invoke(DEMO_QUESTION)

print(f"Retrieved {len(unique_docs)} unique documents")