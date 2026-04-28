import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # document to load and index
FILE_ENCODING = "utf-8"   # file encoding
CHUNK_SIZE = 1000         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
RETRIEVER_K = 3           # number of documents to retrieve
LLM_TEMPERATURE = 0       # LLM temperature (0 = deterministic)
DEMO_QUERY = "Explain the key concepts"  # demo query to run
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

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

# Custom prompt with citation instructions
template = """Answer the question based on the following context. 
After your answer, list the sources you used with their page numbers.

Context:
{context}

Question: {question}

Answer format:
[Your detailed answer]

Sources:
- [Source 1 with page number]
- [Source 2 with page number]
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs_with_sources(docs):
    """Format documents with source information"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


# Build chain
rag_chain_with_sources = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query
response = rag_chain_with_sources.invoke(DEMO_QUERY)
print(response)