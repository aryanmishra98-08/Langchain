import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
DEMO_QUERY = "What is machine learning?"  # demo query to run
_CHROMA_DIR = str(_BASE_DIR / "chroma_db")  # ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────────────

# Initialize
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

# Define prompt template
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: Provide a detailed answer based on the context. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
"""

prompt = ChatPromptTemplate.from_template(template)


# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Build RAG chain using LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query
response = rag_chain.invoke(DEMO_QUERY)
print(response)