import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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
SEARCH_TYPE = "similarity"  # retriever search type: "similarity" or "mmr"
RETRIEVER_K = 3           # number of documents to retrieve
LLM_TEMPERATURE = 0       # LLM temperature (0 = deterministic)
DEMO_QUERY = "What is the main topic of the document?"  # demo query to run
_CHROMA_DIR = str(_BASE_DIR / "chroma_db")              # ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────────────

# Initialize components
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=LLM_TEMPERATURE,
)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
)

# Load vector store
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

# Create retriever
retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs={"k": RETRIEVER_K},
)

# Define prompt
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Build RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Query
result = rag_chain.invoke({"input": DEMO_QUERY})

print("Answer:", result["answer"])
print("\nSources:")
for doc in result["context"]:
    print(f"- {doc.metadata.get('source', 'Unknown')}")