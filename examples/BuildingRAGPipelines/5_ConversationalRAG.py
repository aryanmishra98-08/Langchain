import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # document to load and index
FILE_ENCODING = "utf-8"   # file encoding
CHUNK_SIZE = 1000         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
LLM_TEMPERATURE = 0       # LLM temperature (0 = deterministic)
DEMO_QUERIES = [          # multi-turn conversation to demo (edit freely)
    "What were the main causes of the French Revolution?",
    "What were the major events during the Revolution?",   # uses prior context
    "Can you explain more about the first event?",         # references prior answer
]
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

retriever = vectorstore.as_retriever()

# Step 1: Contextualize question — rewrite follow-ups using chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed; otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step 2: Answer prompt — uses retrieved context + chat history
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise.\n\n"
    "Context: {context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine into full RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manual chat history (replaces ConversationBufferMemory)
chat_history = []

# Multi-turn conversation
for query in DEMO_QUERIES:
    print(f"\nUser: {query}")
    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history,
    })
    answer = result["answer"]
    print(f"Assistant: {answer}")

    # Append turn to history for next iteration
    chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer),
    ])