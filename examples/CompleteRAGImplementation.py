"""
PDF Q&A Chatbot with LangChain (v1.x)
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / "keys" / ".env")

_CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parent
PDF_DIRECTORY = str(_BASE_DIR / "data")  # directory containing PDF files to chat with
CHUNK_SIZE = 1000         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
LLM_TEMPERATURE = 0       # LLM temperature (0 = deterministic)
RETRIEVER_K = 3           # number of documents to retrieve per query
PDF_GLOB = "**/*.pdf"     # glob pattern for PDF files inside PDF_DIRECTORY
MMR_FETCH_K_MULTIPLIER = 3  # fetch_k = RETRIEVER_K * MMR_FETCH_K_MULTIPLIER
# ──────────────────────────────────────────────────────────────────────────────


class PDFChatbot:
    """
    Chatbot for answering questions about PDF documents.
    """

    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = _CHROMA_DIR,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
    ):
        """
        Initialize the PDF chatbot.

        Args:
            pdf_directory: Directory containing PDF files
            persist_directory: Where to store vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
        """
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k

        # Initialize components
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=LLM_TEMPERATURE,
        )
        self.vectorstore: Chroma | None = None
        self.chain = None
        self.chat_history: List[BaseMessage] = []

        print("✓ PDF Chatbot initialized")

    def load_pdfs(self) -> List[Document]:
        """Load all PDFs from directory."""
        print(f"\n📂 Loading PDFs from {self.pdf_directory}...")

        loader = DirectoryLoader(
            self.pdf_directory,
            glob=PDF_GLOB,
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )

        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages from PDFs")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        print("\n✂️  Splitting documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")

        if chunks:
            print("\nSample chunk:")
            print(f"Length: {len(chunks[0].page_content)} chars")
            print(f"Preview: {chunks[0].page_content[:150]}...")

        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> None:
        """Create or load vector store."""
        print("\n🔢 Creating vector store...")

        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            doc_count = len(self.vectorstore.get()["ids"])
            if doc_count == 0:
                print("Existing vector store is empty, adding documents...")
                self.vectorstore.add_documents(chunks)
        else:
            print(f"Creating new vector store at {self.persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
            )

        doc_count = len(self.vectorstore.get()["ids"])
        print(f"✓ Vector store ready with {doc_count} embeddings")

    def setup_chain(self) -> None:
        """Setup the conversational RAG chain."""
        print("\n🔗 Setting up conversational chain...")

        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * MMR_FETCH_K_MULTIPLIER},
        )

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
            self.llm, retriever, contextualize_q_prompt
        )

        # Step 2: Answer prompt — uses retrieved context + chat history
        qa_system_prompt = (
            "You are a helpful AI assistant that answers questions about PDF documents.\n\n"
            "Use the following context to answer the user's question. "
            "If you don't know the answer or can't find it in the context, "
            "say \"I cannot find this information in the provided documents.\"\n\n"
            "Instructions:\n"
            "1. Answer based on the provided context\n"
            "2. Be specific and cite relevant information\n"
            "3. Maintain conversation continuity using chat history\n"
            "4. Answer thoughtfully and accurately\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print("✓ Conversational chain ready")

    def initialize(self) -> None:
        """Initialize the complete chatbot."""
        documents = self.load_pdfs()
        if not documents:
            raise ValueError(f"No PDF files found in '{self.pdf_directory}'. Please add PDF files and try again.")
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.setup_chain()
        print("\n✅ Chatbot is ready! Start asking questions.\n")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot.

        Args:
            question: The user's question

        Returns:
            Dict with answer and sources
        """
        if not self.chain:
            raise ValueError("Chatbot not initialized. Call initialize() first.")

        result = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })

        answer = result["answer"]

        # Append turn to history for next iteration
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer),
        ])

        return {
            "answer": answer,
            "sources": [
                {
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:150] + "...",
                }
                for doc in result["context"]
            ],
        }

    def chat(self) -> None:
        """Interactive chat loop."""
        print("=" * 60)
        print("PDF Q&A CHATBOT")
        print("=" * 60)
        print("Ask questions about your PDF documents!")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'reset' to clear conversation history")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("You: ").strip()
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break

            if question.lower() in {"quit", "exit", "q"}:
                print("\n👋 Goodbye!")
                break

            if question.lower() == "reset":
                self.chat_history.clear()
                print("\n🔄 Conversation history cleared!\n")
                continue

            if not question:
                continue

            try:
                result = self.ask(question)

                print(f"\n🤖 Assistant: {result['answer']}")

                if result["sources"]:
                    print("\n📚 Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"   {i}. {source['source']} (Page {source['page']})")

                print()

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


# Main execution
if __name__ == "__main__":
    # Configuration
    PERSIST_DIRECTORY = _CHROMA_DIR

    os.makedirs(PDF_DIRECTORY, exist_ok=True)

    chatbot = PDFChatbot(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        k=RETRIEVER_K,
    )

    chatbot.initialize()
    chatbot.chat()