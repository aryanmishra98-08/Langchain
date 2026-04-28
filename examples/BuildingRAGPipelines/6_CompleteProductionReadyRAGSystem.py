import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
_BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = _BASE_DIR / "data" / "TheFrenchRevolution.txt"  # document to load and index
FILE_ENCODING = "utf-8"         # file encoding
CHUNK_SIZE = 1000               # characters per chunk
CHUNK_OVERLAP = 200             # overlap between consecutive chunks
DEFAULT_TEMPERATURE = 0.0      # LLM temperature (0 = deterministic)
DEFAULT_K = 4                   # number of documents to retrieve
DEFAULT_STREAMING = True        # enable streaming output
SEARCH_TYPE = "mmr"             # retriever search type: "similarity" or "mmr"
FETCH_K_MULTIPLIER = 3          # fetch_k = DEFAULT_K * FETCH_K_MULTIPLIER for MMR
DEMO_K = 3                      # k to use in the __main__ demo
DEMO_QUESTION = "What is the main topic discussed?"  # demo question to run
_CHROMA_DIR = str(_BASE_DIR / "chroma_db")           # ChromaDB persistence directory
# ──────────────────────────────────────────────────────────────────────────────


class ProductionRAGSystem:
    """
    Production-ready RAG system with best practices.
    """

    def __init__(
        self,
        persist_directory: str = _CHROMA_DIR,
        temperature: float = DEFAULT_TEMPERATURE,
        k: int = DEFAULT_K,
        streaming: bool = DEFAULT_STREAMING,
    ):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=temperature,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        # Ingest data into the vector store
        loader = TextLoader(str(DATA_FILE), encoding=FILE_ENCODING)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)

        self.retriever = self.vectorstore.as_retriever(
            search_type=SEARCH_TYPE,
            search_kwargs={"k": k, "fetch_k": k * FETCH_K_MULTIPLIER},
        )
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build the RAG chain that returns both answer and source docs."""
        template = """You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific parts of the context when possible
4. Be concise but thorough

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(
                f"[Source {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )

        # Answer sub-chain: takes {context, question}, returns string
        answer_chain = prompt | self.llm | StrOutputParser()

        # Full chain: retrieve once, run answer + return docs in parallel
        chain = RunnableParallel(
            {
                "source_documents": self.retriever,
                "question": RunnablePassthrough(),
            }
        ).assign(
            answer=lambda x: answer_chain.invoke(
                {"context": format_docs(x["source_documents"]), "question": x["question"]}
            )
        )

        return chain

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        result = self.chain.invoke(question)

        return {
            "answer": result["answer"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in result["source_documents"]
            ],
            "num_sources": len(result["source_documents"]),
        }

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions."""
        return [self.query(q) for q in questions]


# Usage
if __name__ == "__main__":
    rag = ProductionRAGSystem(
        persist_directory=_CHROMA_DIR,
        k=DEMO_K,
    )

    result = rag.query(DEMO_QUESTION)

    print("\n\nAnswer:", result["answer"])
    print(f"\nUsed {result['num_sources']} sources")
    print("\nSources:")
    for i, source in enumerate(result["source_documents"]):
        print(f"{i+1}. {source['metadata']}")