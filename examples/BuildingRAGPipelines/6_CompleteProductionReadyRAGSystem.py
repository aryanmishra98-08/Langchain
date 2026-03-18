from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Dict
import os

class ProductionRAGSystem:
    """
    Production-ready RAG system with best practices
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model: str = "gpt-4",
        temperature: float = 0,
        k: int = 4
    ):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Use MMR for diversity
            search_kwargs={"k": k, "fetch_k": k * 3}
        )
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain"""
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
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str) -> Dict[str, any]:
        """
        Query the RAG system
        """
        # Get relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        # Generate answer
        answer = self.chain.invoke(question)
        
        # Prepare response
        response = {
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ],
            "num_sources": len(relevant_docs)
        }
        
        return response
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        Process multiple questions
        """
        return [self.query(q) for q in questions]

# Usage
if __name__ == "__main__":
    # Initialize system
    rag = ProductionRAGSystem(
        persist_directory="./chroma_db",
        model="gpt-4",
        k=3
    )
    
    # Query
    result = rag.query("What is the main topic discussed?")
    
    print("\n\nAnswer:", result["answer"])
    print(f"\nUsed {result['num_sources']} sources")
    print("\nSources:")
    for i, source in enumerate(result["source_documents"]):
        print(f"{i+1}. {source['metadata']}")