"""
PDF Q&A Chatbot with LangChain
Author: Your Name
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


class PDFChatbot:
    """
    Chatbot for answering questions about PDF documents
    """
    
    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = "./pdf_chatbot_db",
        model: str = "gpt-4",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4
    ):
        """
        Initialize the PDF chatbot
        
        Args:
            pdf_directory: Directory containing PDF files
            persist_directory: Where to store vector database
            model: OpenAI model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
        """
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.vectorstore = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = None
        
        print("✓ PDF Chatbot initialized")
    
    def load_pdfs(self) -> List:
        """Load all PDFs from directory"""
        print(f"\n📂 Loading PDFs from {self.pdf_directory}...")
        
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages from PDFs")
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        print(f"\n✂️  Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Show sample chunk
        if chunks:
            print(f"\nSample chunk:")
            print(f"Length: {len(chunks[0].page_content)} chars")
            print(f"Preview: {chunks[0].page_content[:150]}...")
        
        return chunks
    
    def create_vectorstore(self, chunks: List):
        """Create or load vector store"""
        print(f"\n🔢 Creating vector store...")
        
        # Check if vectorstore exists
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector store at {self.persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        print(f"✓ Vector store ready with {self.vectorstore._collection.count()} embeddings")
    
    def setup_chain(self):
        """Setup the conversational chain"""
        print(f"\n🔗 Setting up conversational chain...")
        
        # Custom prompt
        system_template = """You are a helpful AI assistant that answers questions about PDF documents.

Use the following context to answer the user's question. If you don't know the answer or can't find it in the context, say so clearly.

Context:
{context}

Chat History:
{chat_history}

Instructions:
1. Answer based on the provided context
2. Be specific and cite relevant information
3. If the answer isn't in the context, say "I cannot find this information in the provided documents"
4. Maintain conversation continuity using chat history

Answer the question thoughtfully and accurately."""
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 3}
        )
        
        # Create chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_template(system_template)
            }
        )
        
        print("✓ Conversational chain ready")
    
    def initialize(self):
        """Initialize the complete chatbot"""
        # Load PDFs
        documents = self.load_pdfs()
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        self.create_vectorstore(chunks)
        
        # Setup chain
        self.setup_chain()
        
        print("\n✅ Chatbot is ready! Start asking questions.\n")
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question to the chatbot
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.chain:
            raise ValueError("Chatbot not initialized. Call initialize() first.")
        
        # Get response
        result = self.chain({"question": question})
        
        # Format response
        response = {
            "answer": result["answer"],
            "sources": [
                {
                    "page": doc.metadata.get("page", "N/A"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:150] + "..."
                }
                for doc in result["source_documents"]
            ]
        }
        
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("="*60)
        print("PDF Q&A CHATBOT")
        print("="*60)
        print("Ask questions about your PDF documents!")
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'reset' to clear conversation history")
        print("="*60 + "\n")
        
        while True:
            # Get user input
            question = input("You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Check for reset
            if question.lower() == 'reset':
                self.memory.clear()
                print("\n🔄 Conversation history cleared!\n")
                continue
            
            # Skip empty input
            if not question:
                continue
            
            # Get response
            try:
                result = self.ask(question)
                
                # Print answer
                print(f"\n🤖 Assistant: {result['answer']}")
                
                # Print sources
                if result['sources']:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['source']} (Page {source['page']})")
                
                print()  # Empty line for readability
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


# Main execution
if __name__ == "__main__":
    # Configuration
    PDF_DIRECTORY = "./data/pdfs"  # Put your PDFs here
    PERSIST_DIRECTORY = "./pdf_chatbot_db"
    
    # Create PDF directory if it doesn't exist
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    # Initialize chatbot
    chatbot = PDFChatbot(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        model="gpt-4",
        chunk_size=1000,
        chunk_overlap=200,
        k=3
    )
    
    # Initialize (load PDFs, create embeddings)
    chatbot.initialize()
    
    # Start interactive chat
    chatbot.chat()