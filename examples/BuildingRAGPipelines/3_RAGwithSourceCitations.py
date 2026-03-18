from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(
            f"[Document {i+1}] (Source: {source}, Page: {page})\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# Build chain
rag_chain_with_sources = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query
response = rag_chain_with_sources.invoke("Explain the key concepts")
print(response)