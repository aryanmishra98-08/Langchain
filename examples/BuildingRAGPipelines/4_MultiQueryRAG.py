from langchain.retrievers.multi_query import MultiQueryRetriever
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
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Multi-query retriever generates multiple search queries
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Single user query generates multiple searches
question = "What are the benefits of machine learning?"
unique_docs = multi_query_retriever.get_relevant_documents(query=question)

print(f"Retrieved {len(unique_docs)} unique documents")