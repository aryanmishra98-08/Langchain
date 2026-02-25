# Q&A Template

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

qa_template = """
Use the following context to answer the question. 
If you cannot answer based on the context, say "I don't know."

Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_template
)

# Use the template
context = "LangChain is a framework for building LLM applications. It was created in 2022."
question = "When was LangChain created?"
quwestion2 = "Who created LangChain?"

formatted = prompt.format(context=context, question=question)
formatted2 = prompt.format(context=context, question=quwestion2)

llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)

response2 = llm.invoke(formatted2)
print(response2.content)