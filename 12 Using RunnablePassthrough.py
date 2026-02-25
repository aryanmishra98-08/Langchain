# Using RunnablePassthrough
# 13_runnable_passthrough.py
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv("keys/.env")

# This chain adds context to the answer
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below:

Context: {context}

Question: {question}

Answer:
""")

model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

# Create a chain that passes through context
chain = (
    {
        "context": RunnablePassthrough(),  # Pass context as-is
        "question": RunnablePassthrough()   # Pass question as-is
    }
    | prompt
    | model
    | output_parser
)

result = chain.invoke({
    "context": "The sky appears blue due to Rayleigh scattering.",
    "question": "Why is the sky blue?"
})

print(result)