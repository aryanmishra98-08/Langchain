# Programing Concept - Code
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv("keys/.env")

prompt = ChatPromptTemplate.from_template("""
Explain the following concept using a code example: {concept} in the following language: {language}
""")
model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = (
    {
        "concept": RunnablePassthrough(),  # Pass concept as-is
        "language": RunnablePassthrough()   # Pass language as-is
    }
    | prompt
    | model
    | output_parser
)

concept = "chaining multiple operations in Langchain"
language = "Python"
result = chain.invoke({"concept": concept, "language": language})
print(result)

for chunk in chain.stream({"concept": concept, "language": language}):
    print(chunk, end="", flush=True)