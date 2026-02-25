# Streaming Chains
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

prompt = ChatPromptTemplate.from_template("Write a story about {topic}")
model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Stream the output
for chunk in chain.stream({"topic": "a robot learning to paint"}):
    print(chunk, end="", flush=True)