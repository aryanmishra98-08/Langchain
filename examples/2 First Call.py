# 01_first_llm_call.py
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Initialize the model
llm = AzureChatOpenAI(
    model="gpt-5.1-chat",
    temperature=1 # Controls randomness (0-2)
)

# Invoke the model
response = llm.invoke("What is LangChain?")
print(response.content)