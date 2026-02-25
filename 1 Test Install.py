from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")


# This creates a ChatOpenAI instance
llm = AzureChatOpenAI(model="gpt-5.1-chat", temperature=0.9)
print("LangChain installed successfully!")