from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat")

# Stream the response token by token
for chunk in llm.stream("Write a short poem about Python programming"):
    print(chunk.content, end="", flush=True)