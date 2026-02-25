from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
load_dotenv("keys/.env")

llm = AzureChatOpenAI(model="gpt-5.1-chat")

# Create a conversation with different message types
messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print(response.content)