# Window Memory Pattern - N messages
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv('keys/.env')

# Custom window memory implementation
class WindowChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """Keeps only the last k messages."""
    
    messages: List[BaseMessage] = Field(default_factory=list)
    k: int = Field(default=10)  # Keep last 10 messages
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add messages and keep only last k."""
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]  # Keep only last k
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

# Setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Store with window memory
window_store = {}

def get_window_history(session_id: str):
    if session_id not in window_store:
        window_store[session_id] = WindowChatMessageHistory(k=4)  # Keep last 4 msgs
    return window_store[session_id]

chain_with_window = RunnableWithMessageHistory(
    chain,
    get_window_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Test it
config = {"configurable": {"session_id": "window_test"}}

# Add many messages
messages_to_send = [
    "My favorite color is blue",
    "I like pizza",
    "I have a dog named Max",
    "I live in California",
    "I work as an engineer"
]

for msg in messages_to_send:
    response = chain_with_window.invoke({"input": msg}, config=config)
    print(f"User: {msg}")
    print(f"AI: {response}\n")

# Older messages forgotten (outside window)
response = chain_with_window.invoke(
    {"input": "What's my favorite color?"},
    config=config
)
print(f"AI: {response}")  # Might not remember (too old)

# Recent messages remembered
response = chain_with_window.invoke(
    {"input": "What's my job?"},
    config=config
)
print(f"AI: {response}")  # "You work as an engineer" ✅