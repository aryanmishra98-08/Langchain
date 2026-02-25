# Multi User Memory Support
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv('keys/.env')

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat")| StrOutputParser()

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# User 1's conversation
user1_config = {"configurable": {"session_id": "alice_123"}}

print("=== User 1 (Alice) ===")
response = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config=user1_config
)
print(f"AI: {response}\n")

# User 2's conversation (different session)
user2_config = {"configurable": {"session_id": "bob_456"}}

print("=== User 2 (Bob) ===")
response = chain_with_history.invoke(
    {"input": "My name is Bob"},
    config=user2_config
)
print(f"AI: {response}\n")

# User 1 again - remembers Alice
print("=== User 1 (Alice) ===")
response = chain_with_history.invoke(
    {"input": "What's my name?"},
    config=user1_config
)
print(f"AI: {response}")  # "Your name is Alice"

# User 2 again - remembers Bob
print("\n=== User 2 (Bob) ===")
response = chain_with_history.invoke(
    {"input": "What's my name?"},
    config=user2_config
)
print(f"AI: {response}")  # "Your name is Bob"