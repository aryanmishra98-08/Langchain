# Conversation Memory
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Step 1: Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant. Remember what users tell you."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Step 2: Build the chain
model = AzureChatOpenAI(model="gpt-5.1-chat")
chain = prompt | model | StrOutputParser()

# Step 3: Set up history storage
conversation_store = {}

def get_chat_history(session_id: str):
    if session_id not in conversation_store:
        conversation_store[session_id] = InMemoryChatMessageHistory()
    return conversation_store[session_id]

# Step 4: Wrap with history
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# Step 5: Have a conversation
session_config = {"configurable": {"session_id": "session_001"}}

print("Conversation 1:")
print("User: Hi, I'm learning LangChain!")
response = conversational_chain.invoke(
    {"question": "Hi, I'm learning LangChain!"},
    config=session_config
)
print(f"AI: {response}\n")

print("User: What am I learning?")
response = conversational_chain.invoke(
    {"question": "What am I learning?"},
    config=session_config
)
print(f"AI: {response}\n")  # Will say "LangChain"!

print("User: Can you quiz me on it?")
response = conversational_chain.invoke(
    {"question": "Can you quiz me on it?"},
    config=session_config
)
print(f"AI: {response}\n")