# Using Redis Memory Support
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Setup chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Redis history factory with TTL (Time To Live)
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=86400  # Messages expire after 24 hours
    )

# Wrap with Redis history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_redis_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use it
config = {"configurable": {"session_id": "redis_session"}}

response = chain_with_history.invoke(
    {"input": "Store this in Redis!"},
    config=config
)
print(response)
