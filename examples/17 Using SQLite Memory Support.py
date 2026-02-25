# Using SQLite Memory Support
from langchain_community.chat_message_histories import SQLChatMessageHistory
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

# SQLite history factory
def get_sql_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"  # Persistent database
    )

# Wrap with SQL history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_sql_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use it - history persists across program restarts!
config = {"configurable": {"session_id": "persistent_session"}}

response = chain_with_history.invoke(
    {"input": "Remember: my favorite color is blue"},
    config=config
)
print(response)

# Even after restarting the program, this will remember!
response = chain_with_history.invoke(
    {"input": "What's my favorite color?"},
    config=config
)
print(response)  