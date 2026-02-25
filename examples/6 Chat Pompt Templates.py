from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Create a chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {profession} assistant."),
    ("human", "Help me with: {task} and believe me, I need it badly for {reason}")
])

# Format the prompt
formatted = chat_template.format_messages(
    profession="Python programming",
    task="understanding decorators",
    reason="an upcoming project deadline"
)

# Invoke the LLM
llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)