# Translation Template
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate from {source_lang} to {target_lang}."),
    ("human", "{text}")
])

llm = AzureChatOpenAI(model="gpt-5.1-chat")

# Translate to Spanish
messages = translation_template.format_messages(
    source_lang="English",
    target_lang="Spanish",
    text="Hello, how are you?"
)

response = llm.invoke(messages)
print(response.content)
# Expected: Hola, ¿cómo estás?