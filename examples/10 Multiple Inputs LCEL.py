# Multiple Inputs LCEL
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Template with multiple variables
prompt = ChatPromptTemplate.from_template(
    "Translate the following {source_lang} text to {target_lang}: {text}"
)

model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Invoke with multiple inputs
result = chain.invoke({
    "source_lang": "English",
    "target_lang": "French",
    "text": "Good morning"
})

print(result)
# Expected: Bonjour