# LCEL (LangChain Expression Language)
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Step 1: Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Step 2: Create a model
model = AzureChatOpenAI(model="gpt-5.1-chat")

# Step 3: Create an output parser
output_parser = StrOutputParser()

# Step 4: Chain them together using the | operator
chain = prompt | model | output_parser

# Step 5: Invoke the chain
result = chain.invoke({"input": "What is 2+2?"})
print(result)