# Chaining Multiple Operations
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

# First chain: Generate a topic
topic_prompt = ChatPromptTemplate.from_template(
    "Suggest a random topic related to {subject}"
)
topic_chain = topic_prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Second chain: Write about the topic
writing_prompt = ChatPromptTemplate.from_template(
    "Write a short paragraph about: {topic}"
)
writing_chain = writing_prompt | AzureChatOpenAI(model="gpt-5.1-chat") | StrOutputParser()

# Manually chain them (in practice, you'd use RunnablePassthrough)
subject = "artificial intelligence"
topic = topic_chain.invoke({"subject": subject})
print(f"Generated topic: {topic}\n")

paragraph = writing_chain.invoke({"topic": topic})
print(f"Paragraph:\n{paragraph}")