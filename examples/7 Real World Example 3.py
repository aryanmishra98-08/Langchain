# Summarization Template
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

summary_template = """
Summarize the following text in {num_sentences} sentences:

{text}

Summary:
"""

prompt = PromptTemplate(
    input_variables=["text", "num_sentences"],
    template=summary_template
)

long_text = """
Python is a high-level programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
It has a large standard library and an extensive ecosystem of third-party packages.
Python is widely used in web development, data science, machine learning, and automation.
"""

formatted = prompt.format(text=long_text, num_sentences=2)

llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)