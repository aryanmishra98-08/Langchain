# Code Review Template
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

code_review_template = """
You are a senior software engineer performing a code review for the following language: {language}.
Review the following code and provide feedback on potential issues, improvements, and best practices.
{code}
Feedback:
"""

prompt = PromptTemplate(
    input_variables=["language", "code"],
    template=code_review_template
)

code_snippet = """
def add(a, b):
    return a + b
"""
formatted = prompt.format(language="Python", code=code_snippet)
llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted)
print(response.content)
