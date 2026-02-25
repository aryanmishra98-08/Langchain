from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv("keys/.env")

# Create a template with a variable
template = "Tell me a {adjective} joke about {topic}."

prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template
)

# Format the prompt with values
formatted_prompt = prompt.format(adjective="funny", topic="programming")
print(formatted_prompt)
# Output: Tell me a funny joke about programming.

# Use with an LLM
llm = AzureChatOpenAI(model="gpt-5.1-chat")
response = llm.invoke(formatted_prompt)
print(response.content)