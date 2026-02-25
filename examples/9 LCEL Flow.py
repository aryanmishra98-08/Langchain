
# Chain Flow
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv("keys/.env")

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = AzureChatOpenAI(model="gpt-5.1-chat")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Let's trace what happens at each step:
print("Step 1: Input dict")
input_data = {"topic": "Python"}
print(input_data)

print("\nStep 2: After prompt formatting")
formatted = prompt.invoke(input_data)
print(formatted)

print("\nStep 3: After model invocation")
model_output = model.invoke(formatted)
print(type(model_output), model_output.content[:50])

print("\nStep 4: After parsing")
final_output = output_parser.invoke(model_output)
print(final_output)

print("\n--- OR just use the chain ---")
print(chain.invoke({"topic": "Python"}))