# =============================================================================
# Section 4.1 — Understanding Agent Thought Process: Verbose Mode
# Topic:  Enabling verbose=True on AgentExecutor to print the full ReAct
#         reasoning trace (Thought → Action → Observation → Final Answer)
#         to stdout for development and debugging.
# =============================================================================
# Expected output shape:
#
#   > Entering new AgentExecutor chain...
#   I need to find the company's revenue first, then the number of employees.
#
#   Action: company_knowledge
#   Action Input: revenue
#   Observation: Annual revenue for 2023 was $5 million.
#   Thought: Now I need to find the number of employees.
#   Action: company_knowledge
#   Action Input: employees
#   Observation: We have 50 employees across 3 offices.
#   Thought: Now I can calculate revenue per employee.
#   Action: calculator
#   Action Input: 5000000 / 50
#   Observation: The result is: 100000.0
#   Thought: I now know the final answer.
#   Final Answer: The revenue per employee is $100,000.
#
#   > Finished chain.
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0                          # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
RETRIEVER_K     = 2                          # documents retrieved per query
DEMO_QUERY      = "What is the revenue per employee?"  # demo question
# ──────────────────────────────────────────────────────────────────────────────

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version=API_VERSION,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=LLM_TEMPERATURE,
)


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations.
    Input should be a valid math expression like '2 + 2' or '(10 * 5) / 2'.
    """
    try:
        result = numexpr.evaluate(expression).item()
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


documents = [
    "The company was founded in 2020.",
    "Our main product is an AI assistant.",
    "We have 50 employees across 3 offices.",
    "Annual revenue for 2023 was $5 million.",
]

vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        api_version=API_VERSION,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    ),
)

retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
    name="company_knowledge",
    description=(
        "Search for information about the company, including founding date, "
        "products, employees, and revenue."
    ),
)

tools = [calculator, retriever_tool]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Prints step-by-step reasoning
)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": DEMO_QUERY})
    print("\nFinal Answer:", result["output"])
