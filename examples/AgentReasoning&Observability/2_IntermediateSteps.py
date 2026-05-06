# =============================================================================
# Section 4.2 — Capturing Intermediate Steps
# Topic:  Setting return_intermediate_steps=True on AgentExecutor to access
#         the full reasoning chain programmatically. Each step is a tuple of
#         (AgentAction, observation_string).
# =============================================================================
# The intermediate_steps list enables:
#   - Post-hoc auditing of agent decisions
#   - Custom step-level logging or metrics
#   - Building UIs that show reasoning progress
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
        "Search for information about the company, including employees and revenue."
    ),
)

tools = [calculator, retriever_tool]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# Set return_intermediate_steps=True on the executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": DEMO_QUERY})

    # Access the reasoning chain
    print("Final Answer:", result["output"])
    print("\nReasoning Steps:")
    for step in result["intermediate_steps"]:
        action, observation = step
        print(f"\nAction: {action.tool}")
        print(f"Input: {action.tool_input}")
        print(f"Output: {observation}")
