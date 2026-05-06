# =============================================================================
# Section 4.4 — Logging for Production
# Topic:  A ProductionAgentCallback that writes structured log entries to a
#         timestamped file. verbose=False keeps the console clean in production.
# =============================================================================
# Pro tip: For production observability, consider LangSmith
# (set LANGCHAIN_TRACING_V2=true). It captures full traces including LLM
# calls, token usage, and latencies without writing custom callbacks.
#
# Log levels used:
#   INFO  → normal agent lifecycle events (action, tool output, finish)
#   ERROR → LLM errors
# =============================================================================

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool, create_retriever_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import numexpr

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / "keys" / ".env")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
LLM_TEMPERATURE = 0              # 0 = deterministic output
API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION")  # Azure OpenAI API version
RETRIEVER_K     = 2              # documents retrieved per query
LOG_LEVEL       = logging.INFO   # logging verbosity level
DEMO_QUERY      = "What is the revenue per employee?"  # demo question
# ──────────────────────────────────────────────────────────────────────────────

# Setup logging
logging.basicConfig(
    filename=f'agent_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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


class ProductionAgentCallback(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        logging.info(
            f"AGENT_ACTION | Tool: {action.tool} | Input: {action.tool_input}"
        )

    def on_tool_end(self, output, **kwargs):
        logging.info(f"TOOL_OUTPUT | Output: {str(output)[:200]}")

    def on_agent_finish(self, finish, **kwargs):
        logging.info(f"AGENT_FINISH | Result: {finish.return_values}")

    def on_llm_error(self, error: BaseException, **kwargs):
        logging.error(f"LLM_ERROR | Error: {str(error)}")


# Use in production
prod_callback = ProductionAgentCallback()
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[prod_callback],
    verbose=False,  # Disable console output in production
)

if __name__ == "__main__":
    result = agent_executor.invoke({"input": DEMO_QUERY})
    print("Result:", result["output"])
    print("(Check the generated agent_logs_*.log file for detailed logs)")
