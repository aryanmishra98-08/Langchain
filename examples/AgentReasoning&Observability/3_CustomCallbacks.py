# =============================================================================
# Section 4.3 — Custom Callbacks for Monitoring
# Topic:  A custom BaseCallbackHandler that intercepts agent lifecycle events
#         (action taken, tool start, tool end, agent finish) and stores them
#         for later inspection. Passed via config={"callbacks": [...]}.
#
# BaseCallbackHandler still works in LangChain 1.0 and is passed the same way.
# For new production code, prefer the middleware system (see ProductionPatterns/).
# Callbacks remain useful for dev-time inspection and third-party integrations.
#
# Note: on_tool_end receives a ToolMessage object in modern versions;
# always use str(output) for safe string conversion.
# =============================================================================

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from langchain_core.tools import tool, create_retriever_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_agent

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

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant.",
)


class CustomAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action"""
        print(f"\nAgent is using: {action.tool}")
        print(f"   Input: {action.tool_input}")
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
        })

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes"""
        print(f"\nAgent finished: {finish.return_values}")
        self.steps.append({
            "type": "finish",
            "output": finish.return_values,
        })

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when tool starts"""
        print(f"   Tool starting...")

    def on_tool_end(self, output: str, **kwargs):
        """Called when tool ends"""
        # output is now a ToolMessage in modern versions; use str() for safety
        output_str = str(output)
        print(f"   Tool output: {output_str[:100]}...")


if __name__ == "__main__":
    callback_handler = CustomAgentCallbackHandler()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": DEMO_QUERY}]},
        config={"callbacks": [callback_handler]},
    )

    print("\n\nCaptured Steps:")
    for i, step in enumerate(callback_handler.steps, 1):
        print(f"{i}. {step}")

    print("\nFinal Answer:", result["messages"][-1].content)
