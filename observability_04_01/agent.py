import asyncio
import logging
import os

from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins import BasePlugin

from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("✅ Logging configured")

from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools.agent_tool import AgentTool

from tools.serp import serpapi_search

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型
model = model_service.create_model(SELECTED_MODEL)

# ---- Intentionally pass incorrect datatype - `str` instead of `List[str]` ----
def count_papers(papers: str):
    """
    This function counts the number of papers in a list of strings.
    Args:
      papers: A list of strings, where each string is a research paper.
    Returns:
      The number of papers in the list.
    """
    return len(papers)


# Google Search agent
google_search_agent = LlmAgent(
    name="google_search_agent",
    model=model,
    description="Searches for information using Google search",
    instruction="""Use the google_search tool to find information on the given topic. Return the raw search results.
    If the user asks for a list of papers, then give them the list of research papers you found and not the summary.""",
    tools=[serpapi_search]
)


# Root agent
root_agent = LlmAgent(
    name="research_paper_finder_agent",
    model=model,
    instruction="""Your task is to find research papers and count them. 

    You MUST ALWAYS follow these steps:
    1) Find research papers on the user provided topic using the 'google_search_agent'. 
    2) Then, pass the papers to 'count_papers' tool to count the number of papers returned.
    3) Return both the list of research papers and the total number of papers.
    """,
    tools=[AgentTool(agent=google_search_agent), count_papers]
)

runner = InMemoryRunner(
    agent=root_agent,
    plugins=[
        LoggingPlugin()
    ]
)

async def run_agent():
    print("🚀 Running agent with LoggingPlugin...")
    print("📊 Watch the comprehensive logging output below:\n")

    response = await runner.run_debug("Find recent papers on quantum computing")

if __name__ == "__main__":
    asyncio.run(run_agent())