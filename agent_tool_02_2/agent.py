import asyncio
import uuid
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("✅ ADK components imported successfully.")

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型

model = model_service.create_model(SELECTED_MODEL)

# MCP integration with Everything Server
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

print("✅ MCP Tool created")


root_agent = LlmAgent(
    model=model,
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)

async def run_debug(question: str):
    """运行调试会话"""
    print("\n🚀 开始调试会话...")
    runner = InMemoryRunner(agent=root_agent)
    result = await runner.run_debug(question, verbose=True)


if __name__ == "__main__":
    asyncio.run(run_debug("Provide a sample tiny image"))
