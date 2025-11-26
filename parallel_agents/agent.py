import asyncio

from google.adk import Agent
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.runners import InMemoryRunner

from tools.serp import serpapi_search

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型

model = model_service.create_model(SELECTED_MODEL)

# Tech Researcher: Focuses on AI and ML trends.
tech_researcher = LlmAgent(
    name="TechResearcher",
    model=model,
    instruction="""Research the latest AI/ML trends. 
                    Include 3 key developments,
                    the main companies involved, and the potential impact. 
                    Keep the report very concise (100 words).""",
    tools=[serpapi_search],
    output_key="tech_research",
)

print("✅ tech_researcher created.")

health_researcher = LlmAgent(
    name="HealthResearcher",
    model=model,
    instruction=""" Research recent medical breakthroughs. 
                    Include 3 significant advances,
                    their practical applications, and estimated timelines. 
                    Keep the report concise (100 words).""",
    tools=[serpapi_search],
    output_key="health_research",
)

print("✅ health_researcher created.")

# Finance Researcher: Focuses on fintech trends.
finance_researcher = LlmAgent(
    name="FinanceResearcher",
    model=model,
    instruction="""Research current fintech trends. 
                   Include 3 key trends,
                   their market implications, 
                   and the future outlook. Keep the report concise (100 words).""",
    tools=[serpapi_search],
    output_key="finance_research",  # The result will be stored with this key.
)

print("✅ finance_researcher created.")

aggregator_agent = LlmAgent(
    name="AggregatorAgent",
    model=model,
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research 
    findings into a single executive summary:

    **Technology Trends:**
    {tech_research}

    **Health Breakthroughs:**
    {health_research}

    **Finance Innovations:**
    {finance_research}

    Your summary should highlight common themes, surprising connections, 
    and the most important key takeaways from all three reports. 
    The final summary should be around 200 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

print("✅ aggregator_agent created.")

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

print("✅ Parallel and Sequential Agents created.")


async def run_debug(question: str):
    """运行调试会话"""
    print("\n🚀 开始调试会话...")
    runner = InMemoryRunner(agent=root_agent)
    result = await runner.run_debug(question)
    print(result)


if __name__ == "__main__":
    asyncio.run(run_debug("Run the daily executive briefing on Tech, Health, and Finance"))
