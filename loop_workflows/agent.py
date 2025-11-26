import asyncio

from google.adk import Agent
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool

from tools.serp import serpapi_search

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型

model = model_service.create_model(SELECTED_MODEL)

initial_writer_agent = LlmAgent(
    name="InitialWriterAgent",
    model=model,
    instruction="""Based on the user's prompt, 
                    write the first draft of a short story (around 100-150 words).
                    Output only the story text, with no introduction or explanation.""",
    output_key="current_story",  # Stores the first draft in the state.
)

print("✅ initial_writer_agent created.")

# This agent's only job is to provide feedback or the approval signal. It has no tools.
critic_agent = LlmAgent(
    name="CriticAgent",
    model=model,
    instruction="""You are a constructive story critic. Review the story provided below.
    Story: {current_story}

    Evaluate the story's plot, characters, and pacing.
    - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique",  # Stores the feedback in the state.
)

print("✅ critic_agent created.")


# This is the function that the RefinerAgent will call to exit the loop.
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED',
    indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


print("✅ exit_loop function created.")

refiner_agent = LlmAgent(
    name="RefinerAgent",
    model=model,
    instruction="""You are a story refiner. You have a story draft and critique.

    Story Draft: {current_story}
    Critique: {critique}

    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
    output_key="current_story",  # It overwrites the story with the new, refined version.
    tools=[FunctionTool(exit_loop)]  # The tool is now correctly initialized with the function reference.
)

print("✅ refiner_agent created.")

# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,  # Prevents infinite loops
)

# The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop],
)

print("✅ Loop and Sequential Agents created.")

async def run_debug(question: str):
    """运行调试会话"""
    print("\n🚀 开始调试会话...")
    runner = InMemoryRunner(agent=root_agent)
    result = await runner.run_debug(question)
    print(result)


if __name__ == "__main__":
    asyncio.run(run_debug("Run the daily executive briefing on Tech, Health, and Finance"))
