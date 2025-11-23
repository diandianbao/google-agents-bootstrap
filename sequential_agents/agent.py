import asyncio

from google.adk import Agent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner

from services import model_service

# 选择要使用的模型（可以修改这个变量来切换模型）
SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型

model = model_service.create_model(SELECTED_MODEL)

outline_agent = LlmAgent(
    name="OutlineAgent",
    model=model,
    description="""
        Create a blog outline for the given topic with:
        1. A catchy headline
        2. An introduction hook
        3. 3-5 main sections with 2-3 bullet points for each
        4. A concluding thought
        """,
    output_key="blog_outline"
)

print("✅ outline_agent created.")

writer_agent = LlmAgent(
    name="WriterAgent",
    model=model,
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key="blog_draft",  # The result of this agent will be stored with this key.
)

print("✅ writer_agent created.")

editor_agent = LlmAgent(
    name="EditorAgent",
    model=model,
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key="final_blog",  # This is the final output of the entire pipeline.
)

print("✅ editor_agent created.")

root_agent = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_agent, writer_agent, editor_agent],
)

print("✅ Sequential Agent created.")

async def main():
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(
        "写一篇关于太阳黑子的博客"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
