import asyncio

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_memory_tool import load_memory
from google.genai import types

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型


async def main():
    model = model_service.create_model(SELECTED_MODEL)

    async def run_session(
        runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
    ):
        """Helper function to run queries in a session and display responses."""
        print(f"\n### Session: {session_id}")

        # Create or retrieve session
        try:
            session = await session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
        except:
            session = await session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )

        # Convert single query to list
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        # Process each query
        for query in user_queries:
            print(f"\nUser > {query}")
            query_content = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream agent response
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        print(f"Model: > {text}")


    print("✅ Helper functions defined.")

    memory_service = (
        InMemoryMemoryService()
    )  #

    # Define constants used throughout the notebook
    APP_NAME = "MemoryDemoApp"
    USER_ID = "demo_user"

    # Create agent
    # user_agent = LlmAgent(
    #     model=model,
    #     name="MemoryDemoAgent",
    #     instruction="Answer user questions in simple words.",
    # )
    #
    # print("✅ Agent created")
    #
    # session_service = InMemorySessionService()  # Handles conversations
    #
    # # Create runner with BOTH services
    # runner = Runner(
    #     agent=user_agent,
    #     app_name="MemoryDemoApp",
    #     session_service=session_service,
    #     memory_service=memory_service,  # Memory service is now available!
    # )
    #
    # print("✅ Agent and Runner created with memory support!")
    #
    # await run_session(
    #     runner,
    #     "My favorite color is blue-green. Can you write a Haiku about it?",
    #     "conversation-01",  # Session ID
    # )
    #
    # session = await session_service.get_session(
    #     app_name=APP_NAME, user_id=USER_ID, session_id="conversation-01"
    # )
    #
    # # Let's see what's in the session
    # print("📝 Session contains:")
    # for event in session.events:
    #     text = (
    #         event.content.parts[0].text[:60]
    #         if event.content and event.content.parts
    #         else "(empty)"
    #     )
    #     print(f"  {event.content.role}: {text}...")
    #
    #     # This is the key method!
    # await memory_service.add_session_to_memory(session)
    # print("✅ Session added to memory!")

    # Create agent
    user_agent = LlmAgent(
        model=model,
        name="MemoryDemoAgent",
        instruction="Answer user questions in simple words. Use load_memory tool if you need to recall past conversations.",
        tools=[
            load_memory
        ],  # Agent now has access to Memory and can search it whenever it decides to!
    )

    print("✅ Agent with load_memory tool created.")

    session_service = InMemorySessionService()  # Handles conversations
    # Create a new runner with the updated agent
    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # await run_session(runner, "What is my favorite color?", "color-test")

    await run_session(runner, "My birthday is on March 15th.", "birthday-session-01")

    birthday_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="birthday-session-01"
    )

    await memory_service.add_session_to_memory(birthday_session)

    print("✅ Birthday session saved to memory!")

    await run_session(
        runner, "When is my birthday?", "birthday-session-02"  # Different session ID
    )

    search_response = await memory_service.search_memory(
        app_name=APP_NAME, user_id=USER_ID, query="What is the user's favorite color?"
    )

    print("🔍 Search Results:")
    print(f"  Found {len(search_response.memories)} relevant memories")
    print()

    for memory in search_response.memories:
        if memory.content and memory.content.parts:
            text = memory.content.parts[0].text[:80]
            print(f"  [{memory.author}]: {text}...")

if __name__ == "__main__":
    asyncio.run(main())

