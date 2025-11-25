import asyncio

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.apps.app import EventsCompactionConfig, App
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.genai import types

from services import model_service

SELECTED_MODEL = "qwen3:30b"  # 这里可以改成任意可用的模型

model = model_service.create_model(SELECTED_MODEL)

APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session
MODEL_NAME = SELECTED_MODEL

# Step 1: Create the same agent (notice we use LlmAgent this time)
chatbot_agent = LlmAgent(
    model=model,
    name="text_chat_bot",
    description="A text chatbot with persistent memory",
)

print("✅ Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")

research_app_compacting = App(
    name="research_app_compacting",
    root_agent=chatbot_agent,
    # This is the new part!
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

db_url = "sqlite+aiosqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

research_runner_compacting = Runner(
    app=research_app_compacting, session_service=session_service
)

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")

print("✅ Helper functions defined.")

async def main():
    await run_session(
        research_runner_compacting,
        "What is the latest news about AI in healthcare?",
        "compaction_demo",
    )

    # Turn 2
    await run_session(
        research_runner_compacting,
        "Are there any new developments in drug discovery?",
        "compaction_demo",
    )

    # Turn 3 - Compaction should trigger after this turn!
    await run_session(
        research_runner_compacting,
        "Tell me more about the second development you found.",
        "compaction_demo",
    )

    # Turn 4
    await run_session(
        research_runner_compacting,
        "Who are the main companies involved in that?",
        "compaction_demo",
    )

    print("---------------------------------------------------")

    # Get the final session state
    final_session = await session_service.get_session(
        app_name=research_runner_compacting.app_name,
        user_id=USER_ID,
        session_id="compaction_demo",
    )

    print("--- Searching for Compaction Summary Event ---")
    found_summary = False
    for event in final_session.events:
        # Compaction events have a 'compaction' attribute
        if event.actions and event.actions.compaction:
            print("\n✅ SUCCESS! Found the Compaction Event:")
            print(f"  Author: {event.author}")
            print(f"\n Compacted information: {event}")
            found_summary = True
            break

    if not found_summary:
        print(
            "\n❌ No compaction event found. Try increasing the number of turns in the demo."
        )

if __name__ == "__main__":
    asyncio.run(main())

