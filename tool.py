import asyncio
from agents import function_tool


@function_tool
async def weather_tool(location: str) -> str:
    """Get weather information of the location."""

    import requests

    location = location.strip().replace(' ', "%20")
    url = f"https://wttr.in/{location}"
    response = requests.get(url)

    return response.text


@function_tool
async def datetime_tool() -> str:
    """Get date and time."""

    from datetime import datetime

    now = datetime.now().astimezone()

    return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")


async def prompt_user(question: str) -> str:
    """Async input prompt function"""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, question)


async def chat(ip: str = 'localhost', port: str = '11434') -> None:

    from openai import AsyncOpenAI
    from agents import (
        Agent,
        Runner,
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled
    )

    openai_client = AsyncOpenAI(
        api_key="local",
        base_url=f"http://{ip}:{port}/v1",
    )

    # Configure agents SDK
    set_tracing_disabled(True)
    set_default_openai_client(openai_client)
    set_default_openai_api("chat_completions")

    # Create agent
    agent = Agent(
        name="My Agent",
        instructions="You are a helpful assistant.",
        tools=[weather_tool, datetime_tool],
        model="gpt-oss:20b",
    )

    while True:
        user_input = await prompt_user("🧑 > ")
        if user_input.strip().lower() in ("exit", "quit"):
            break

        print("🤖 > ", end="", flush=True)
        result = Runner.run_streamed(agent, user_input)

        # Process streaming results
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                data = event.data
                if data.type == "response.output_text.delta":
                    print(data.delta, end="", flush=True)

                continue

            # if event.type == "run_item_stream_event":
            #     if event.item.type == "tool_call_item":
            #         print(f"[{event.item.agent.tools[0].name}]",
            #               end=" ", flush=True)

        print("", flush=True)


if __name__ == "__main__":

    from dotenv import load_dotenv
    import os

    load_dotenv()
    ip = os.environ.get('ip', 'localhost')
    print(f"Connecting to {ip}")

    asyncio.run(chat(ip))
