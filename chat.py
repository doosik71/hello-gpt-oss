import asyncio


async def prompt_user(question: str) -> str:

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, question)


async def chat(url: str = 'http://localhost:11434/v1',
               model: str = 'gpt-oss:20b') -> None:

    from openai import AsyncOpenAI
    from agents import (Agent, Runner,
                        set_default_openai_client,
                        set_default_openai_api,
                        set_tracing_disabled)

    openai_client = AsyncOpenAI(
        api_key="local",
        base_url=url,
    )

    set_tracing_disabled(True)
    set_default_openai_client(openai_client)
    set_default_openai_api("chat_completions")

    agent = Agent(
        name="Simple Chat Agent",
        instructions="You are a helpful assistant.",
        model=model,
    )

    while True:
        user_input = await prompt_user("🧑 > ")
        if user_input.strip().lower() in ("exit", "quit"):
            break

        print("🤖 > ", end="", flush=True)
        result = Runner.run_streamed(agent, user_input)

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                data = event.data
                if data.type == "response.output_text.delta":
                    delta = getattr(data, "delta", "")
                    if delta:
                        print(delta, end="", flush=True)

        print("", flush=True)


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    load_dotenv()
    url = os.environ.get('url', 'http://localhost:11434/v1')
    model = os.environ.get('model', 'gpt-oss:20b')
    print(f"URL = {url}")
    print(f"Model = {model}")
    print('Input "quit" or "exit" to terminate dialog')

    asyncio.run(chat(url, model))
