import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


class ChatWebApp:
    def __init__(self, url: str = 'http://localhost:11434/v1', model: str = 'gpt-oss:20b'):
        self.app = FastAPI()
        self.url = url
        self.model = model
        self.openai_client = None
        self.agent = None
        self.active_tasks = {}
        self.chat_histories = {}
        self.setup_routes()
        self.initialize_agent()

    def initialize_agent(self):
        """AI 에이전트 초기화"""
        try:
            from openai import AsyncOpenAI
            from agents import (Agent,
                                set_default_openai_client,
                                set_default_openai_api,
                                set_tracing_disabled)

            self.openai_client = AsyncOpenAI(
                api_key="local",
                base_url=self.url,
            )

            set_tracing_disabled(True)
            set_default_openai_client(self.openai_client)
            set_default_openai_api("chat_completions")

            self.agent = Agent(
                name="Research Chat Agent",
                instructions="You are a versatile AI assistant who provides researcher-level technical answers for expert queries while maintaining a warm, natural tone for everyday conversations.",
                model=self.model,
            )
            print(f"✅ Agent initialized with model: {self.model}")

        except ImportError as e:
            print(f"❌ Error importing agents library: {e}")
            self.agent = None
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            self.agent = None

    def setup_routes(self):
        """웹 라우트 설정"""

        @self.app.get("/")
        async def get_chat_page():
            return HTMLResponse(content=self.get_html_content())

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_tasks[websocket] = None
            self.chat_histories[websocket] = []

            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    if data.get("type") == "user_message":
                        user_input = data.get("content", "").strip()
                        history = self.chat_histories.get(websocket, [])
                        
                        if self.active_tasks.get(websocket):
                            self.active_tasks[websocket].cancel()

                        task = asyncio.create_task(
                            self.stream_ai_response(websocket, user_input, history)
                        )
                        self.active_tasks[websocket] = task

                    elif data.get("type") == "user_request_stop":
                        if self.active_tasks.get(websocket):
                            self.active_tasks[websocket].cancel()
                            self.active_tasks[websocket] = None
                    
                    elif data.get("type") == "clear_chat":
                        if websocket in self.chat_histories:
                            self.chat_histories[websocket] = []

            except WebSocketDisconnect:
                print("클라이언트 연결 종료")
                if self.active_tasks.get(websocket):
                    self.active_tasks[websocket].cancel()
                if websocket in self.active_tasks:
                    del self.active_tasks[websocket]
                if websocket in self.chat_histories:
                    del self.chat_histories[websocket]
            except Exception as e:
                print(f"WebSocket 오류: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"오류 발생: {str(e)}"
                    }))
                except Exception:
                    pass

    async def stream_ai_response(self, websocket: WebSocket, user_input: str, history: list):
        """AI 응답을 스트리밍으로 전송"""
        if not self.agent:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "AI 에이전트가 초기화되지 않았습니다. agents 라이브러리를 확인해주세요."
            }))
            return

        try:
            from agents import Runner

            await websocket.send_text(json.dumps({
                "type": "ai_response_start"
            }))

            history.append({"role": "user", "content": user_input})

            result = Runner.run_streamed(self.agent, history)
            full_response = ""

            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    data = event.data
                    if data.type == "response.output_text.delta":
                        delta = getattr(data, "delta", "")
                        if delta:
                            full_response += delta
                            await websocket.send_text(json.dumps({
                                "type": "ai_response_delta",
                                "content": delta
                            }))
            
            history.append({"role": "assistant", "content": full_response})

            await websocket.send_text(json.dumps({
                "type": "ai_response_end"
            }))

        except asyncio.CancelledError:
            print("AI 응답 스트리밍이 중단되었습니다.")
            history.append({"role": "assistant", "content": full_response})
            await websocket.send_text(json.dumps({
                "type": "ai_response_end"
            }))
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"AI 응답 생성 중 오류: {str(e)}"
            }))
        finally:
            if websocket in self.active_tasks:
                self.active_tasks[websocket] = None

    def get_html_content(self):
        """HTML 콘텐츠 반환"""
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """웹 앱 실행"""
        print(f"URL: {self.url}")
        print(f"Model: {self.model}")
        print(f"웹 서버: http://{host}:{port}")

        uvicorn.run(self.app, host=host, port=port)


load_dotenv()

chat_app = ChatWebApp(
    url=os.environ.get('url', 'http://localhost:11434/v1'),
    model=os.environ.get('model', 'gpt-oss:20b')
)
app = chat_app.app


if __name__ == "__main__":
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 9000))
    chat_app.run(host=host, port=port)
