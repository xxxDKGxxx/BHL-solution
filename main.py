import json

from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket

from database.database import Database
from database.sentence_transformer_embedder import SentenceTransformerEmbedder
from handler.defaulthandler import PromptHandler
from llms.gemini_llm import GeminiLLM

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

origins = [
    "http://localhost",
    "http://127.0.0.1",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # albo ["*"] w dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prompthandler = PromptHandler(
    db_context=Database(SentenceTransformerEmbedder()),
    model=GeminiLLM())


class Prompt(BaseModel):
    prompt: str
    skip_cached: bool

@app.post("/prompt")
async def get_prompt(prompt: Prompt):

    prompthandler.model = GeminiLLM()

    result, cached  = prompthandler.generate_answer(prompt.prompt, prompt.skip_cached)

    return {"result": result, "cached": cached}


@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()

    prompthandler.model = GeminiLLM()

    while True:
        try:
            raw_message = await ws.receive_text()

            # Oczekujemy JSON
            try:
                data = json.loads(raw_message)
                prompt_obj = Prompt(**data)
                response, cached = prompthandler.generate_answer(prompt_obj.prompt, prompt_obj.skip_cached)
                await ws.send_text(json.dumps({
                    "result": response,
                    "cached": cached
                }))

            except (json.JSONDecodeError, ValidationError) as e:
                print(e)
                await ws.send_text(f"Invalid payload: {e}")
                continue

        except Exception as e:
            print(e)
            break

    # await ws.close()