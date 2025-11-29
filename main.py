from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

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

    result, cached  = prompthandler.generate_answer(prompt.prompt, prompt.skip_cached)

    return {"result": result, "cached": cached}
