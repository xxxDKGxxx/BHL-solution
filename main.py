from fastapi import FastAPI
from pydantic import BaseModel

from database.database import Database
from database.sentence_transformer_embedder import SentenceTransformerEmbedder
from handler.defaulthandler import PromptHandler
from llms.gemini_llm import GeminiLLM

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})


prompthandler = PromptHandler(
    db_context=Database(SentenceTransformerEmbedder()),
    model=GeminiLLM())


class Prompt(BaseModel):
    prompt: str

@app.post("/prompt")
async def get_prompt(prompt: Prompt, skip_cached: bool = True):

    result, cached  = prompthandler.generate_answer(prompt.prompt, skip_cached)

    return {"result": result, "cached": cached}
