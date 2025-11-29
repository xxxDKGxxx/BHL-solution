from fastapi import FastAPI

from handler.defaulthandler import PromptHandler
from database.mockdbcontext import MockDbContext
from models.defaultmodel import  DefaultModel
app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})


prompthandler = PromptHandler(db_context=MockDbContext(),model=DefaultModel())


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}



@app.post("/prompt_process")
async def get_prompt(prompt: str):

    result  = prompthandler.generate_answer(prompt)

    return {"result": result}
