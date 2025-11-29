from fastapi import FastAPI

from Handler.handler import PromptHandler
from Database.mockdbcontext import MockDbContext
app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})


prompthandler = PromptHandler(db_context=MockDbContext())

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}



@app.post("/prompt_process")
async def get_prompt(prompt: str):

    prompthandler.preprocess()


    return None










