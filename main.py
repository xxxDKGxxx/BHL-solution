from fastapi import FastAPI

app = FastAPI()






@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}



@app.post("/prompt_process")
async def get_prompt(prompt: str):
    result = .preprocess(prompt)
    return result
    #










