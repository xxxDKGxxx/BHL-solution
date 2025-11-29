from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="BHL-solution API",
    description=(
        "API demo for greeting users and processing prompts.\n\n"
        "Swagger UI is available at /docs and ReDoc at /redoc."
    ),
    version="0.1.0",
    contact={
        "name": "BHL-solution",
        "url": "https://example.com",
        "email": "contact@example.com",
    },
    swagger_ui_parameters={"syntaxHighlight": False},
)

class MessageResponse(BaseModel):
    message: str = Field(..., example="Hello World")

class PromptRequest(BaseModel):
    prompt: str = Field(..., example="Tell me a joke about FastAPI")
    user_id: Optional[str] = Field(None, example="user-123")

@app.get("/", response_model=MessageResponse, tags=["General"], summary="Health check")
async def root():
    return {"message": "Hello World"}


@app.get(
    "/hello/{name}",
    response_model=MessageResponse,
    tags=["General"],
    summary="Say hello",
    description="Greets a user by name.",
)
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post(
    "/prompt_process",
    response_model=MessageResponse,
    tags=["Prompts"],
    summary="Process a prompt",
    description=(
        "Accepts a prompt in the request body and returns a placeholder response.\n"
        "Replace the placeholder with your real processing logic."
    ),
)
async def get_prompt(body: PromptRequest):
    # TODO: Integrate your real processing here, e.g. model inference or DB lookup
    # result = some_service.preprocess(body.prompt)
    # return {"message": result}
    return {"message": f"Received prompt: {body.prompt}"}










