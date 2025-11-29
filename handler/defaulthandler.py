from interface.abstractdatabasecontext import DatabaseContext
from interface.abstractprompthandler import AbstractPromptHandler
from interface.abstractmodel import AbstractModel

class PromptHandler(AbstractPromptHandler):

    def __init__(self,db_context: DatabaseContext,model: AbstractModel):
        pass

    def generate_answer(self, prompt: str) -> str:
        pass