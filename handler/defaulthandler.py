from generalize_answer import generalize_prompt
from interface.abstractdatabasecontext import DatabaseContext
from interface.abstractprompthandler import AbstractPromptHandler
from interface.abstractmodel import AbstractModel
from relevance_checkers.cross_encoder_relevance_checker import CrossEncoderRelevanceChecker
from sklearn.metrics.pairwise import cosine_similarity

class PromptHandler(AbstractPromptHandler):

    encoder = CrossEncoderRelevanceChecker()

    def __init__(self,db_context: DatabaseContext,model: AbstractModel):
        self._db_context = db_context
        self._model = model

    def generate_answer(self, prompt: str,threshold=0.5) -> str:

        answer,_ = self._db_context.get(prompt)

        if (self.encoder.check_relevance(prompt,answer)<= threshold or answer is None):

            (generalized_prompt,default_answer) = generalize_prompt(prompt)

            self._db_context.insert(generalize_prompt,default_answer)

            return default_answer

        return answer









