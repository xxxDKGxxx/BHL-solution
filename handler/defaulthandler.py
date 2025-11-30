from llms.general_llm import GeneralLLM
from .generalize_answer import generalize_prompt
from interface.abstractdatabasecontext import DatabaseContext
from interface.abstractprompthandler import AbstractPromptHandler
from interface.abstractmodel import AbstractModel
from relevance_checkers.cross_encoder_relevance_checker import CrossEncoderRelevanceChecker

class PromptHandler(AbstractPromptHandler):

    encoder = CrossEncoderRelevanceChecker()

    def __init__(self, db_context: DatabaseContext, model: AbstractModel) -> None:
        self._db_context = db_context
        self.model = model

    def generate_answer(self, prompt: str, skip_cached: bool, threshold=0.5) -> tuple[str, bool]:
        answer=""

        if not skip_cached:
            answer,_ = self._db_context.get(prompt)

        if (skip_cached
                or self.encoder.check_relevance(prompt, answer)<= threshold
                or answer is None):

            math_score = self.encoder.check_relevance(prompt,"mathematics")
            bio_score = self.encoder.check_relevance(prompt, "biology")
            code_score = self.encoder.check_relevance(prompt, "programming")

            scores = {
                "mathematics": math_score,
                "biology": bio_score,
                "programming": code_score,
            }
            top_category = max(scores, key=scores.get)

            model = GeneralLLM(top_category)




            (generalized_prompt,default_answer) = generalize_prompt(model,prompt)



            self._db_context.insert(generalized_prompt,default_answer)

            return default_answer, False

        return answer, True














