from app.prompts_classification.fact_or_generative_classifier import FactOrGenerativeClassifier
from .generalize_answer import generalize_prompt
from app.interface.abstractdatabasecontext import DatabaseContext
from app.interface.abstractprompthandler import AbstractPromptHandler
from app.interface.abstractmodel import AbstractModel
from app.relevance_checkers.cross_encoder_relevance_checker import CrossEncoderRelevanceChecker

class PromptHandler(AbstractPromptHandler):

    encoder = CrossEncoderRelevanceChecker()

    def __init__(
            self,
            db_context: DatabaseContext,
            model: AbstractModel,
            fact_classifier: FactOrGenerativeClassifier) -> None:
        self._db_context = db_context
        self.model = model
        self.fact_classifier = fact_classifier

    def generate_answer(self, prompt: str, skip_cached: bool, threshold: object = 0.5) -> tuple[str, bool, str]:
        answer=""

        is_generative = self.fact_classifier.predict(prompt)

        if is_generative:
            print("Not a fact")

        if not skip_cached and not is_generative:
            answer,_ = self._db_context.get(prompt)

        if (skip_cached
                or is_generative
                or self.encoder.check_relevance(prompt, answer) <= threshold
                or answer is None):

            question = f"What is the prompt type of: {prompt}"
            math_score = self.encoder.check_relevance(question, "Prompts that involve abstract reasoning, formal definitions, theorems, proofs, symbolic manipulation, equations, or quantitative problem-solving across areas such as calculus, algebra, geometry, number theory, logic, probability, or discrete mathematics.")
            bio_score = self.encoder.check_relevance(question, "Prompts that concern living organisms, biological processes, cellular or molecular mechanisms, genetics, physiology, ecology, evolution, biochemistry, anatomy, or explanations of how biological systems function at any scale.")
            code_score = self.encoder.check_relevance(question, "Prompts that request code generation, debugging, algorithm design, data structures, API usage, software architecture, system design, development tools, optimization techniques, or any reasoning related to computer programming and software engineering.")

            scores = {
                "math": math_score,
                "bio": bio_score,
                "code": code_score,
            }

            print(scores)

            top_category = max(scores, key=scores.get)

            if scores[top_category] < 0.0161:
                top_category = "General"
                model = self.model
            else:
                model = self.model  # GeneralLLM(top_category)

            (generalized_prompt,default_answer) = generalize_prompt(model,prompt)
            self._db_context.insert(generalized_prompt,default_answer)

            return default_answer, False, top_category

        return answer, True, "None"











