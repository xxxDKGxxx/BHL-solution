import os

import pandas as pd

from app.database.database import Database
from app.database.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.llms.gemini_llm import GeminiLLM

from app.prompts_classification.fact_or_generative_classifier import FactOrGenerativeClassifier
from app.interface.abstractdatabasecontext import DatabaseContext
from app.interface.abstractprompthandler import AbstractPromptHandler
from app.interface.abstractmodel import AbstractModel
from app.relevance_checkers.cross_encoder_relevance_checker import CrossEncoderRelevanceChecker
from app.handler.generalize_answer import generalize_prompt


class SingleModelWithoutFactClassificationPromptHandler(AbstractPromptHandler):
    encoder = CrossEncoderRelevanceChecker()

    def __init__(
            self,
            db_context: DatabaseContext,
            model: AbstractModel) -> None:
        self._db_context = db_context
        self.model = model

    def generate_answer(self, prompt: str, skip_cached: bool, threshold: object = 0.5) -> tuple[str, bool, str]:
        answer = ""

        if not skip_cached:
            answer, _ = self._db_context.get(prompt)

        if (skip_cached
                or self.encoder.check_relevance(prompt, answer) <= threshold
                or answer is None):
            model = self.model

            (generalized_prompt, default_answer) = generalize_prompt(model, prompt)
            self._db_context.insert(generalized_prompt, default_answer)

            return default_answer, False, "None"

        return answer, True, "None"

class SingleModelPromptHandler(AbstractPromptHandler):
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

            model = self.model

            (generalized_prompt,default_answer) = generalize_prompt(model,prompt)
            self._db_context.insert(generalized_prompt,default_answer)

            return default_answer, False, "None"

        return answer, True, "None"

class LlmCacheBuildingTester:
    def __init__(self, questions: list[str], similair_questions: list[str], name=""):
        self.questions = questions
        self.similair_questions = similair_questions
        self.db = Database(SentenceTransformerEmbedder())
        self.handler = SingleModelPromptHandler(
            self.db,
            GeminiLLM(),
            FactOrGenerativeClassifier()
        )
        self.name = name
        self.number = 1
    def run(self):
        if not os.path.exists("results"):
            os.mkdir("results")

        if os.path.exists(f"results/question_answers_{self.name}.csv"):
            df = pd.read_csv(f"results/question_answers_{self.name}.csv")
            for _, record in df.iterrows():
                self.db.insert(record["question"], record["answer"])
        else:
            question_answer_pairs = []
            for question in self.questions:
                print(question)
                question_answer_pairs.append((
                    question,
                    self.handler.generate_answer(question, False)
                ))
                print(f"Answer: {question_answer_pairs[-1][1]}")

            df = pd.DataFrame(question_answer_pairs, columns=["question", "answer"])
            df.to_csv(f"results/question_answers_{self.name}_{self.number}.csv", index=False)

        while os.path.exists(f"results/question_is_cached_{self.name}_{self.number}.csv"):
            self.number += 1

        question_is_cached_pairs = []
        for similair_question in self.similair_questions:
            print(f"Similair question: {similair_question}")
            _, is_cached, _ = self.handler.generate_answer(similair_question, False)
            question_is_cached_pairs.append((similair_question, is_cached))
            print(f"Is cached: {is_cached}")

        df = pd.DataFrame(question_is_cached_pairs, columns=["question", "is_cached"])
        df.to_csv(f"results/question_is_cached_{self.name}_{self.number}.csv", index=False)



def summarise_batch_reports(name="") -> pd.DataFrame:
    number = 1

    df = pd.DataFrame(columns=["question", "is_cached"])

    while os.path.exists(f"results/question_is_cached_{name}_{number}.csv"):
        df = pd.concat([df, pd.read_csv(f"results/question_is_cached_{name}_{number}.csv")], axis=0)
        number += 1

    return df