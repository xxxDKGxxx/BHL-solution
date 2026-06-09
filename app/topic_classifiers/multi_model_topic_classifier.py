import sys
import numpy as np
import pickle
from interface.abstracttopicclassifier import AbstractTopicClassifier
from interface.model_interface import ModelInterface


class MultiModelTopicClassifier(AbstractTopicClassifier):
    def __init__(self):
        self.math_model = self._import_model("topic_classifiers/models/math_model.pkl")
        self.bio_model = self._import_model("topic_classifiers/models/bio_model.pkl")
        self.code_model = self._import_model("topic_classifiers/models/code_model.pkl")

    def classify_topic(self, prompt: str) -> str:
        results: dict[str, float] = {}
        results["math"] = float(self.math_model.predict_proba(np.array([prompt]))[0, 1])
        results["bio"] = float(self.bio_model.predict_proba(np.array([prompt]))[0, 1])
        results["code"] = float(self.code_model.predict_proba(np.array([prompt]))[0, 1])

        max_class = max(results, key=results.get)
        max_score = results[max_class]

        if max_score < 0.9:
            return "General"

        print(results)

        return max_class

    def _import_model(self, path: str) -> ModelInterface:
        with open(path, "rb") as f:
            model = pickle.load(f)

        return model
