import torch.nn
from sentence_transformers import CrossEncoder

from interface.abstractrelevancechecker import AbstractRelevanceChecker


class CrossEncoderRelevanceChecker(AbstractRelevanceChecker):
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            activation_fn=torch.nn.Sigmoid()
        )
    def check_relevance(self, prompt, answer):
        score = self.model.predict([(prompt, answer)])
        # CrossEncoder.predict returns a list/ndarray; return scalar float
        try:
            return float(score[0])
        except Exception:
            return float(score)


if __name__ == "__main__":
    s = CrossEncoderRelevanceChecker()
    print(s.check_relevance("Am i good?", "Yes, you are very good."))