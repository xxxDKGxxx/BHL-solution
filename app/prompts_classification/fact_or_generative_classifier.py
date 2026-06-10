import pickle
import numpy as np
from interface.model_interface import ModelInterface


class FactOrGenerativeClassifier:
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        self.model: ModelInterface = self._import_model('prompts_classification/model/fact_or_generative_model.pkl')
        self.threshold = threshold
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "nvidia/prompt-task-and-complexity-classifier"
        # )
        # self.config = AutoConfig.from_pretrained(
        #     "nvidia/prompt-task-and-complexity-classifier"
        # )
        # self.model = CustomModel(
        #     target_sizes=self.config.target_sizes,
        #     task_type_map=self.config.task_type_map,
        #     weights_map=self.config.weights_map,
        #     divisor_map=self.config.divisor_map,
        # ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")

    def predict(self, prompt: str) -> bool:
        """
        :param prompt:
        :return: True if prompt is a generative, False otherwise
        """

        # encoded_texts = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     add_special_tokens=True,
        #     max_length=512,
        #     padding="max_length",
        #     truncation=True,
        # )
        # result = self.model(encoded_texts)
        # result = not (
        #     result["task_type_1"][0]
        #     in [
        #         "Open QA",
        #         "Closed QA",
        #         # 'Summarization', 'Classification', 'Extraction' # to do zastanowienia, czy chcemy
        #     ]
        # )
        #
        # return result

        proba = self.model.predict_proba(np.array([prompt]))

        return float(proba[0, 0]) >= self.threshold

    def _import_model(self, path: str) -> ModelInterface:
        with open(path, "rb") as f:
            return pickle.load(f)
