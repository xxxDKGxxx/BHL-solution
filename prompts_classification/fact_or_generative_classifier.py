from transformers import AutoTokenizer, AutoConfig

from prompts_classification.nvidia_model import CustomModel


class FactOrGenerativeClassifier:
    def __init__(self, ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/prompt-task-and-complexity-classifier"
        )
        self.config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
        self.model = CustomModel(
            target_sizes=self.config.target_sizes,
            task_type_map=self.config.task_type_map,
            weights_map=self.config.weights_map,
            divisor_map=self.config.divisor_map,
        ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")


    def predict(self, prompt: str) -> bool:
        """
        :param prompt:
        :return: True if promt is a generative, False otherwise
        """

        encoded_texts = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        result = self.model(encoded_texts)
        result = not (result['task_type_1'][0] in ['Open QA', 'Closed QA',
                                                   #'Summarization', 'Classification', 'Extraction' # to do zastanowienia, czy chcemy
                                                   ])

        return result
