from llms.biogpt_llm import BioGPTForBiology
from llms.deepseekcoder_llm import DeepSeekCoderForMathematics
from llms.starcoder_llm import StarCoderForIT
from llms.gemini_llm import GeminiLLM
from interface.abstractmodel import AbstractModel

class GeneralLLM(AbstractModel):
    def __init__(self, topic: str):
        if topic == "bio":
            self.model = BioGPTForBiology()
        elif topic == "math":
            self.model = DeepSeekCoderForMathematics()
        elif topic == "code":
            self.model = StarCoderForIT()
    def generate_answer(self, prompt: str) -> str:
        return self.model.generate_answer(prompt)
