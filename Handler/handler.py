from abc import ABC, abstractmethod


class PromptHandler(ABC):
    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        pass

