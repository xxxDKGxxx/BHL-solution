from abc import ABC, abstractmethod


class AbstractPromptHandler(ABC):
    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        pass

