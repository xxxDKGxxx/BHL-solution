from abc import ABC

from app.interface.abstractmodel import AbstractModel


class DefaultModel(AbstractModel, ABC):


    def generate_answer(self, prompt: str) -> str:
        return 0.5
