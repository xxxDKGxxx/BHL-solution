from abc import ABC, abstractmethod


class AbstractModel(ABC):
	@abstractmethod
	def generate_answer(self, prompt: str) -> str:
		pass


