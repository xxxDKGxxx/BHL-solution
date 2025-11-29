from abc import ABC, abstractmethod


class Model(ABC):
	@abstractmethod
	def generate_answer(self, prompt: str) -> str:
		pass


