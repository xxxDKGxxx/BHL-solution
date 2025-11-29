from abc import ABC, abstractmethod


class Embedder(ABC):
	@abstractmethod
	def embedd(self, v: str) -> list[str]:
		pass

