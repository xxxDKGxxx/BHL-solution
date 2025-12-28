from abc import abstractmethod, ABC

class AbstractRelevanceChecker(ABC):
	@abstractmethod
	def check_relevance(self, prompt, answer):
		pass
