from abc import ABC, abstractmethod

class DatabaseContext(ABC):
    @abstractmethod
    def get(self,prompt: str) ->tuple[str,float]:
        pass

    @abstractmethod
    def insert(self, prompt: str, answer: str) -> bool:
        pass




