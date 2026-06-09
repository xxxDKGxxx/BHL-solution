from abc import ABC, abstractmethod


class AbstractTopicClassifier(ABC):
    @abstractmethod
    def classify_topic(self, prompt: str) -> str:
        pass
