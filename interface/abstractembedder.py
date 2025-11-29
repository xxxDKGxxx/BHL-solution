from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
	@abstractmethod
	def embedd(self, v: str) -> np.ndarray[np.float32]:
		pass