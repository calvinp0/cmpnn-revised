from abc import ABC, abstractmethod
from typing import Tuple, List

class BaseSplitter(ABC):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def split(self, dataset, 
              train_frac: float = 0.8, 
              val_frac: float = 0.1, 
              test_frac: float = 0.1
             ) -> Tuple[List, List, List]:
        """
        Splits a dataset into train, val, test.
        Must return three lists of dataset elements.
        """
        pass
