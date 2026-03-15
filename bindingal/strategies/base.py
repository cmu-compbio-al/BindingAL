from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import List

class BaseSelector(ABC):
    """
    Abstract base class for active learning strategies. All active learning strategies should inherit from this class
    and implement the select method.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def select(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        pass