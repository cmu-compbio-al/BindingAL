from torch.utils.data import DataLoader
from typing import List
from .base import BaseSelector

# TODO: Implement diversity-based sampling strategy
class DiversitySelector(BaseSelector):
    """
    Batch mode with diversity-based sampling strategy for active learning.
    """
    def __init__(self, model):
        pass

    def select(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        pass