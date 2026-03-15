from torch.utils.data import DataLoader
from typing import List
from .base import BaseSelector

# TODO: Implement density-based sampling strategy
class DensitySelector(BaseSelector):
    """
    A density-based selector for active learning.
    """
    def __init__(self, model):
        pass

    def select(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        pass