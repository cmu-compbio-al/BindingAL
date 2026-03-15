from typing import List

from torch.utils.data import DataLoader

from .base import BaseSelector


# TODO: Implement uncertainty-based sampling strategy
class UncertaintySelector(BaseSelector):
    """
    An uncertainty-based selector for active learning.
    """

    def __init__(self, model):
        pass

    def select(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        pass
