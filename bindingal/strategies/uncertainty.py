from typing import List

from torch.utils.data import DataLoader

from .base import BaseSelector


# TODO: Implement uncertainty-based sampling strategy
class UncertaintySelector(BaseSelector):
    """
    An uncertainty-based selector for active learning.
    """

    def __init__(self, models):
        self.models = models

    def _select_mc_dropout(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        # Implement MC Dropout uncertainty sampling
        pass

    def _select_ensemble(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        # Implement ensemble-based uncertainty sampling
        pass

    def select(self, unlabeled_dataloader: DataLoader, batch_size: int, strategy: str) -> List[int]:
        if strategy == "mc_dropout":
            return self._select_mc_dropout(unlabeled_dataloader, batch_size)
        elif strategy == "ensemble":
            return self._select_ensemble(unlabeled_dataloader, batch_size)
        else:
            raise ValueError(f"Unsupported uncertainty strategy: {strategy}")
        
