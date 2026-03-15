from typing import List

from torch.utils.data import DataLoader

from .base import BaseSelector


# TODO: Implement query-by-committee strategy
class QBCSelector(BaseSelector):
    """
    A query-by-committee (QBC) selector for active learning.
    """

    def __init__(self, committee_models):
        pass

    def select(self, unlabeled_dataloader: DataLoader, batch_size: int) -> List[int]:
        pass
