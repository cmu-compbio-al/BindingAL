from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseSelector
from utils.utils import pad_embeddings


class UncertaintySelector(BaseSelector):
    """
    An uncertainty-based selector for active learning.
    """

    def __init__(self, models, n_mc_samples: int = 10):
        self.models = models
        self.n_mc_samples = n_mc_samples

    def _select_mc_dropout(self, dataset, batch_size: int, infer_batch_size: int = 24) -> list:
        model = self.models
        device = next(model.parameters()).device

        # Free GPU memory occupied by the training process before running inference
        torch.cuda.empty_cache()

        # Set model to eval mode to freeze BatchNorm, then re-enable Dropout layers
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

        all_probs = []

        for _ in range(self.n_mc_samples):
            run_probs = []
            with torch.no_grad():
                for batch in dataset.iter_batches(batch_size=infer_batch_size, batch_format="numpy"):
                    heavy_padded, heavy_mask = pad_embeddings(batch["heavy_embedding"])
                    light_padded, light_mask = pad_embeddings(batch["light_embedding"])
                    antigen_padded, antigen_mask = pad_embeddings(batch["antigen_embedding"])

                    heavy_padded, heavy_mask = heavy_padded.to(device), heavy_mask.to(device)
                    light_padded, light_mask = light_padded.to(device), light_mask.to(device)
                    antigen_padded, antigen_mask = antigen_padded.to(device), antigen_mask.to(device)

                    antibody_embeddings = torch.cat([heavy_padded, light_padded], dim=1)
                    antibody_mask = torch.cat([heavy_mask, light_mask], dim=1)

                    outputs = model(
                        ab_embeddings=antibody_embeddings,
                        ag_embeddings=antigen_padded,
                        ag_mask=antigen_mask,
                        ab_mask=antibody_mask,
                    )
                    probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
                    run_probs.append(probs)
            all_probs.append(np.concatenate(run_probs))

        # Stack into (N, n_mc_samples) and compute variance as uncertainty score
        all_probs = np.stack(all_probs, axis=1)
        uncertainties = all_probs.var(axis=1)

        # Return the most uncertain records
        sorted_indices = np.argsort(uncertainties)[::-1]
        selected_indices = sorted_indices[:batch_size].tolist()

        all_records = dataset.take_all()
        return [all_records[i] for i in selected_indices]

    def _select_ensemble(self, dataset: DataLoader, batch_size: int) -> List[int]:
        # TODO: Implement ensemble-based uncertainty sampling
        pass

    def select(self, dataset: DataLoader, batch_size: int, strategy: str) -> List[int]:
        if strategy == "mc_dropout":
            return self._select_mc_dropout(dataset, batch_size)
        elif strategy == "ensemble":
            return self._select_ensemble(dataset, batch_size)
        else:
            raise ValueError(f"Unsupported uncertainty strategy: {strategy}")
