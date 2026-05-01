"""3073-d antibody–antigen interaction vector from concatenated ESM pools (1536 each chain)."""
import numpy as np


def build_interaction_features(x_all: np.ndarray, emb_dim: int = 1536) -> np.ndarray:
    # x_all is [H|L|Ag] mean-pools, 4608 total
    x_heavy = x_all[:, :emb_dim]
    x_light = x_all[:, emb_dim : 2 * emb_dim]
    x_antigen = x_all[:, 2 * emb_dim :]

    # single-chain antibodies: light block is all zeros in our cache
    has_light = np.any(x_light != 0, axis=1)
    x_antibody = np.where(has_light[:, None], (x_heavy + x_light) / 2.0, x_heavy)

    hadamard = x_antibody * x_antigen
    abs_diff = np.abs(x_antibody - x_antigen)
    ab_norm = np.linalg.norm(x_antibody, axis=1, keepdims=True) + 1e-10
    ag_norm = np.linalg.norm(x_antigen, axis=1, keepdims=True) + 1e-10
    cosine = np.sum(x_antibody * x_antigen, axis=1, keepdims=True) / (ab_norm * ag_norm)
    return np.hstack([hadamard, abs_diff, cosine])
