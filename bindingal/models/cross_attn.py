import torch
import torch.nn as nn

class CrossAttentionBinder(nn.Module):
    """
    A cross-attention module for binding two sets of embeddings.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        pass

    def forward(self, query_embeddings: torch.Tensor, key_value_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-attention between the query embeddings and the key-value embeddings.

        Note:
            For Active Learning, you may want to return the hidden states as well.

        Args:
            query_embeddings (torch.Tensor): The query embeddings of shape (seq_len_q, batch_size, embed_dim).
            key_value_embeddings (torch.Tensor): The key-value embeddings of shape (seq_len_kv, batch_size, embed_dim).
        
        Returns:
            output (torch.Tensor): The output of the cross-attention mechanism of shape (seq_len_q, batch_size, embed_dim).
        """
        pass