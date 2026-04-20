import torch
import torch.nn as nn
from typing import Optional

class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling.
 
    Learns a per-position scalar score so that CDR residues carry more weight
    than framework residues, instead of being averaged away by mean pooling.
    """
 
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)
 
    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input embeddings of shape (B, L, D)
            valid_mask: a boolean mask of shape (B, L) indicating valid tokens

        Returns:
            pooled: attention-weighted pooled embedding of shape (B, D)
        """
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = (weights.unsqueeze(-1) * x).sum(dim=1)

        return out

 
class CrossAttentionLayer(nn.Module):
    """One bidirectional cross-attention layer.
 
    Note:
      ab_out = Antibody attends Antigen   (Ab -> Ag)
      ag_out = Antigen  attends Antibody  (Ag -> Ab)
 
    The two outputs are merged back to embed_dim so the layer can be stacked.
    """
 
    def __init__(self, embed_dim: int = 1536, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
 
        self.ab_to_ag = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ag_to_ab = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.pool = AttentionPooling(embed_dim)

        # Merge the two directions back to embed_dim
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
 
        self.norm_ab = nn.LayerNorm(embed_dim)
        self.norm_ag = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        ab: torch.Tensor,
        ag: torch.Tensor,
        ab_pad_mask: torch.Tensor,
        ag_pad_mask: torch.Tensor,
    ):
        # Ab attends Ag  (antibody queries antigen context)
        ab_out, _ = self.ab_to_ag(ab, ag, ag, key_padding_mask=ag_pad_mask)
        ab_out = self.norm_ab(ab + self.dropout(ab_out))
 
        # Ag attends Ab  (antigen queries antibody context)
        ag_out, _ = self.ag_to_ab(ag, ab, ab, key_padding_mask=ab_pad_mask)
        ag_out = self.norm_ag(ag + self.dropout(ag_out))
 
        # Merge outputs by concatenating the attention-pooled antigen summary to each antibody token, then projecting back to embed_dim
        ag_valid = (~ag_pad_mask).float() # (B, L_ag)
        ag_summary = self.pool(ag_out, ag_valid) # (B, D)
        ag_summary = ag_summary.unsqueeze(1).expand_as(ab_out) # (B, L_ab, D)
 
        merged = self.merge(torch.cat([ab_out, ag_summary], dim=-1))  # (B, L_ab, D)
        
        return merged, ag_out
 
 
class CrossAttentionBinder(nn.Module):
    """
    Bidirectional cross-attention classifier for antigen-antibody
    binding prediction.
 
    Architecture
    ------------
    1. Bidirectional cross-attention layer
       Ab attends Ag AND Ag attends Ab, both with residual connections and LayerNorm.
       Ag summary is pooled and concatenated back to each Ab token to preserve global context.
    2. Attention-weighted pooling over the antibody sequence
       (preserves CDR signal instead of diluting it with mean pooling).
    3. MLP head -> logit.
 
    Args:
        embed_dim:  Per-token embedding dimension (1536 for ESM3).
        num_heads:  Number of attention heads.
        dropout:    Dropout probability applied inside attention and MLP.
    """
 
    def __init__(
        self,
        embed_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.crossattn_layer = CrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.pool = AttentionPooling(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
 
    def forward(
        self,
        ab_embeddings: torch.Tensor,
        ag_embeddings: torch.Tensor,
        ab_mask: Optional[torch.Tensor] = None,
        ag_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the cross-attention binder.

        Args:
            ab_embeddings: Antibody embeddings of shape (B, L_ab, D)
            ag_embeddings: Antigen embeddings of shape (B, L_ag, D)
            ab_mask: (B, L_ab) — True = valid token (optional)
            ag_mask: (B, L_ag) — True = valid token (optional)

        """
        B = ab_embeddings.size(0)

        # Convert "valid" masks to padding masks expected by MultiheadAttention
        # (True = position should be IGNORED)
        if ab_mask is not None:
            ab_pad_mask = ~ab_mask.bool()
        else:
            ab_pad_mask = torch.zeros(B, ab_embeddings.size(1), dtype=torch.bool, device=ab_embeddings.device)

        if ag_mask is not None:
            ag_pad_mask = ~ag_mask.bool()
        else:
            ag_pad_mask = torch.zeros(B, ag_embeddings.size(1), dtype=torch.bool, device=ag_embeddings.device)

        # Stacked bidirectional cross-attention
        ab_embeddings, ag_embeddings = self.crossattn_layer(ab_embeddings, ag_embeddings, ab_pad_mask, ag_pad_mask)

        # Attention-weighted pooling over antibody sequence
        if ab_mask is not None:
            valid_ab = ab_mask.bool()
        else:
            valid_ab = torch.ones(B, ab_embeddings.size(1), dtype=torch.bool, device=ab_embeddings.device)
        pooled = self.pool(ab_embeddings, valid_ab)
        
        return self.classifier(pooled)
 
