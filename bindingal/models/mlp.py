import torch

class MLPBinder(torch.nn.Module):
    """
    A simple multi-layer perceptron (MLP) model for binding affinity prediction.
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query_embeddings: torch.Tensor, key_value_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the binding affinity prediction from the query and key-value embeddings.

        Note:
            The query embeddings typically represent one modality (e.g., protein), while the key-value embeddings represent another modality (e.g., ligand).
            The MLP will learn to combine these embeddings to predict the binding affinity.
        
        Args:
            query_embeddings (torch.Tensor): The query embeddings of shape (batch_size, seq_len, embed_dim).
            key_value_embeddings (torch.Tensor): The key-value embeddings of shape (batch_size, seq_len, embed_dim).

        Returns:
            output (torch.Tensor): The predicted binding affinity of shape (batch_size, output_dim).
        """
        # Concatenate the query and key-value embeddings along the feature dimension
        combined_embeddings = torch.cat((query_embeddings, key_value_embeddings), dim=-1)
        x = self.fc1(combined_embeddings)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)

        return output
