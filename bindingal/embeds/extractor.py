import torch
from transformers import AutoTokenizer, AutoModel
from typing import List

class EmbeddingExtractor:
    """
    A class for extracting embeddings from a pre-trained language model.
    It can load model from huggingface or from a local path, and provides a method to extract embeddings for a given input text.
    """
    def __init__(self, model_name_or_path: str, device: str = 'cpu'):
        """
        Initializes the embedding extractor by loading the model and tokenizer.

        Args:
            model_name_or_path (str): The name of the pre-trained model or the path to the local model directory.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract(self, sequences: List[str]) -> torch.Tensor:
        """
        Extracts embeddings for a list of input sequences.

        Args:
            sequences (List[str]): A list of input text sequences.
        
        Returns:
            embeddings (torch.Tensor): A tensor containing the extracted embeddings for the input sequences.
        """
        pass