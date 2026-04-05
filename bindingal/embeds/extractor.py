
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from transformers import AutoModel, AutoTokenizer


class EmbeddingExtractor:
    """
    A versatile class for extracting embeddings from various protein language models.
    Supports standard Hugging Face models (ESM-2, BERT, etc.) and ESM-3.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        """
        Initializes the extractor by detecting the model type and loading appropriately.
        """
        self.device = device
        self.model_name = model_name_or_path.lower()

        # Check if the model is ESM-3
        if "esm3" in self.model_name:
            print(f"Loading ESM-3 model: {model_name_or_path}")
            self.model = ESM3.from_pretrained(model_name_or_path).to(self.device)
            self.is_esm3 = True
        else:
            # Standard Hugging Face loading for ESM-2 or others
            print(f"Loading Standard HF model: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
            self.is_esm3 = False

        self.model.eval()

    @torch.no_grad()
    def extract(self, sequence: str) -> torch.Tensor:
        """
        Extracts embedding for a single input sequence.
        """
        if self.is_esm3:
            return self._extract_esm3(sequence)
        else:
            return self._extract_standard(sequence)

    def _extract_standard(self, sequence: str) -> torch.Tensor:
        """
        Extraction logic for standard Hugging Face models (e.g., ESM-2) for a single sequence.
        """
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)

        # Shape: (hidden_size,)
        return outputs.last_hidden_state[0, 0, :].cpu()

    def _extract_esm3(self, sequence: str) -> torch.Tensor:
        """
        Extraction logic for ESM-3 using its specific SDK for a single sequence.

        Returns per-residue embeddings with shape (L, hidden_size).
        """
        # Construct a protein object for ESM-3
        protein = ESMProtein(sequence=sequence)

        # Encode and forward pass
        protein_tensor = self.model.encode(protein)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.model(sequence_tokens=protein_tensor.sequence.unsqueeze(0))

        res_embeddings = output.embeddings.squeeze(0)  # (L, D)
        return res_embeddings.cpu()
