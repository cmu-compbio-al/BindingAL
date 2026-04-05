import argparse
import os
import subprocess
import sys

from typing import Dict

import numpy as np
import ray

from bindingal.embeds.extractor import EmbeddingExtractor


def _install_sagemaker_requirements_if_needed() -> None:
    """Install requirements-sagemaker.txt only when running inside SageMaker.

    This is gated on the SM_TRAINING_ENV environment variable so that
    local runs (python scripts/extract.py ...) are not affected.
    """

    if "SM_TRAINING_ENV" not in os.environ:
        return

    req_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements-sagemaker.txt")
    if not os.path.exists(req_path):
        return


    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])


# Ensure SageMaker-specific requirements are installed when launched via run_sagemaker.py
_install_sagemaker_requirements_if_needed()

class EmbeddingPredictor:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.extractor = EmbeddingExtractor(
            model_name_or_path=model_name_or_path,
            device=device,
        )

    def __call__(
        self,
        batch: Dict[str, np.ndarray],
        sequence_column: str,
        embedding_column: str,
    ):
        # Assume batch contains a single sequence per row for now (batch size = 1)
        # TODO: Batch processing for multiple sequences in the batch (padding needed for variable-length sequences)
        sequence = batch[sequence_column][0]
        emb = self.extractor.extract(sequence)  # torch.Tensor (L, D) or (D,)
        emb_np = emb.float().cpu().numpy()
        return {
            sequence_column: batch[sequence_column],
            embedding_column: [emb_np],
        }


def run_extraction(
    input_path: str,
    output_dir_path: str,
    model_name_or_path: str,
    sequence_column: str = "sequence",
    embedding_column: str = "embedding",
    num_actors: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
):
    ray.init(ignore_reinit_error=True)
    # Support both CSV file input(multiple sequences in one file) and directory of text files (one sequence per file)
    if input_path.endswith(".csv"):
        ds = ray.data.read_csv(input_path)
    elif os.path.isdir(input_path) or input_path.startswith("s3://"):
        ds = ray.data.read_text(input_path)
        if ds.count() == 0:
            raise ValueError(
                f"No text files found in the specified directory: {input_path}"
            )

        sequence_column = "text"
    else:
        raise ValueError(
            f"Unsupported input path: {input_path}. Must be a CSV file or a directory of text files."
        )

    # Parallel inference with ActorPool for each GPU
    predictions = ds.map_batches(
        EmbeddingPredictor,
        compute=ray.data.ActorPoolStrategy(
            min_size=max(1, num_actors), max_size=max(1, num_actors)
        ),
        fn_constructor_kwargs={
            "model_name_or_path": model_name_or_path,
            "device": device,
        },
        fn_kwargs={
            "sequence_column": sequence_column,
            "embedding_column": embedding_column,
        },
        batch_format="numpy",
        batch_size=batch_size,  # TODO: Process one sequence at a time for now (can be increased with padding logic)
        num_gpus=1 if device == "cuda" else 0,
    )
    predictions.write_parquet(output_dir_path)
    ray.shutdown()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a CSV file using a model."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input CSV file containing sequences.",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save the output Parquet files with embeddings.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="esm3-sm-open-v1",
        help="Name of the model or path to local model directory.",
    )
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="sequence",
        help="Name of the column in the input CSV that contains the sequences.",
    )
    parser.add_argument(
        "--embedding_column",
        type=str,
        default="embedding",
        help="Name of the column to store the extracted embeddings in the output Parquet files.",
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=1,
        help="Number of actors to use for parallel inference.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    run_extraction(
        input_path=args.input_path,
        output_dir_path=args.output_dir_path,
        model_name_or_path=args.model_name_or_path,
        sequence_column=args.sequence_column,
        embedding_column=args.embedding_column,
        num_actors=args.num_actors,
        batch_size=args.batch_size,
        device=args.device,
    )
