import os
import time
import tempfile
import argparse
import json

import torch
import ray
import psutil

import boto3
import numpy as np
import pandas as pd

from botocore.exceptions import NoCredentialsError
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from ray.train.torch import TorchTrainer, prepare_model
from ray.train import ScalingConfig, get_dataset_shard, report, Checkpoint, RunConfig
from ray.data import ActorPoolStrategy
from ray.train import get_context

from bindingal.models import get_model
from bindingal.strategies.uncertainty import UncertaintySelector

from utils.distributed import _get_sagemaker_hosts, _get_current_host, _get_num_gpus, init_ray_cluster

class EmbeddingMapper:
    def __init__(self, embed_dict_ref):
        # Receive an ObjectRef to avoid embedding the large dict in the UDF closure.
        self.embed_dict = ray.get(embed_dict_ref)

    def __call__(self, batch: dict) -> dict:
        heavy_seqs = batch["heavy_seq"]
        light_seqs = batch["light_seq"]
        antigen_seqs = batch["antigen_seq"]

        heavy_embs, light_embs, antigen_embs = [], [], []
        keep_mask = []

        # Look up embeddings and mark rows with missing embeddings to drop
        for h, l, a in zip(heavy_seqs, light_seqs, antigen_seqs):
            heavy_emb = self.embed_dict.get(h)
            antigen_emb = self.embed_dict.get(a)

            # Handle empty light_seq: use zero-length embedding instead of looking up
            if l == "":
                light_emb = np.zeros((0, 1536), dtype=np.float32)
            else:
                light_emb = self.embed_dict.get(l)

            if heavy_emb is None or antigen_emb is None:
                keep_mask.append(False)
                # append placeholders to keep array lengths aligned (will be filtered out)
                heavy_embs.append(None)
                light_embs.append(None)
                antigen_embs.append(None)
            else:
                keep_mask.append(True)
                heavy_embs.append(heavy_emb)
                light_embs.append(light_emb)
                antigen_embs.append(antigen_emb)

        # Build numpy arrays and filter batch rows with missing embeddings
        batch["heavy_embedding"] = np.array(heavy_embs, dtype=object)
        batch["light_embedding"] = np.array(light_embs, dtype=object)
        batch["antigen_embedding"] = np.array(antigen_embs, dtype=object)

        if not all(keep_mask):
            mask = np.array(keep_mask, dtype=bool)
            for k, v in list(batch.items()):
                try:
                    arr = np.array(v, dtype=object)
                except Exception:
                    continue
                batch[k] = arr[mask]

        return batch

def pad_embeddings(embeddings):
    if len(embeddings) == 0:
        return torch.zeros((0, 0, 1536), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.float32)

    lengths = [len(emb) for emb in embeddings]
    max_len = max(lengths)

    if max_len == 0:
        batch_size = len(embeddings)
        return torch.zeros((batch_size, 0, 1536), dtype=torch.float32), torch.zeros((batch_size, 0), dtype=torch.float32)

    # reshape(-1, 1536) ensures empty arrays (0,) become (0, 1536) for pad_sequence compatibility
    tensor_list = [torch.tensor(emb, dtype=torch.float32).reshape(-1, 1536) for emb in embeddings]
    padded_seqs = pad_sequence(tensor_list, batch_first=True)
    lengths_tensor = torch.tensor(lengths)
    mask = torch.arange(max_len)[None, :] < lengths_tensor[:, None]
    return padded_seqs, mask.float()

def run_forward(model: nn.Module, batch: dict, criterion: nn.Module):
    """Shared forward pass logic for train and val to avoid code duplication.
    
    Args:
        model (nn.Module): The model to evaluate.
        batch (dict): The input batch of data.
        criterion (nn.Module): The loss function.

    Returns:
        loss (torch.Tensor): The computed loss for the batch.
        correct (int): The number of correct predictions in the batch.
        n (int): The total number of samples in the batch.
    """
    device = next(model.parameters()).device
    y = torch.tensor(batch["label"], dtype=torch.float32).to(device)
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
    loss = criterion(outputs.squeeze(-1), y)
    preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).to(torch.int64)
    correct = (preds == y.to(torch.int64)).sum().item()
    return loss, correct, y.shape[0]

def train_loop_per_worker(config):
    train_data_shard = get_dataset_shard("train")
    val_data_shard = get_dataset_shard("val")

    # Retrieve the model from the config
    model = get_model("cross_attn", embed_dim=1536, num_heads=8)
    model.load_state_dict(config["model_state_dict"])
    model = prepare_model(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    rank = get_context().get_world_rank()

    for epoch in range(config["epochs"]):
        # Train on the training set
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for batch in train_data_shard.iter_batches(batch_size=config["batch_size"], batch_format="numpy"):
            loss, _, n = run_forward(model, batch, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * n
            train_total += n

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0

        # Evaluate on the validation set
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_data_shard.iter_batches(batch_size=config["batch_size"], batch_format="numpy"):
                loss, correct, n = run_forward(model, batch, criterion)
                val_loss_sum += loss.item() * n
                val_correct += correct
                val_total += n

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}

        # Only rank 0 prints metrics
        if rank == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            with tempfile.TemporaryDirectory() as temp_ckpt_dir:
                torch.save(model.module.state_dict(), os.path.join(temp_ckpt_dir, "model.pt"))
                checkpoint = Checkpoint.from_directory(temp_ckpt_dir)
                report(metrics, checkpoint=checkpoint)
        else:
            report(metrics)


def save_model(state_dict, save_path):
    """
    Save the model state dictionary to an S3 path or local path.

    Args:
        state_dict: The model state dictionary to save.
        save_path: The S3 or local path to save the model.

    Returns:
        None
    """
    if save_path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket_name = save_path.split("/")[2]
        s3_key = "/".join(save_path.split("/")[3:])
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "model.pth")
            torch.save(state_dict, local_path)
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"Model successfully uploaded to {save_path}")
            except NoCredentialsError:
                print("S3 upload failed: No AWS credentials found.")
    else:
        torch.save(state_dict, save_path)
        print(f"Model successfully saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with query strategy-based active learning.")

    # General arguments
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to the training data (local or S3).")
    parser.add_argument("--embed_data_path", type=str, required=True,
                        help="Path to the embedding data (local or S3).")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training (default: 24).")
    parser.add_argument("--query_size", type=int, default=96, help="Number of samples to query in each active learning iteration (default: 96).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--finetune_performance_threshold", type=float, default=0.75,
                        help="Performance threshold for training to stop. (default: 0.75)")
    parser.add_argument("--report_file_path", type=str, default="report.json", help="Path to save the training report (optional).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for distributed training.")
    parser.add_argument("--s3_artifact_path", type=str, default=None,
                        help="S3 URI for Ray Train checkpoint storage shared across all nodes (e.g. s3://bucket/artifacts).")
    parser.add_argument("--strategy", type=str, choices=["passive", "mc_dropout", "ensemble"], default="passive",
                        help="Query strategy for active learning (default: passive).")

    return parser.parse_args()

def load_and_prepare_data(train_data_path, embed_data_path):
    """Load and prepare datasets, including embedding mapping."""
    train_data = ray.data.read_csv(train_data_path)
    embed_data = ray.data.read_parquet(embed_data_path)

    embed_records = embed_data.take_all()
    embed_dict = {record["text"]: record["embedding"].reshape(-1, 1536) for record in embed_records}
    embed_dict_ref = ray.put(embed_dict)

    def map_embeddings(ds):
        return ds.map_batches(
            EmbeddingMapper,
            fn_constructor_kwargs={"embed_dict_ref": embed_dict_ref},
            batch_format="numpy",
            compute=ActorPoolStrategy(min_size=1, max_size=1),
        )

    train_ds = map_embeddings(train_data)

    return train_ds

def write_report(report_file_path, report_data):
    """Write the training report to a file (supports both local and S3 paths)."""
    if report_file_path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket_name = report_file_path.split("/")[2]
        s3_key = "/".join(report_file_path.split("/")[3:])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "temp_report.json")
            with open(local_path, "w") as f:
                json.dump(report_data, f, indent=4)
            
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"Report successfully uploaded to {report_file_path}")
            except NoCredentialsError:
                print("S3 upload failed for report: No AWS credentials found.")
            except Exception as e:
                print(f"S3 upload failed for report: {e}")
                
    else:
        with open(report_file_path, "w") as f:
            json.dump(report_data, f, indent=4)


def train_model(config, train_ds, val_ds):
    """Train the model using query strategy-based active learning."""
    strategy = config.get("strategy", "passive")  # Default to passive learning
    batch_size = config["batch_size"]
    query_size = config.get("query_size", 24)  # Default query size if not provided
    epochs = config["epochs"]
    finetune_performance_threshold = config["finetune_performance_threshold"]
    report_file_path = config["report_file_path"]

    # Initialize model and criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("cross_attn", embed_dim=1536, num_heads=8).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # ScalingConfig: same calculation as before
    hosts = _get_sagemaker_hosts()
    num_nodes = max(len(hosts), 1)
    num_gpus_per_node = _get_num_gpus()
    total_workers = num_nodes * max(num_gpus_per_node, 1)
    use_gpu = num_gpus_per_node > 0
    scaling_cfg = ScalingConfig(
        num_workers=total_workers,
        use_gpu=use_gpu,
        resources_per_worker={
            "CPU": max(psutil.cpu_count(logical=False) // max(num_gpus_per_node, 1), 1),
            **({"GPU": 1} if use_gpu else {}),
        },
    )

    # Initialize performance tracking
    current_performance = 0.0
    selected_data = train_ds.take(query_size)  # Random initial batch
    report_data = []

    # Train the model on the initial batch
    fine_tune_ds = ray.data.from_items(selected_data)
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for batch in fine_tune_ds.iter_batches(batch_size=batch_size, batch_format="numpy"):
            loss, _, n = run_forward(model, batch, criterion)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * n
            train_total += n

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
        print(f"Epoch {epoch}: Initial Training Loss={train_loss:.4f}")

    batch_num = 0
    while current_performance < finetune_performance_threshold:
        print(f"Current performance: {current_performance:.4f}, Threshold: {finetune_performance_threshold:.4f}")

        # Select batch based on strategy
        batch_num += 1
        if strategy == "passive":
            batch = train_ds.take(query_size)
        elif strategy == "mc_dropout":
            selector = UncertaintySelector(model, method="mc_dropout")
            batch = selector.select_batch(train_ds, query_size)
        elif strategy == "ensemble":
            selector = UncertaintySelector(model, method="ensemble")
            batch = selector.select_batch(train_ds, query_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        selected_data.extend(batch)
        batch_full_seqs = {record["heavy_seq"] + record["light_seq"] + record["antigen_seq"] for record in batch}
        train_ds = train_ds.filter(lambda r: (r["heavy_seq"] + r["light_seq"] + r["antigen_seq"]) not in batch_full_seqs)

        fine_tune_ds = ray.data.from_items(selected_data)

        # Add the model to the config
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        config["model_state_dict"] = cpu_state_dict

        # Fine-tune the model using DDP
        trainer = TorchTrainer(
            train_loop_per_worker,
            train_loop_config=config,
            datasets={"train": fine_tune_ds, "val": val_ds},
            run_config=RunConfig(
                storage_path=config.get("s3_artifact_path"),
                name=f"train_iter_{batch_num}",
            ),
            scaling_config=scaling_cfg,
        )
        result = trainer.fit()

        if result.checkpoint:
            with result.checkpoint.as_directory() as ckpt_dir:
                ckpt_state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
            cleaned_state = {k.replace("module.", ""): v for k, v in ckpt_state.items()}
            model.load_state_dict(cleaned_state)

        # Evaluate performance on validation set
        model.eval()
        val_loss_sum, val_total, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for tb in val_ds.iter_batches(batch_size=batch_size, batch_format="numpy"):
                loss_t, correct_t, n_t = run_forward(model, tb, criterion)
                val_loss_sum += loss_t.item() * n_t
                val_correct += correct_t
                val_total += n_t

        current_performance = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        print(f"Batch: {batch_num}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {current_performance:.4f}")

        # Log progress to the report file
        report_entry = {
            "current_performance": current_performance,
            "batch": batch_num,
            "strategy": strategy,
            "val_loss": val_loss,
        }
        report_data.append(report_entry)

        # Prevent infinite loop in case of issues
        if batch_num >= 50:
            print("Stopping training early to prevent infinite loop.")
            print(f"Current performance after {batch_num} batches: {current_performance:.4f}")
            break

    # Save the final model
    write_report(report_file_path, report_data)
    save_model(model.state_dict(), config["s3_artifact_path"] + "/final_model.pth")
    print("Training completed. Final performance met the threshold.")
    return


if __name__ == "__main__":
    args = parse_args()

    init_ray_cluster(num_workers_per_node=args.num_workers)

    hosts = _get_sagemaker_hosts()
    current_host = _get_current_host()
    head_host = hosts[0] if hosts else current_host

    if current_host != head_host:
        print(f"[{current_host}] Worker node ready. Waiting for tasks from head node.")
        while True:
            time.sleep(30)

    # Load and prepare data
    train_ds = load_and_prepare_data(args.train_data_path, args.embed_data_path)
    
    # NOTE: Uncomment the block below to re-enable spike/COVID filtering.
    # covid_keywords = ['sars-cov-2', 'sars-cov2', 'covid-19', 'spike']
    # def _is_spike(record):
    #     name = record.get("antigen_name")
    #     if name is None:
    #         return False
    #     try:
    #         return str(name).lower().startswith("spike")
    #     except Exception:
    #         return False
    # non_spike_ds = train_ds.filter(lambda r: not _is_spike(r))
    # spike_ds = train_ds.filter(_is_spike)
    # train_ds = non_spike_ds

    train_split, val_split = train_ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    print(f"Train: {train_split.count()}, Val: {val_split.count()}")

    train_model(vars(args), train_split, val_split)
