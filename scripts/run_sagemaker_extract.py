import argparse
import os

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SageMaker ESM3 embedding extraction job (generalized)"
    )

    parser.add_argument(
        "--arn_role",
        type=str,
        required=True,
        help="SageMaker execution role ARN",
    )
    parser.add_argument(
        "--s3_input_path",
        type=str,
        required=True,
        help="S3 path to input data (text files dir)",
    )
    parser.add_argument(
        "--s3_embedding_output_path",
        type=str,
        required=True,
        help="S3 path to store embeddings output (Parquet directory)",
    )
    parser.add_argument(
        "--s3_artifact_path",
        type=str,
        required=True,
        help="S3 path to store SageMaker job artifacts",
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        required=True,
        help="SageMaker instance type (e.g., ml.g5.12xlarge)",
    )

    # Optional arguments
    parser.add_argument(
        "--instance_count",
        type=int,
        default=1,
        help="Number of instances for the SageMaker job (default: 1)",
    )
    parser.add_argument(
        "--volume_size",
        type=int,
        default=200,
        help="EBS volume size in GB (default: 200)",
    )
    parser.add_argument(
        "--framework_version",
        type=str,
        default="2.2",
        help="PyTorch framework version for SageMaker container (default: 2.2)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (overrides HF_TOKEN env if provided)",
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=4,
        help="Number of Ray actors for parallel inference inside each instance (default: 4)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="esm3_sm_open_v1",
        help="ESM model name or local path (default: esm3_sm_open_v1)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Session & role
    sess = sagemaker.Session()
    role = args.arn_role

    input_s3 = args.s3_input_path
    embeddings_output_s3 = args.s3_embedding_output_path
    artifact_s3 = args.s3_artifact_path

    # HF token: CLI > ENV
    hf_token = args.hf_token if args.hf_token is not None else os.environ.get("HF_TOKEN")

    # 2. Input channel
    shared_input = TrainingInput(
        s3_data=input_s3,
        distribution="ShardedByS3Key",
    )

    # 3. SageMaker estimator
    estimator = PyTorch(
        entry_point="scripts/extract.py",
        source_dir=".",
        role=role,
        framework_version=args.framework_version,
        py_version="py310",
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        hyperparameters={
            "num_actors": args.num_actors,
            "input_path": "/opt/ml/input/data/training",
            "output_dir_path": embeddings_output_s3,
            "model_name_or_path": args.model_name_or_path,
        },
        environment={
            "HF_TOKEN": hf_token or "",
        },
        output_path=artifact_s3,
        sagemaker_session=sess,
    )

    # 4. Submit job
    estimator.fit({"training": shared_input})
