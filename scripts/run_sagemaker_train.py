import argparse
import os

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SageMaker training job."
    )

    # Required arguments
    parser.add_argument("--arn_role", type=str, required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3_train_data_path", type=str, required=True, help="S3 path to training data")
    parser.add_argument("--s3_embed_data_path", type=str, required=True, help="S3 path to embedding data")
    parser.add_argument("--s3_artifact_path", type=str, required=True, help="S3 path to store SageMaker job artifacts")
    
    # Optional arguments
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge",
                        help="SageMaker instance type (default: ml.g4dn.xlarge)")
    parser.add_argument("--instance_count", type=int, default=1,
                        help="Number of instances for the SageMaker job (default: 1)")
    parser.add_argument("--volume_size", type=int, default=200,
                        help="EBS volume size in GB (default: 200)")
    parser.add_argument("--framework_version", type=str, default="2.2",
                        help="PyTorch framework version for SageMaker container (default: 2.2)")
    parser.add_argument("--py_version", type=str, default="py310",
                        help="Python version for SageMaker container (default: py310)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training (default: 24)")
    parser.add_argument("--query_size", type=int, default=96, help="Number of samples to query in each active learning iteration (default: 96)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--report_file", type=str, default='report.json', help="Path to report file (local or S3)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers per instance for distributed training (default: 1)")
    parser.add_argument("--strategy", type=str, default="passive", help="Active learning strategy to use (default: passive)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Session & role
    sess = sagemaker.Session()
    role = args.arn_role

    # 2. Input channels
    train_input = TrainingInput(
        s3_data=args.s3_train_data_path,
        distribution="ShardedByS3Key",
    )

    embed_input = TrainingInput(
        s3_data=args.s3_embed_data_path,
        distribution="FullyReplicated",
    )

    # 3. SageMaker estimator
    estimator = PyTorch(
        entry_point="scripts/train.py",
        source_dir=".",
        role=role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        hyperparameters={
            "train_data_path": args.s3_train_data_path,
            "embed_data_path": args.s3_embed_data_path,
            "batch_size": args.batch_size,
            "query_size": args.query_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "report_file": args.report_file,
            "num_workers": args.num_workers,
            "s3_artifact_path": args.s3_artifact_path,
            "strategy": args.strategy,
        },
        output_path=args.s3_artifact_path,
        sagemaker_session=sess,
    )

    # 4. Submit job
    estimator.fit({"train": train_input, "embed": embed_input})
