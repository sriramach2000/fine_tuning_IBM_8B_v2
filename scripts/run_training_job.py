#!/usr/bin/env python3
"""
Launch SageMaker Training Job for Granite-8B Fine-tuning

This script orchestrates the SageMaker training job for fine-tuning
IBM Granite-8B on automotive code (AVB/TSN).

Features:
- HuggingFace Estimator with QLoRA configuration
- Spot instance support for cost savings
- Progress monitoring with CloudWatch
- Automatic S3 data management

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime

import boto3
from sagemaker import Session
from sagemaker.huggingface import HuggingFace
# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables must be set externally


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_training_data(s3_client, bucket: str, prefix: str = "data/splits") -> bool:
    """Verify training data exists in S3"""
    try:
        # Check for train.jsonl
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}/train.jsonl")
        print(f"[SageMaker] Found: s3://{bucket}/{prefix}/train.jsonl")

        # Check for val.jsonl
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}/val.jsonl")
        print(f"[SageMaker] Found: s3://{bucket}/{prefix}/val.jsonl")

        return True
    except Exception as e:
        print(f"[SageMaker] Training data not found: {e}")
        return False


def upload_training_data(
    s3_client,
    bucket: str,
    local_dir: str = "./data/splits",
    s3_prefix: str = "data/splits"
):
    """Upload local training data to S3"""
    local_path = Path(local_dir)

    if not local_path.exists():
        print(f"[SageMaker] Local data directory not found: {local_dir}")
        return False

    for file_path in local_path.glob("*.jsonl"):
        s3_key = f"{s3_prefix}/{file_path.name}"
        print(f"[SageMaker] Uploading: {file_path} -> s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)

    return True


def create_estimator(
    config: dict,
    role_arn: str,
    session: Session,
    instance_type: str = None,
    use_spot: bool = True
) -> HuggingFace:
    """Create HuggingFace estimator for training"""

    # Training configuration
    training_config = config.get('training', {})
    qlora_config = config.get('qlora', {})
    model_config = config.get('model', {})

    # Hyperparameters for training script
    hyperparameters = {
        # Model
        'model-name': model_config.get('name', 'ibm-granite/granite-8b-code-instruct-128k'),
        'max-seq-length': model_config.get('max_seq_length', 4096),

        # QLoRA
        'load-in-4bit': True,
        'bnb-4bit-compute-dtype': qlora_config.get('bnb_4bit_compute_dtype', 'bfloat16'),
        'bnb-4bit-quant-type': qlora_config.get('bnb_4bit_quant_type', 'nf4'),
        'bnb-4bit-use-double-quant': qlora_config.get('bnb_4bit_use_double_quant', True),

        # LoRA
        'lora-r': qlora_config.get('lora_r', 32),
        'lora-alpha': qlora_config.get('lora_alpha', 64),
        'lora-dropout': qlora_config.get('lora_dropout', 0.05),

        # Training
        'num-train-epochs': training_config.get('num_epochs', 5),
        'per-device-train-batch-size': training_config.get('per_device_train_batch_size', 2),
        'gradient-accumulation-steps': training_config.get('gradient_accumulation_steps', 8),
        'learning-rate': training_config.get('learning_rate', 1e-4),
        'warmup-ratio': training_config.get('warmup_ratio', 0.05),
        'lr-scheduler-type': training_config.get('lr_scheduler_type', 'cosine'),
        'weight-decay': training_config.get('weight_decay', 0.01),
        'max-grad-norm': training_config.get('max_grad_norm', 0.3),
        'optim': training_config.get('optim', 'paged_adamw_8bit'),

        # Logging
        'logging-steps': training_config.get('logging_steps', 10),
        'eval-steps': training_config.get('eval_steps', 50),
        'save-steps': training_config.get('save_steps', 100),

        # Early stopping
        'early-stopping-patience': training_config.get('early_stopping_patience', 3),
    }

    # Instance configuration
    aws_config = config.get('aws', {}).get('training_job', {})
    instance_type = instance_type or aws_config.get('instance_type', 'ml.p4d.24xlarge')
    volume_size = aws_config.get('volume_size_gb', 200)
    max_runtime = aws_config.get('max_runtime_seconds', 28800)

    # Spot instance configuration
    use_spot = use_spot and aws_config.get('use_spot_instances', True)
    max_wait = aws_config.get('max_wait_seconds', 36000) if use_spot else None

    # Create timestamp for job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"granite-8b-avb-tsn-{timestamp}"

    print(f"\n[SageMaker] Creating estimator...")
    print(f"  Instance type: {instance_type}")
    print(f"  Spot instances: {use_spot}")
    print(f"  Volume size: {volume_size} GB")
    print(f"  Max runtime: {max_runtime} seconds")

    estimator = HuggingFace(
        entry_point='train_granite_qlora.py',
        source_dir='./training',
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        volume_size=volume_size,
        max_run=max_runtime,
        use_spot_instances=use_spot,
        max_wait=max_wait,
        transformers_version='4.49.0',
        pytorch_version='2.5.1',
        py_version='py311',
        hyperparameters=hyperparameters,
        sagemaker_session=session,
        base_job_name='granite-8b-avb-tsn',
        tags=[
            {'Key': 'Project', 'Value': 'granite-8b-avb-tsn-finetuning'},
            {'Key': 'Model', 'Value': 'ibm-granite/granite-8b-code-instruct-128k'},
            {'Key': 'Domain', 'Value': 'automotive-embedded'}
        ]
    )

    return estimator, job_name


def monitor_training(estimator, job_name: str):
    """Monitor training job progress"""
    print(f"\n[SageMaker] Monitoring training job: {job_name}")
    print("[SageMaker] Press Ctrl+C to stop monitoring (job will continue)")

    try:
        # Wait for training to complete
        estimator.fit(wait=True, logs='All')
        print("\n[SageMaker] Training completed successfully!")

    except KeyboardInterrupt:
        print("\n[SageMaker] Monitoring stopped. Job continues in background.")
        print(f"[SageMaker] Check status: aws sagemaker describe-training-job --training-job-name {job_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch SageMaker training job for Granite-8B"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--instance-type',
        type=str,
        help='Override instance type (e.g., ml.g5.12xlarge)'
    )
    parser.add_argument(
        '--no-spot',
        action='store_true',
        help='Disable spot instances'
    )
    parser.add_argument(
        '--upload-data',
        action='store_true',
        help='Upload local training data to S3'
    )
    parser.add_argument(
        '--local-data-dir',
        type=str,
        default='./data/splits',
        help='Local data directory'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        help='Override max training steps (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without launching job'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # AWS configuration
    aws_config = config.get('aws', {})
    region = aws_config.get('region', 'us-east-1')
    bucket = aws_config.get('s3', {}).get('bucket_name', 'granite-8b-unified-automotive-data')
    role_name = aws_config.get('iam', {}).get('role_name', 'SageMakerGranite8BFineTuningRole')

    # Get account ID
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

    print("="*70)
    print("[SageMaker] Granite-8B Fine-tuning Job Launcher")
    print("="*70)
    print(f"  Region: {region}")
    print(f"  S3 Bucket: {bucket}")
    print(f"  IAM Role: {role_arn}")
    print("="*70)

    # Initialize AWS clients
    s3_client = boto3.client('s3', region_name=region)
    session = Session(boto_session=boto3.Session(region_name=region))

    # Upload data if requested
    if args.upload_data:
        print("\n[SageMaker] Uploading training data...")
        upload_training_data(s3_client, bucket, args.local_data_dir)

    # Verify training data
    if not verify_training_data(s3_client, bucket):
        print("\n[SageMaker] ERROR: Training data not found in S3!")
        print("[SageMaker] Run with --upload-data to upload local data")
        sys.exit(1)

    # Create estimator
    use_spot = not args.no_spot
    estimator, job_name = create_estimator(
        config=config,
        role_arn=role_arn,
        session=session,
        instance_type=args.instance_type,
        use_spot=use_spot
    )

    # Dry run - just show config
    if args.dry_run:
        print("\n[SageMaker] DRY RUN - Configuration:")
        print(f"  Job name: {job_name}")
        print(f"  Entry point: train_granite_qlora.py")
        print(f"  Instance: {args.instance_type or config['aws']['training_job']['instance_type']}")
        print(f"  Spot: {use_spot}")
        print("\n[SageMaker] Hyperparameters:")
        for k, v in estimator.hyperparameters().items():
            print(f"    {k}: {v}")
        return

    # Define S3 input channels
    s3_data_prefix = f"s3://{bucket}/data/splits"

    inputs = {
        'train': f"{s3_data_prefix}/train.jsonl",
        'val': f"{s3_data_prefix}/val.jsonl"
    }

    print("\n[SageMaker] Input channels:")
    for channel, path in inputs.items():
        print(f"  {channel}: {path}")

    # Launch training
    print(f"\n[SageMaker] Launching training job: {job_name}")

    try:
        estimator.fit(inputs, job_name=job_name, wait=False)
        print(f"\n[SageMaker] Training job launched: {job_name}")
        print(f"[SageMaker] Monitor in console: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")

        # Optionally monitor
        print("\n[SageMaker] Starting log streaming (Ctrl+C to stop)...")
        monitor_training(estimator, job_name)

    except Exception as e:
        print(f"\n[SageMaker] ERROR launching training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
