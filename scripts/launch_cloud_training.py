#!/usr/bin/env python3
"""
Launch Cloud-Native Training Job on AWS SageMaker

This script launches the iterative distillation pipeline as a SageMaker Training Job.
All processing happens in the cloud - no local data downloads required.

Cloud-Native Architecture:
- Data: S3 → SageMaker Training Container (no local downloads)
- Teacher: Bedrock Claude (AWS-native)
- Training: SageMaker Training Job with GPU instances
- Output: Checkpoints and model saved to S3

Usage:
    # Launch with default settings
    python scripts/launch_cloud_training.py

    # Launch with spot instances (cost savings)
    python scripts/launch_cloud_training.py --use-spot

    # Launch with specific instance type
    python scripts/launch_cloud_training.py --instance-type ml.g5.12xlarge

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables must be set externally


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch iterative distillation as SageMaker Training Job"
    )

    # Job settings
    parser.add_argument('--job-name-prefix', type=str, default='granite-8b-distillation',
                        help='Prefix for the training job name')

    # Instance settings
    parser.add_argument('--instance-type', type=str, default='ml.g5.12xlarge',
                        help='SageMaker instance type (default: ml.g5.12xlarge = 4xA10G)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of training instances')
    parser.add_argument('--volume-size', type=int, default=100,
                        help='EBS volume size in GB')
    parser.add_argument('--max-run-seconds', type=int, default=86400,
                        help='Maximum training time in seconds (default: 24h)')

    # Spot instances (cost savings)
    parser.add_argument('--use-spot', action='store_true',
                        help='Use spot instances for cost savings')
    parser.add_argument('--max-wait-seconds', type=int, default=172800,
                        help='Max wait for spot capacity (default: 48h)')

    # S3 locations
    parser.add_argument('--s3-bucket', type=str,
                        default='granite-8b-unified-automotive-data',
                        help='S3 bucket for data and outputs')
    parser.add_argument('--train-prefix', type=str, default='data/processed',
                        help='S3 prefix for training data')
    parser.add_argument('--eval-prefix', type=str, default='data/eval',
                        help='S3 prefix for evaluation prompts')
    parser.add_argument('--output-prefix', type=str, default='output/distillation',
                        help='S3 prefix for outputs')

    # Training hyperparameters
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--quality-threshold', type=float, default=7.0)
    parser.add_argument('--convergence-threshold', type=float, default=8.0)
    parser.add_argument('--max-corrections-per-epoch', type=int, default=500)
    parser.add_argument('--eval-samples', type=int, default=200)

    # Docker image
    parser.add_argument('--image-uri', type=str, default=None,
                        help='Custom training container image URI')

    # IAM role
    parser.add_argument('--role', type=str,
                        default=os.environ.get('SAGEMAKER_ROLE'),
                        help='SageMaker execution role ARN')

    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration without launching')

    return parser.parse_args()


def get_training_job_name(prefix: str) -> str:
    """Generate unique training job name"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{prefix}-{timestamp}"


def get_default_role(session: Session) -> str:
    """Get default SageMaker execution role"""
    try:
        return session.get_caller_identity_arn()
    except Exception:
        # Fallback to environment variable
        role = os.environ.get('SAGEMAKER_ROLE')
        if role:
            return role
        raise ValueError(
            "No SageMaker role found. Set SAGEMAKER_ROLE environment variable "
            "or pass --role argument."
        )


def get_training_image(args, region: str) -> str:
    """
    Get the training container image URI.

    Uses either a custom image or the HuggingFace DLC.
    """
    if args.image_uri:
        return args.image_uri

    # Use HuggingFace Deep Learning Container
    # https://github.com/aws/deep-learning-containers
    account_map = {
        'us-east-1': '763104351884',
        'us-west-2': '763104351884',
        'eu-west-1': '763104351884',
    }

    account = account_map.get(region, '763104351884')

    # HuggingFace PyTorch training container
    return (
        f"{account}.dkr.ecr.{region}.amazonaws.com/"
        f"huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04"
    )


def create_hyperparameters(args) -> dict:
    """Create hyperparameters dict for the training job"""
    return {
        'max-epochs': str(args.max_epochs),
        'quality-threshold': str(args.quality_threshold),
        'convergence-threshold': str(args.convergence_threshold),
        'max-corrections-per-epoch': str(args.max_corrections_per_epoch),
        'eval-samples': str(args.eval_samples),
        'model-name': 'ibm-granite/granite-8b-code-instruct-128k',
        'teacher-model': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'teacher-workers': '10',
    }


def create_estimator(args, session: Session, role: str) -> Estimator:
    """Create SageMaker Estimator for the training job"""
    region = session.boto_region_name

    # Metric definitions for CloudWatch
    metric_definitions = [
        {'Name': 'train_loss', 'Regex': r'Train loss: ([0-9\.]+)'},
        {'Name': 'avg_student_score', 'Regex': r'Avg student score: ([0-9\.]+)'},
        {'Name': 'correction_rate', 'Regex': r'Correction rate: ([0-9\.]+)%'},
        {'Name': 'num_corrections', 'Regex': r'Corrections: ([0-9]+)'},
    ]

    # Environment variables (including Bedrock access)
    environment = {
        'AWS_REGION': region,
        'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
        # Bedrock credentials inherited from IAM role
    }

    estimator_kwargs = {
        'image_uri': get_training_image(args, region),
        'role': role,
        'instance_count': args.instance_count,
        'instance_type': args.instance_type,
        'volume_size': args.volume_size,
        'max_run': args.max_run_seconds,
        'hyperparameters': create_hyperparameters(args),
        'metric_definitions': metric_definitions,
        'environment': environment,
        'entry_point': 'run_iterative_pipeline.py',
        'source_dir': str(Path(__file__).parent.parent),
        'sagemaker_session': session,
        'output_path': f"s3://{args.s3_bucket}/{args.output_prefix}",
        'checkpoint_s3_uri': f"s3://{args.s3_bucket}/{args.output_prefix}/checkpoints",
        'base_job_name': args.job_name_prefix,
    }

    # Spot instance configuration
    if args.use_spot:
        estimator_kwargs['use_spot_instances'] = True
        estimator_kwargs['max_wait'] = args.max_wait_seconds

    return Estimator(**estimator_kwargs)


def create_data_channels(args) -> dict:
    """Create input data channels for the training job"""
    s3_base = f"s3://{args.s3_bucket}"

    return {
        'train': TrainingInput(
            s3_data=f"{s3_base}/{args.train_prefix}",
            s3_data_type='S3Prefix',
            content_type='application/jsonlines',
        ),
        'eval': TrainingInput(
            s3_data=f"{s3_base}/{args.eval_prefix}",
            s3_data_type='S3Prefix',
            content_type='application/jsonlines',
        ),
    }


def print_job_config(args, job_name: str, role: str, region: str):
    """Print job configuration summary"""
    print("\n" + "="*70)
    print("SAGEMAKER TRAINING JOB CONFIGURATION")
    print("="*70)
    print(f"\n[Job Details]")
    print(f"  Job name: {job_name}")
    print(f"  Region: {region}")
    print(f"  Role: {role[:50]}...")

    print(f"\n[Instance Configuration]")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Instance count: {args.instance_count}")
    print(f"  Volume size: {args.volume_size} GB")
    print(f"  Max runtime: {args.max_run_seconds // 3600} hours")
    print(f"  Spot instances: {'Yes' if args.use_spot else 'No'}")

    print(f"\n[Data Channels]")
    print(f"  S3 bucket: {args.s3_bucket}")
    print(f"  Training data: s3://{args.s3_bucket}/{args.train_prefix}")
    print(f"  Eval prompts: s3://{args.s3_bucket}/{args.eval_prefix}")
    print(f"  Output: s3://{args.s3_bucket}/{args.output_prefix}")

    print(f"\n[Hyperparameters]")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Quality threshold: {args.quality_threshold}")
    print(f"  Convergence threshold: {args.convergence_threshold}")
    print(f"  Max corrections/epoch: {args.max_corrections_per_epoch}")
    print(f"  Eval samples: {args.eval_samples}")

    # Cost estimate
    instance_costs = {
        'ml.g5.xlarge': 1.41,
        'ml.g5.2xlarge': 2.82,
        'ml.g5.4xlarge': 5.64,
        'ml.g5.12xlarge': 7.09,
        'ml.p4d.24xlarge': 40.77,
    }
    hourly_cost = instance_costs.get(args.instance_type, 0)
    max_cost = hourly_cost * (args.max_run_seconds / 3600) * args.instance_count

    if args.use_spot:
        max_cost *= 0.3  # Spot is typically 70% cheaper

    print(f"\n[Cost Estimate]")
    print(f"  Instance cost: ${hourly_cost:.2f}/hour")
    print(f"  Max cost (worst case): ${max_cost:.2f}")
    if args.use_spot:
        print(f"  (Spot pricing - actual may vary)")

    print("="*70)


def main():
    args = parse_args()

    # Initialize SageMaker session
    session = Session()
    region = session.boto_region_name

    # Get IAM role
    role = args.role or get_default_role(session)

    # Generate job name
    job_name = get_training_job_name(args.job_name_prefix)

    # Print configuration
    print_job_config(args, job_name, role, region)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. No job launched.")
        return

    # Confirm launch
    print("\n[LAUNCH] Starting SageMaker Training Job...")

    # Create estimator
    estimator = create_estimator(args, session, role)

    # Create data channels
    data_channels = create_data_channels(args)

    # Launch training job
    estimator.fit(
        inputs=data_channels,
        job_name=job_name,
        wait=False  # Don't block - job runs asynchronously
    )

    print(f"\n[SUCCESS] Training job launched!")
    print(f"  Job name: {job_name}")
    print(f"  Status: IN_PROGRESS")
    print(f"\n[Monitor] View in AWS Console:")
    print(f"  https://{region}.console.aws.amazon.com/sagemaker/home"
          f"?region={region}#/jobs/{job_name}")
    print(f"\n[CLI] Check status:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
    print(f"\n[CLI] View logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow "
          f"--log-stream-name-prefix {job_name}")


if __name__ == "__main__":
    main()
