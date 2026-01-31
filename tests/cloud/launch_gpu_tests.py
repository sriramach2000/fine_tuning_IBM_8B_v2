#!/usr/bin/env python3
"""
Launch GPU/Hardware Tests as SageMaker Training Job

Packages the project, uploads to S3, and launches a GPU training job
that runs the pytest GPU test suite.

Usage:
    python tests/cloud/launch_gpu_tests.py
    python tests/cloud/launch_gpu_tests.py --instance-type ml.g5.2xlarge
    python tests/cloud/launch_gpu_tests.py --dry-run
    python tests/cloud/launch_gpu_tests.py --wait
"""

import os
import sys
import json
import time
import argparse
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.parent

# AWS DLC (Deep Learning Container) images for HuggingFace
# Format: {account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:{tag}
DLC_ACCOUNTS = {
    'us-east-1': '763104351884',
    'us-west-2': '763104351884',
    'eu-west-1': '763104351884',
    'ap-southeast-1': '763104351884',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch GPU hardware tests on SageMaker"
    )
    parser.add_argument('--instance-type', type=str, default='ml.g5.xlarge',
                        help='GPU instance (default: ml.g5.xlarge - 1x A10G 24GB, ~$1.41/hr)')
    parser.add_argument('--max-runtime', type=int, default=3600,
                        help='Max runtime seconds (default: 3600)')
    parser.add_argument('--s3-bucket', type=str,
                        default='granite-8b-unified-automotive-data')
    parser.add_argument('--region', type=str,
                        default=os.environ.get('AWS_REGION', 'us-east-1'))
    parser.add_argument('--role', type=str,
                        default=os.environ.get('SAGEMAKER_ROLE'))
    parser.add_argument('--with-data', action='store_true',
                        help='Attach S3 training/validation data channels')
    parser.add_argument('--data-prefix', type=str, default='data/processed',
                        help='S3 prefix for train/val data (default: data/processed)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job completion and stream logs')
    return parser.parse_args()


def get_role(args):
    """Get SageMaker execution role"""
    if args.role:
        return args.role

    # Try to get from SageMaker config or IAM
    iam = boto3.client('iam')
    try:
        # Look for the project's SageMaker role
        response = iam.get_role(RoleName='granite-8b-avb-tsn-finetuning-sagemaker-role')
        return response['Role']['Arn']
    except Exception:
        pass

    raise ValueError(
        "No SageMaker role found. Set SAGEMAKER_ROLE env variable or "
        "create the 'granite-8b-avb-tsn-finetuning-sagemaker-role' IAM role."
    )


def get_training_image(region):
    """Get HuggingFace PyTorch GPU training container image"""
    account = DLC_ACCOUNTS.get(region, '763104351884')
    # PyTorch 2.5.1 + Transformers 4.49.0 + CUDA + Python 3.11
    tag = '2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04'
    return f"{account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:{tag}"


def package_and_upload(s3_bucket, job_name, region):
    """Package project source and upload to S3"""
    s3 = boto3.client('s3', region_name=region)

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = Path(tmpdir) / 'source.tar.gz'

        with tarfile.open(tarball, 'w:gz') as tar:
            # Entry point
            tar.add(PROJECT_ROOT / 'run_gpu_tests.py', arcname='run_gpu_tests.py')

            # Test files
            for f in (PROJECT_ROOT / 'tests' / 'cloud').glob('*.py'):
                tar.add(f, arcname=f'tests/cloud/{f.name}')
            for f in (PROJECT_ROOT / 'tests').glob('*.py'):
                tar.add(f, arcname=f'tests/{f.name}')

            # Training modules (needed for import tests)
            training_dir = PROJECT_ROOT / 'training'
            if training_dir.exists():
                for f in training_dir.glob('*.py'):
                    tar.add(f, arcname=f'training/{f.name}')

            # Evaluation module
            eval_dir = PROJECT_ROOT / 'evaluation'
            if eval_dir.exists():
                for f in eval_dir.glob('*.py'):
                    tar.add(f, arcname=f'evaluation/{f.name}')

            # Scripts module
            scripts_dir = PROJECT_ROOT / 'scripts'
            if scripts_dir.exists():
                for f in scripts_dir.glob('*.py'):
                    tar.add(f, arcname=f'scripts/{f.name}')

            # Config
            tar.add(PROJECT_ROOT / 'config.yaml', arcname='config.yaml')

            # Requirements
            tar.add(PROJECT_ROOT / 'requirements.txt', arcname='requirements.txt')
            test_req = PROJECT_ROOT / 'tests' / 'requirements-test.txt'
            if test_req.exists():
                tar.add(test_req, arcname='tests/requirements-test.txt')

        s3_key = f"gpu-tests/{job_name}/source.tar.gz"
        s3.upload_file(str(tarball), s3_bucket, s3_key)

        return f"s3://{s3_bucket}/{s3_key}"


def print_config(args, job_name, role):
    costs = {
        'ml.g5.xlarge': 1.41, 'ml.g5.2xlarge': 1.52,
        'ml.g5.4xlarge': 2.03, 'ml.g5.12xlarge': 7.09,
        'ml.p4d.24xlarge': 37.69,
    }
    hourly = costs.get(args.instance_type, 2.0)
    max_cost = hourly * (args.max_runtime / 3600)

    print("\n" + "=" * 70)
    print("GPU HARDWARE TEST JOB")
    print("=" * 70)
    print(f"  Job:      {job_name}")
    print(f"  Instance: {args.instance_type}")
    print(f"  Region:   {args.region}")
    print(f"  Role:     {role[:60]}...")
    print(f"  Image:    HuggingFace PyTorch 2.5.1 GPU (CUDA 12.4)")
    print(f"  Runtime:  {args.max_runtime // 60} min max")
    print(f"  Cost:     ~${hourly:.2f}/hr (max ${max_cost:.2f})")
    print("=" * 70)


def wait_for_job(sm_client, job_name, region):
    """Poll job status and stream output"""
    logs_client = boto3.client('logs', region_name=region)
    log_group = '/aws/sagemaker/TrainingJobs'

    print("\n[WAITING] Monitoring job progress...\n")

    last_status = None
    while True:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']

        if status != last_status:
            print(f"  Status: {status}")
            last_status = status

        if status in ('Completed', 'Failed', 'Stopped'):
            break

        time.sleep(15)

    # Print final details
    print(f"\n{'=' * 70}")
    print(f"JOB RESULT: {status}")
    print(f"{'=' * 70}")

    if status == 'Failed':
        reason = response.get('FailureReason', 'Unknown')
        print(f"  Failure reason: {reason}")

    duration = response.get('TrainingTimeInSeconds', 0)
    print(f"  Duration: {duration // 60}m {duration % 60}s")

    if status == 'Completed':
        model_artifacts = response.get('ModelArtifacts', {}).get('S3ModelArtifacts')
        if model_artifacts:
            print(f"  Results: {model_artifacts}")

    # Try to get CloudWatch logs
    try:
        log_streams = logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            orderBy='LastEventTime',
            descending=True,
            limit=1,
        )
        if log_streams.get('logStreams'):
            stream = log_streams['logStreams'][0]['logStreamName']
            events = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream,
                startFromHead=False,
                limit=50,
            )
            print(f"\n[LOGS] Last 50 log lines:")
            for event in events.get('events', []):
                print(f"  {event['message'].rstrip()}")
    except Exception as e:
        print(f"\n[LOGS] Could not fetch logs: {e}")

    return status


def main():
    args = parse_args()
    role = get_role(args)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"gpu-hw-tests-{timestamp}"

    print_config(args, job_name, role)

    if args.with_data:
        print(f"  Data:     s3://{args.s3_bucket}/{args.data_prefix}/")
        print(f"            train.jsonl + val.jsonl")

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. No job launched.")
        return

    # Upload source code
    print("\n[UPLOAD] Packaging and uploading source code...")
    source_uri = package_and_upload(args.s3_bucket, job_name, args.region)
    print(f"  Uploaded to: {source_uri}")

    # Create training job via boto3
    sm_client = boto3.client('sagemaker', region_name=args.region)

    # When data is attached, run both gpu and sagemaker marker tests
    test_marker = 'gpu or sagemaker' if args.with_data else 'gpu'

    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': role,
        'AlgorithmSpecification': {
            'TrainingImage': get_training_image(args.region),
            'TrainingInputMode': 'File',
        },
        'HyperParameters': {
            'sagemaker_program': 'run_gpu_tests.py',
            'sagemaker_submit_directory': source_uri,
            'test-marker': test_marker,
            's3-bucket': args.s3_bucket,
            'region': args.region,
        },
        'ResourceConfig': {
            'InstanceCount': 1,
            'InstanceType': args.instance_type,
            'VolumeSizeInGB': 100,
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': args.max_runtime,
        },
        'OutputDataConfig': {
            'S3OutputPath': f"s3://{args.s3_bucket}/gpu-tests/{job_name}/output",
        },
        'Environment': {
            'S3_BUCKET': args.s3_bucket,
            'AWS_REGION': args.region,
        },
    }

    # Add data channels when --with-data is specified
    if args.with_data:
        training_params['InputDataConfig'] = [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{args.s3_bucket}/{args.data_prefix}/train.jsonl",
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/jsonlines',
                'InputMode': 'File',
            },
            {
                'ChannelName': 'val',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{args.s3_bucket}/{args.data_prefix}/val.jsonl",
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/jsonlines',
                'InputMode': 'File',
            },
        ]

    print("\n[LAUNCH] Creating SageMaker Training Job...")
    sm_client.create_training_job(**training_params)
    print(f"  Job submitted: {job_name}")

    if args.wait:
        status = wait_for_job(sm_client, job_name, args.region)
        sys.exit(0 if status == 'Completed' else 1)
    else:
        print(f"\n[MONITOR] View in console:")
        print(f"  https://{args.region}.console.aws.amazon.com/sagemaker/home"
              f"?region={args.region}#/jobs/{job_name}")
        print(f"\n[CLI] Check status:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
        print(f"\n[CLI] Stream logs:")
        print(f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")


if __name__ == "__main__":
    main()
