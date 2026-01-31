#!/usr/bin/env python3
"""
Launch Cloud-Native Tests as SageMaker Processing Job

This script deploys the test suite to run in AWS SageMaker,
validating real integration with S3, Bedrock, and other services.

Usage:
    # Launch tests in AWS
    python tests/cloud/launch_cloud_tests.py

    # Dry run (show config without launching)
    python tests/cloud/launch_cloud_tests.py --dry-run

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import argparse
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
from sagemaker import Session
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables must be set externally


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Launch cloud-native tests as SageMaker Processing Job"
    )

    parser.add_argument('--job-name-prefix', type=str, default='granite-cloud-tests',
                        help='Prefix for the job name')
    parser.add_argument('--instance-type', type=str, default='ml.t3.medium',
                        help='Instance type (default: ml.t3.medium - cheap for tests)')
    parser.add_argument('--instance-count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--max-runtime', type=int, default=3600,
                        help='Max runtime in seconds (default: 1 hour)')

    parser.add_argument('--s3-bucket', type=str,
                        default='granite-8b-unified-automotive-data',
                        help='S3 bucket for testing')
    parser.add_argument('--output-prefix', type=str, default='test-results',
                        help='S3 prefix for test results')

    parser.add_argument('--region', type=str,
                        default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region')
    parser.add_argument('--role', type=str,
                        default=os.environ.get('SAGEMAKER_ROLE'),
                        help='SageMaker execution role ARN')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print configuration without launching')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job to complete')

    return parser.parse_args()


def get_job_name(prefix: str) -> str:
    """Generate unique job name"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{prefix}-{timestamp}"


def get_default_role(session: Session) -> str:
    """Get default SageMaker execution role"""
    try:
        return session.get_caller_identity_arn()
    except Exception:
        role = os.environ.get('SAGEMAKER_ROLE')
        if role:
            return role
        raise ValueError(
            "No SageMaker role found. Set SAGEMAKER_ROLE environment variable."
        )


def get_processing_image(region: str) -> str:
    """Get the processing container image URI"""
    account_map = {
        'us-east-1': '763104351884',
        'us-west-2': '763104351884',
        'eu-west-1': '763104351884',
    }

    account = account_map.get(region, '763104351884')

    # Use PyTorch processing container (more widely available)
    return (
        f"{account}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-training:2.0.1-cpu-py310-ubuntu20.04-sagemaker"
    )


def print_job_config(args, job_name: str, role: str):
    """Print job configuration"""
    print("\n" + "="*70)
    print("CLOUD TEST JOB CONFIGURATION")
    print("="*70)
    print(f"\n[Job Details]")
    print(f"  Job name: {job_name}")
    print(f"  Region: {args.region}")
    print(f"  Role: {role[:50]}...")

    print(f"\n[Instance Configuration]")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Instance count: {args.instance_count}")
    print(f"  Max runtime: {args.max_runtime // 60} minutes")

    print(f"\n[Test Configuration]")
    print(f"  S3 bucket: {args.s3_bucket}")
    print(f"  Output prefix: {args.output_prefix}")

    # Cost estimate
    instance_costs = {
        'ml.t3.medium': 0.05,
        'ml.t3.large': 0.10,
        'ml.m5.large': 0.14,
    }
    hourly_cost = instance_costs.get(args.instance_type, 0.10)
    max_cost = hourly_cost * (args.max_runtime / 3600)

    print(f"\n[Cost Estimate]")
    print(f"  Instance cost: ${hourly_cost:.2f}/hour")
    print(f"  Max cost: ${max_cost:.2f}")

    print("="*70)


def package_source_code(project_root: Path, s3_bucket: str, job_name: str) -> str:
    """Package source code and upload to S3"""
    s3_client = boto3.client('s3')

    # Create tarball with required files
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_path = Path(tmpdir) / 'source.tar.gz'

        with tarfile.open(tarball_path, 'w:gz') as tar:
            # Add evaluation module
            eval_dir = project_root / 'evaluation'
            if eval_dir.exists():
                tar.add(eval_dir, arcname='evaluation')

            # Add scripts module
            scripts_dir = project_root / 'scripts'
            if scripts_dir.exists():
                for f in scripts_dir.glob('*.py'):
                    tar.add(f, arcname=f'scripts/{f.name}')

            # Add the test script
            test_script = project_root / 'tests/cloud/run_cloud_tests.py'
            if test_script.exists():
                tar.add(test_script, arcname='run_cloud_tests.py')

        # Upload to S3
        s3_key = f"code/{job_name}/source.tar.gz"
        s3_client.upload_file(str(tarball_path), s3_bucket, s3_key)

        return f"s3://{s3_bucket}/{s3_key}"


def main():
    args = parse_args()

    # Initialize SageMaker session
    session = Session()

    # Get IAM role
    role = args.role or get_default_role(session)

    # Generate job name
    job_name = get_job_name(args.job_name_prefix)

    # Print configuration
    print_job_config(args, job_name, role)

    if args.dry_run:
        print("\n[DRY RUN] Configuration validated. No job launched.")
        return

    print("\n[LAUNCH] Starting SageMaker Processing Job...")

    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Package and upload source code to S3
    print("[UPLOAD] Packaging and uploading source code to S3...")
    source_s3_uri = package_source_code(project_root, args.s3_bucket, job_name)
    print(f"  Source uploaded to: {source_s3_uri}")

    # Create ScriptProcessor
    processor = ScriptProcessor(
        role=role,
        image_uri=get_processing_image(args.region),
        command=['python3'],
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        max_runtime_in_seconds=args.max_runtime,
        base_job_name=args.job_name_prefix,
        sagemaker_session=session,
        env={
            'S3_BUCKET': args.s3_bucket,
            'AWS_REGION': args.region,
            'PYTHONPATH': '/opt/ml/processing/input/source',
        }
    )

    # Define inputs (source code)
    inputs = [
        ProcessingInput(
            source=source_s3_uri,
            destination='/opt/ml/processing/input/source',
            input_name='source',
            s3_data_type='S3Prefix',
            s3_input_mode='File',
        )
    ]

    # Define outputs
    outputs = [
        ProcessingOutput(
            output_name='results',
            source='/opt/ml/processing/output',
            destination=f"s3://{args.s3_bucket}/{args.output_prefix}/{job_name}",
        )
    ]

    # Create wrapper script that extracts code and runs tests
    wrapper_script = '''#!/usr/bin/env python3
import os
import sys
import tarfile

# Extract source code
source_input = '/opt/ml/processing/input/source'
if os.path.exists(source_input):
    for f in os.listdir(source_input):
        if f.endswith('.tar.gz'):
            with tarfile.open(os.path.join(source_input, f), 'r:gz') as tar:
                tar.extractall(source_input)
    print(f"Extracted files: {os.listdir(source_input)}")

# Add to path
sys.path.insert(0, source_input)
os.environ['PYTHONPATH'] = source_input

# Run tests
from run_cloud_tests import main
main()
'''

    # Write wrapper script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper_script)
        wrapper_path = f.name

    try:
        # Run the job
        processor.run(
            code=wrapper_path,
            inputs=inputs,
            outputs=outputs,
            wait=args.wait,
            job_name=job_name,
        )
    finally:
        os.unlink(wrapper_path)

    print(f"\n[SUCCESS] Test job launched!")
    print(f"  Job name: {job_name}")

    if args.wait:
        print(f"\n[RESULTS] Check results at:")
        print(f"  s3://{args.s3_bucket}/{args.output_prefix}/{job_name}/")
    else:
        print(f"\n[MONITOR] View in AWS Console:")
        print(f"  https://{args.region}.console.aws.amazon.com/sagemaker/home"
              f"?region={args.region}#/processing-jobs/{job_name}")
        print(f"\n[CLI] Check status:")
        print(f"  aws sagemaker describe-processing-job --processing-job-name {job_name}")


if __name__ == "__main__":
    main()
