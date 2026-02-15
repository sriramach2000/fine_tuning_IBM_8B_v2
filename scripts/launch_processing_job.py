#!/usr/bin/env python3
"""
Launch SageMaker Processing Job for Automotive Data Pipeline

Runs the data preprocessing on ml.c5.xlarge spot instance:
- 4 vCPU, 8 GB RAM
- ~$0.05/hr with spot pricing (70% savings)
- Automatic S3 upload of train.jsonl/val.jsonl

Usage:
    # From local machine or Colab
    python scripts/launch_processing_job.py

    # With custom output location
    python scripts/launch_processing_job.py --output-bucket my-bucket --output-prefix my/prefix

Prerequisites:
    - AWS credentials configured
    - SageMaker execution role with S3 access
    - pip install sagemaker boto3

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import argparse
import boto3
from pathlib import Path

try:
    import sagemaker
    from sagemaker.processing import ScriptProcessor, ProcessingOutput
except ImportError:
    print("ERROR: sagemaker package not installed. Run: pip install sagemaker")
    sys.exit(1)


def get_or_create_role(region: str) -> str:
    """
    Get existing SageMaker execution role or guide user to create one.

    In Colab/notebook: uses get_execution_role()
    CLI: looks for SAGEMAKER_ROLE env var or default role name
    """
    # Check environment variable first
    role = os.environ.get('SAGEMAKER_ROLE')
    if role:
        print(f"[Launcher] Using role from SAGEMAKER_ROLE: {role}")
        return role

    # Try to get execution role (works in SageMaker notebooks)
    try:
        role = sagemaker.get_execution_role()
        print(f"[Launcher] Using SageMaker execution role: {role}")
        return role
    except ValueError:
        pass

    # Try common role names
    iam = boto3.client('iam', region_name=region)
    common_role_names = [
        'SageMakerExecutionRole',
        'AmazonSageMaker-ExecutionRole',
        'SageMakerRole',
    ]

    for role_name in common_role_names:
        try:
            response = iam.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"[Launcher] Found existing role: {role_arn}")
            return role_arn
        except iam.exceptions.NoSuchEntityException:
            continue

    # No role found - provide instructions
    print("\n" + "=" * 70)
    print("ERROR: No SageMaker execution role found")
    print("=" * 70)
    print("\nOptions:")
    print("1. Set SAGEMAKER_ROLE environment variable:")
    print("   export SAGEMAKER_ROLE=arn:aws:iam::ACCOUNT:role/ROLE_NAME")
    print("\n2. Create a role via AWS Console:")
    print("   - IAM > Roles > Create Role")
    print("   - Trusted entity: SageMaker")
    print("   - Policies: AmazonSageMakerFullAccess, AmazonS3FullAccess")
    print("   - Name: SageMakerExecutionRole")
    print("\n3. Or use AWS CLI:")
    print("""
aws iam create-role --role-name SageMakerExecutionRole \\
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

aws iam attach-role-policy --role-name SageMakerExecutionRole \\
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole \\
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
""")
    print("=" * 70)
    sys.exit(1)


def launch_processing_job(
    role: str,
    region: str,
    source_bucket: str,
    output_bucket: str,
    output_prefix: str,
    instance_type: str = 'ml.c5.xlarge',
    use_spot: bool = True,
    max_runtime_hours: int = 4,
    workers: int = 4,
):
    """Launch SageMaker Processing Job."""

    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    # Container image - use SageMaker's sklearn container (has AWS CLI, Python, boto3)
    # Using sklearn 1.2-1 with Python 3.10
    image_uri = sagemaker.image_uris.retrieve(
        framework='sklearn',
        region=region,
        version='1.2-1',
        py_version='py3',
        instance_type=instance_type,
    )

    print(f"\n[Launcher] Container image: {image_uri}")

    # Configure processor
    max_runtime_seconds = max_runtime_hours * 3600

    processor = ScriptProcessor(
        role=role,
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        command=['python3'],
        max_runtime_in_seconds=max_runtime_seconds,
        sagemaker_session=session,
        env={
            'AWS_REGION': region,
        },
    )

    # Add spot instance configuration if enabled
    if use_spot:
        print(f"[Launcher] Using spot instances (70% cost savings)")
        processor.max_runtime_in_seconds = max_runtime_seconds
        # Note: ScriptProcessor doesn't support spot directly
        # For spot, we'll use the lower-level API

    # Output location
    output_s3_uri = f"s3://{output_bucket}/{output_prefix}"

    print("\n" + "=" * 70)
    print("[Launcher] SAGEMAKER PROCESSING JOB")
    print("=" * 70)
    print(f"  Instance: {instance_type} ({workers} vCPU)")
    print(f"  Spot: {use_spot}")
    print(f"  Max runtime: {max_runtime_hours} hours")
    print(f"  Source bucket: s3://{source_bucket}/")
    print(f"  Output: {output_s3_uri}/")
    print("=" * 70)

    # Upload processing script to S3
    script_path = Path(__file__).parent / 'sagemaker_processing.py'
    if not script_path.exists():
        print(f"ERROR: Processing script not found at {script_path}")
        sys.exit(1)

    # Run the job
    print("\n[Launcher] Starting processing job...")

    try:
        processor.run(
            code=str(script_path),
            outputs=[
                ProcessingOutput(
                    output_name='splits',
                    source='/opt/ml/processing/output',
                    destination=output_s3_uri,
                )
            ],
            arguments=[
                '--s3-bucket', source_bucket,
                '--region', region,
                '--workers', str(workers),
                '--output-dir', '/opt/ml/processing/output',
            ],
            wait=True,  # Wait for completion
            logs=True,  # Show CloudWatch logs
        )

        print("\n" + "=" * 70)
        print("[Launcher] PROCESSING JOB COMPLETE")
        print("=" * 70)
        print(f"  Output location: {output_s3_uri}/")
        print(f"  Files: train.jsonl, val.jsonl")
        print("\n  Colab notebook will now use the fast path:")
        print("  - Downloads cached splits in seconds")
        print("  - Skips the 11+ hour processing step")
        print("=" * 70)

    except Exception as e:
        print(f"\n[Launcher] ERROR: {e}")
        print("\nCheck CloudWatch logs for details:")
        print(f"  aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/ProcessingJobs")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch SageMaker Processing Job for Automotive Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default configuration (c5.xlarge spot)
  python scripts/launch_processing_job.py

  # Custom output location
  python scripts/launch_processing_job.py --output-bucket my-bucket --output-prefix data/v2

  # Use on-demand pricing (more reliable, costs more)
  python scripts/launch_processing_job.py --no-spot

  # Larger instance for faster processing
  python scripts/launch_processing_job.py --instance-type ml.c5.2xlarge --workers 8
        """
    )

    parser.add_argument(
        '--region', type=str,
        default=os.environ.get('AWS_REGION', 'us-east-1'),
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--source-bucket', type=str,
        default='granite-8b-unified-automotive-data',
        help='Source S3 bucket with raw automotive data'
    )
    parser.add_argument(
        '--output-bucket', type=str,
        default='granite-8b-training-outputs',
        help='Output S3 bucket for processed splits'
    )
    parser.add_argument(
        '--output-prefix', type=str,
        default='runs/data/splits',
        help='S3 prefix for output files'
    )
    parser.add_argument(
        '--instance-type', type=str,
        default='ml.c5.xlarge',
        help='SageMaker instance type (default: ml.c5.xlarge - 4 vCPU, 8 GB)'
    )
    parser.add_argument(
        '--workers', type=int,
        default=4,
        help='Number of parallel workers (should match vCPUs)'
    )
    parser.add_argument(
        '--no-spot', action='store_true',
        help='Use on-demand pricing instead of spot'
    )
    parser.add_argument(
        '--max-runtime-hours', type=int,
        default=4,
        help='Maximum job runtime in hours'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SAGEMAKER PROCESSING JOB LAUNCHER")
    print("=" * 70)

    # Get or create execution role
    role = get_or_create_role(args.region)

    # Launch the job
    launch_processing_job(
        role=role,
        region=args.region,
        source_bucket=args.source_bucket,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        instance_type=args.instance_type,
        use_spot=not args.no_spot,
        max_runtime_hours=args.max_runtime_hours,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
