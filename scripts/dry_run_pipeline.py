#!/usr/bin/env python3
"""
Dry Run Pipeline Test for Granite-8B Fine-tuning

This script tests the entire pipeline without using expensive resources:
1. AWS connectivity and permissions
2. S3 bucket access and data availability
3. Bedrock model access
4. Training script configuration
5. Data pipeline processing
6. SageMaker role verification

Run this BEFORE launching actual training to catch issues early.

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PipelineValidator:
    """Validate the entire fine-tuning pipeline"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        self.errors = []

    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Could not load config: {e}")
            return {}

    def _log_result(self, test_name: str, success: bool, message: str):
        """Log test result"""
        status = "[PASS]" if success else "[FAIL]"
        result = f"{status} {test_name}: {message}"
        self.results.append(result)
        if not success:
            self.errors.append(result)
        print(result)

    def test_aws_credentials(self) -> bool:
        """Test AWS credentials are configured"""
        print("\n" + "="*60)
        print("Testing AWS Credentials")
        print("="*60)

        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()

            self._log_result(
                "AWS Credentials",
                True,
                f"Account: {identity['Account']}, User: {identity['Arn']}"
            )
            return True

        except NoCredentialsError:
            self._log_result(
                "AWS Credentials",
                False,
                "No AWS credentials found. Check .env file or AWS config."
            )
            return False

        except Exception as e:
            self._log_result("AWS Credentials", False, str(e))
            return False

    def test_s3_bucket_access(self) -> bool:
        """Test S3 bucket access"""
        print("\n" + "="*60)
        print("Testing S3 Bucket Access")
        print("="*60)

        bucket = self.config.get('aws', {}).get('s3', {}).get('bucket_name', '')
        region = self.config.get('aws', {}).get('region', 'us-east-1')

        if not bucket:
            self._log_result("S3 Bucket Config", False, "No bucket configured in config.yaml")
            return False

        try:
            s3 = boto3.client('s3', region_name=region)

            # Test list objects
            response = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
            count = response.get('KeyCount', 0)

            self._log_result(
                "S3 Bucket Access",
                True,
                f"Bucket: {bucket}, Objects found: {count}"
            )

            # Check for key prefixes
            prefixes_to_check = ['tsn_data/', 'advanced_academic/']
            for prefix in prefixes_to_check:
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                found = response.get('KeyCount', 0) > 0
                self._log_result(
                    f"S3 Prefix '{prefix}'",
                    found,
                    "Found" if found else "Not found"
                )

            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            self._log_result("S3 Bucket Access", False, f"{error_code}: {e}")
            return False

    def test_bedrock_access(self) -> bool:
        """Test Amazon Bedrock access"""
        print("\n" + "="*60)
        print("Testing Amazon Bedrock Access")
        print("="*60)

        region = self.config.get('aws', {}).get('region', 'us-east-1')
        model_id = self.config.get('distillation', {}).get(
            'teacher_model',
            'anthropic.claude-3-5-sonnet-20241022-v2:0'
        )

        try:
            bedrock = boto3.client('bedrock-runtime', region_name=region)

            # Simple test prompt
            test_prompt = "Say 'Hello' in one word."

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": test_prompt}]
            }

            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )

            result = json.loads(response['body'].read())
            output = result.get('content', [{}])[0].get('text', '')

            self._log_result(
                "Bedrock Model Access",
                True,
                f"Model: {model_id}, Response: '{output[:50]}...'"
            )
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                self._log_result(
                    "Bedrock Model Access",
                    False,
                    f"Access denied. Enable model access in Bedrock console for: {model_id}"
                )
            else:
                self._log_result("Bedrock Model Access", False, f"{error_code}: {e}")
            return False

        except Exception as e:
            self._log_result("Bedrock Model Access", False, str(e))
            return False

    def test_iam_role(self) -> bool:
        """Test SageMaker IAM role exists"""
        print("\n" + "="*60)
        print("Testing IAM Role")
        print("="*60)

        role_name = self.config.get('aws', {}).get('iam', {}).get(
            'role_name',
            'SageMakerGranite8BFineTuningRole'
        )

        try:
            iam = boto3.client('iam')
            response = iam.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']

            self._log_result(
                "IAM Role",
                True,
                f"Role: {role_name}, ARN: {role_arn}"
            )
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchEntity':
                self._log_result(
                    "IAM Role",
                    False,
                    f"Role '{role_name}' not found. Deploy CloudFormation stack first."
                )
            else:
                self._log_result("IAM Role", False, f"{error_code}: {e}")
            return False

    def test_huggingface_token(self) -> bool:
        """Test HuggingFace token is configured"""
        print("\n" + "="*60)
        print("Testing HuggingFace Token")
        print("="*60)

        hf_token = os.environ.get('HF_TOKEN', '')

        if hf_token and hf_token.startswith('hf_'):
            self._log_result(
                "HuggingFace Token",
                True,
                f"Token configured: {hf_token[:10]}..."
            )
            return True
        else:
            self._log_result(
                "HuggingFace Token",
                False,
                "HF_TOKEN not found or invalid in .env"
            )
            return False

    def test_config_validity(self) -> bool:
        """Test configuration file validity"""
        print("\n" + "="*60)
        print("Testing Configuration")
        print("="*60)

        required_sections = ['model', 'qlora', 'training', 'aws', 'distillation']
        all_valid = True

        for section in required_sections:
            if section in self.config:
                self._log_result(f"Config Section '{section}'", True, "Found")
            else:
                self._log_result(f"Config Section '{section}'", False, "Missing")
                all_valid = False

        # Validate model name
        model_name = self.config.get('model', {}).get('name', '')
        if 'granite' in model_name.lower():
            self._log_result("Model Name", True, model_name)
        else:
            self._log_result(
                "Model Name",
                False,
                f"Expected Granite model, got: {model_name}"
            )
            all_valid = False

        return all_valid

    def test_local_files(self) -> bool:
        """Test local training files exist"""
        print("\n" + "="*60)
        print("Testing Local Files")
        print("="*60)

        required_files = [
            'training/train_granite_qlora.py',
            'scripts/generate_teacher_outputs.py',
            'scripts/prepare_automotive_data.py',
            'scripts/run_training_job.py',
            'docker/Dockerfile',
            'requirements.txt',
            'config.yaml'
        ]

        all_found = True
        for file_path in required_files:
            exists = Path(file_path).exists()
            self._log_result(f"File '{file_path}'", exists, "Found" if exists else "Missing")
            if not exists:
                all_found = False

        return all_found

    def test_training_script_syntax(self) -> bool:
        """Test training script has valid Python syntax"""
        print("\n" + "="*60)
        print("Testing Training Script Syntax")
        print("="*60)

        script_path = 'training/train_granite_qlora.py'

        try:
            with open(script_path, 'r') as f:
                code = f.read()

            compile(code, script_path, 'exec')
            self._log_result("Training Script Syntax", True, "Valid Python")
            return True

        except SyntaxError as e:
            self._log_result(
                "Training Script Syntax",
                False,
                f"Syntax error at line {e.lineno}: {e.msg}"
            )
            return False

        except Exception as e:
            self._log_result("Training Script Syntax", False, str(e))
            return False

    def test_data_pipeline_import(self) -> bool:
        """Test data pipeline can be imported"""
        print("\n" + "="*60)
        print("Testing Data Pipeline Import")
        print("="*60)

        try:
            sys.path.insert(0, 'scripts')
            import prepare_automotive_data
            self._log_result("Data Pipeline Import", True, "Successfully imported")
            return True

        except ImportError as e:
            self._log_result(
                "Data Pipeline Import",
                False,
                f"Import error: {e}"
            )
            return False

        except Exception as e:
            self._log_result("Data Pipeline Import", False, str(e))
            return False

    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        print("\n" + "="*70)
        print(" GRANITE-8B FINE-TUNING PIPELINE DRY RUN")
        print(" Testing all components before actual training")
        print("="*70)

        tests = [
            ("AWS Credentials", self.test_aws_credentials),
            ("S3 Bucket Access", self.test_s3_bucket_access),
            ("Bedrock Access", self.test_bedrock_access),
            ("IAM Role", self.test_iam_role),
            ("HuggingFace Token", self.test_huggingface_token),
            ("Configuration", self.test_config_validity),
            ("Local Files", self.test_local_files),
            ("Training Script", self.test_training_script_syntax),
            ("Data Pipeline", self.test_data_pipeline_import),
        ]

        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self._log_result(test_name, False, f"Unexpected error: {e}")

        # Summary
        print("\n" + "="*70)
        print(" DRY RUN SUMMARY")
        print("="*70)

        passed = len(self.results) - len(self.errors)
        total = len(self.results)

        print(f"\n  Tests Passed: {passed}/{total}")
        print(f"  Tests Failed: {len(self.errors)}/{total}")

        if self.errors:
            print("\n  FAILURES:")
            for error in self.errors:
                print(f"    {error}")

            print("\n  ACTION REQUIRED: Fix the above issues before running training.")
            return False
        else:
            print("\n  All tests passed! Pipeline is ready for training.")
            return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dry run test for Granite-8B fine-tuning pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--skip-bedrock',
        action='store_true',
        help='Skip Bedrock test (saves API cost)'
    )

    args = parser.parse_args()

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    validator = PipelineValidator(config_path=args.config)

    if args.skip_bedrock:
        print("\n[INFO] Skipping Bedrock test (--skip-bedrock)")
        validator.test_bedrock_access = lambda: validator._log_result(
            "Bedrock Access", True, "Skipped"
        )

    success = validator.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
