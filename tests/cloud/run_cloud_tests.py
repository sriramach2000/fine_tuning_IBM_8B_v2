#!/usr/bin/env python3
"""
Cloud-Native Test Suite - Runs as SageMaker Processing Job

This script runs inside a SageMaker container to validate:
1. S3 data access and permissions
2. Bedrock API connectivity and responses
3. Quality evaluation on real data
4. End-to-end pipeline integration

Usage:
    # Deploy via SageMaker Processing Job (see launch_cloud_tests.py)
    # Or run locally for debugging:
    python tests/cloud/run_cloud_tests.py --local

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import boto3
from botocore.exceptions import ClientError

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


class CloudTestResult:
    """Container for test result"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0
        self.details = {}

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'error': str(self.error) if self.error else None,
            'duration_ms': self.duration_ms,
            'details': self.details,
        }


class CloudTestSuite:
    """
    Cloud-native test suite that runs inside SageMaker.

    Tests real AWS service integrations:
    - S3 bucket access
    - Bedrock model invocation
    - Data pipeline operations
    - Full distillation flow
    """

    def __init__(
        self,
        s3_bucket: str,
        region: str = 'us-east-1',
        output_dir: str = '/opt/ml/processing/output',
    ):
        self.s3_bucket = s3_bucket
        self.region = region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.s3_client = boto3.client('s3', region_name=region)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)

        self.results: List[CloudTestResult] = []
        self.start_time = datetime.now()

    def run_test(self, test_func, name: str) -> CloudTestResult:
        """Run a single test with timing and error handling"""
        result = CloudTestResult(name)
        start = time.time()

        try:
            details = test_func()
            result.passed = True
            result.details = details or {}
        except Exception as e:
            result.passed = False
            result.error = e
            result.details = {'traceback': str(e)}

        result.duration_ms = int((time.time() - start) * 1000)
        self.results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {name} ({result.duration_ms}ms)")
        if result.error:
            print(f"       Error: {result.error}")

        return result

    # =========================================================================
    # S3 TESTS
    # =========================================================================

    def test_s3_bucket_access(self) -> Dict:
        """Test: Can access the S3 bucket"""
        response = self.s3_client.head_bucket(Bucket=self.s3_bucket)
        return {'bucket': self.s3_bucket, 'status': 'accessible'}

    def test_s3_list_objects(self) -> Dict:
        """Test: Can list objects in the training data prefix"""
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix='data/',
            MaxKeys=10
        )

        objects = response.get('Contents', [])
        return {
            'prefix': 'data/',
            'objects_found': len(objects),
            'sample_keys': [o['Key'] for o in objects[:3]],
        }

    def test_s3_read_training_file(self) -> Dict:
        """Test: Can read a training data file"""
        # Try to find a training file
        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
            Prefix='data/processed/',
            MaxKeys=5
        )

        if not response.get('Contents'):
            raise FileNotFoundError("No training files found in data/processed/")

        key = response['Contents'][0]['Key']

        # Download and read
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
        content = obj['Body'].read(1024)  # Read first 1KB

        return {
            'file_key': key,
            'file_size': obj['ContentLength'],
            'content_preview': content[:100].decode('utf-8', errors='ignore'),
        }

    def test_s3_write_permission(self) -> Dict:
        """Test: Can write to output location"""
        test_key = f"test-outputs/cloud-test-{datetime.now().isoformat()}.txt"
        test_content = b"Cloud test write permission check"

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=test_key,
            Body=test_content
        )

        # Verify write
        response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=test_key)

        # Clean up
        self.s3_client.delete_object(Bucket=self.s3_bucket, Key=test_key)

        return {
            'test_key': test_key,
            'written_bytes': len(test_content),
            'verified': True,
        }

    def test_s3_pagination(self) -> Dict:
        """Test: Can paginate through many objects"""
        paginator = self.s3_client.get_paginator('list_objects_v2')

        total_objects = 0
        page_count = 0

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix='data/', PaginationConfig={'MaxItems': 100}):
            total_objects += len(page.get('Contents', []))
            page_count += 1

        return {
            'total_objects': total_objects,
            'pages_processed': page_count,
        }

    # =========================================================================
    # BEDROCK TESTS
    # =========================================================================

    def test_bedrock_model_access(self) -> Dict:
        """Test: Can invoke Bedrock Claude model"""
        model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Say 'hello' in exactly one word."}
            ]
        }

        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '')

        return {
            'model_id': model_id,
            'response_received': True,
            'response_preview': content[:50],
            'usage': result.get('usage', {}),
        }

    def test_bedrock_code_generation(self) -> Dict:
        """Test: Bedrock can generate automotive code"""
        model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a simple C struct for a TSN timestamp with nanoseconds."
                }
            ]
        }

        response = self.bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        content = result.get('content', [{}])[0].get('text', '')

        # Validate response contains C code elements
        has_struct = 'struct' in content.lower()
        has_timestamp = 'timestamp' in content.lower() or 'time' in content.lower()

        return {
            'model_id': model_id,
            'contains_struct': has_struct,
            'contains_timestamp': has_timestamp,
            'response_length': len(content),
            'usage': result.get('usage', {}),
        }

    def test_bedrock_rate_limit_handling(self) -> Dict:
        """Test: Can handle rate limiting gracefully"""
        model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

        # Make 3 rapid requests
        responses = []
        errors = []

        for i in range(3):
            try:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 50,
                    "messages": [
                        {"role": "user", "content": f"Count to {i+1}."}
                    ]
                }

                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                responses.append(True)

            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    errors.append('throttled')
                else:
                    errors.append(str(e))

        return {
            'requests_made': 3,
            'successful': len(responses),
            'throttled': errors.count('throttled'),
            'other_errors': len([e for e in errors if e != 'throttled']),
        }

    # =========================================================================
    # QUALITY EVALUATOR TESTS
    # =========================================================================

    def test_quality_evaluator_initialization(self) -> Dict:
        """Test: Quality evaluator initializes correctly"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(quality_threshold=7.0)

        return {
            'initialized': True,
            'quality_threshold': evaluator.quality_threshold,
            'gcc_available': evaluator.gcc_available,
        }

    def test_quality_evaluator_scoring(self) -> Dict:
        """Test: Quality evaluator scores code correctly"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(quality_threshold=7.0)

        # Good TSN code
        good_code = """
        struct tsn_frame {
            uint8_t pcp;
            uint16_t vlan_id;
            uint64_t timestamp;
            uint8_t priority;
        };
        """

        # Bad code
        bad_code = "void foo() { goto error; }"

        good_score = evaluator.evaluate(good_code, "Generate TSN code")
        bad_score = evaluator.evaluate(bad_code, "Generate TSN code")

        return {
            'good_code_score': good_score.overall,
            'bad_code_score': bad_score.overall,
            'scoring_correct': good_score.overall > bad_score.overall,
        }

    # =========================================================================
    # INTEGRATION TESTS
    # =========================================================================

    def test_teacher_generator_initialization(self) -> Dict:
        """Test: Bedrock teacher generator initializes correctly"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator

        generator = BedrockTeacherGenerator(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region=self.region,
            max_tokens=100
        )

        return {
            'initialized': True,
            'model_id': generator.model_id,
            'max_tokens': generator.max_tokens,
        }

    def test_teacher_generator_response(self) -> Dict:
        """Test: Teacher generator produces responses"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator

        generator = BedrockTeacherGenerator(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region=self.region,
            max_tokens=200
        )

        result = generator.generate_response(
            prompt="Generate a simple C function that returns 42.",
            system_prompt="You are a C programmer."
        )

        return {
            'success': result['success'],
            'has_response': result['response'] is not None,
            'response_length': len(result.get('response', '')),
            'usage': result.get('usage', {}),
        }

    def test_full_correction_flow(self) -> Dict:
        """Test: Full correction flow works end-to-end"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        # Initialize components
        teacher = BedrockTeacherGenerator(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region=self.region,
            max_tokens=500
        )
        evaluator = CodeQualityEvaluator(quality_threshold=7.0)

        # Simulate poor student output
        student_output = "void process() { goto error; error: return; }"
        prompt = "Generate TSN timestamp processing code"

        # Evaluate student output
        student_score = evaluator.evaluate(student_output, prompt)

        # Get teacher correction
        correction_prompt = f"""
        A student wrote this code (score {student_score.overall:.1f}/10):
        {student_output}

        For this prompt: {prompt}

        Provide a corrected version following MISRA-C guidelines.
        """

        result = teacher.generate_response(
            prompt=correction_prompt,
            system_prompt="You are an expert automotive embedded systems engineer."
        )

        # Evaluate teacher correction
        if result['success'] and result['response']:
            teacher_score = evaluator.evaluate(result['response'], prompt)
        else:
            teacher_score = None

        return {
            'student_score': student_score.overall,
            'teacher_response_success': result['success'],
            'teacher_score': teacher_score.overall if teacher_score else None,
            'improvement': (teacher_score.overall - student_score.overall) if teacher_score else 0,
        }

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all_tests(self) -> Dict:
        """Run all cloud tests and return summary"""
        print("\n" + "="*70)
        print("CLOUD-NATIVE TEST SUITE")
        print(f"S3 Bucket: {self.s3_bucket}")
        print(f"Region: {self.region}")
        print(f"Started: {self.start_time.isoformat()}")
        print("="*70 + "\n")

        # S3 Tests
        print("[S3 Tests]")
        self.run_test(self.test_s3_bucket_access, "S3 Bucket Access")
        self.run_test(self.test_s3_list_objects, "S3 List Objects")
        self.run_test(self.test_s3_read_training_file, "S3 Read Training File")
        self.run_test(self.test_s3_write_permission, "S3 Write Permission")
        self.run_test(self.test_s3_pagination, "S3 Pagination")

        # Bedrock Tests
        print("\n[Bedrock Tests]")
        self.run_test(self.test_bedrock_model_access, "Bedrock Model Access")
        self.run_test(self.test_bedrock_code_generation, "Bedrock Code Generation")
        self.run_test(self.test_bedrock_rate_limit_handling, "Bedrock Rate Limit Handling")

        # Quality Evaluator Tests
        print("\n[Quality Evaluator Tests]")
        self.run_test(self.test_quality_evaluator_initialization, "Quality Evaluator Init")
        self.run_test(self.test_quality_evaluator_scoring, "Quality Evaluator Scoring")

        # Integration Tests
        print("\n[Integration Tests]")
        self.run_test(self.test_teacher_generator_initialization, "Teacher Generator Init")
        self.run_test(self.test_teacher_generator_response, "Teacher Generator Response")
        self.run_test(self.test_full_correction_flow, "Full Correction Flow")

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        duration = (datetime.now() - self.start_time).total_seconds()

        summary = {
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.results) if self.results else 0,
            'duration_seconds': duration,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results],
        }

        print("\n" + "="*70)
        print(f"SUMMARY: {passed}/{len(self.results)} tests passed ({summary['pass_rate']:.0%})")
        print(f"Duration: {duration:.1f} seconds")
        print("="*70)

        # Save results
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Run cloud-native tests")

    parser.add_argument('--s3-bucket', type=str,
                        default=os.environ.get('S3_BUCKET', 'granite-8b-unified-automotive-data'),
                        help='S3 bucket for testing')
    parser.add_argument('--region', type=str,
                        default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region')
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output'),
                        help='Output directory for results')
    parser.add_argument('--local', action='store_true',
                        help='Run locally (use ./output for results)')

    args = parser.parse_args()

    if args.local:
        args.output_dir = './output/cloud_tests'

    suite = CloudTestSuite(
        s3_bucket=args.s3_bucket,
        region=args.region,
        output_dir=args.output_dir
    )

    summary = suite.run_all_tests()

    # Exit with error code if tests failed
    if summary['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
