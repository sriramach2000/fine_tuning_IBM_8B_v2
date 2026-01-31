"""
Cloud Test Fixtures

Shared fixtures for cloud-native testing including:
- AWS client mocking for offline tests
- Real AWS client setup for integration tests
- Common test data and error factories
"""

import os
import json
import pytest
from unittest.mock import MagicMock
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


# =============================================================================
# Pytest Marks Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom pytest marks"""
    config.addinivalue_line("markers", "cloud: tests requiring AWS connectivity")
    config.addinivalue_line("markers", "s3: S3-specific tests")
    config.addinivalue_line("markers", "bedrock: Bedrock-specific tests")
    config.addinivalue_line("markers", "integration: end-to-end integration tests")
    config.addinivalue_line("markers", "slow: tests that take longer to run")
    config.addinivalue_line("markers", "offline: tests that work without AWS")
    config.addinivalue_line("markers", "concurrency: tests for parallel/concurrent operations")
    config.addinivalue_line("markers", "resilience: tests for retry/backoff behavior")
    config.addinivalue_line("markers", "gpu: tests requiring GPU/CUDA hardware")
    config.addinivalue_line("markers", "sagemaker: tests requiring SageMaker environment")


# =============================================================================
# Environment Detection
# =============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """Check if CUDA GPU is available"""
    import unittest.mock
    try:
        import torch
        if isinstance(torch.cuda, unittest.mock.MagicMock):
            return False
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def sagemaker_environment():
    """Check if running inside SageMaker"""
    return os.environ.get('SM_TRAINING_ENV') is not None


@pytest.fixture(scope="session")
def aws_credentials_available():
    """Check if AWS credentials are available"""
    try:
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def s3_bucket():
    """Get S3 bucket name from environment"""
    return os.environ.get('S3_BUCKET', 'granite-8b-unified-automotive-data')


@pytest.fixture(scope="session")
def aws_region():
    """Get AWS region from environment"""
    return os.environ.get('AWS_REGION', 'us-east-1')


# =============================================================================
# Real AWS Clients (for integration tests)
# =============================================================================

@pytest.fixture
def real_s3_client(aws_region):
    """Real S3 client for integration tests"""
    return boto3.client('s3', region_name=aws_region)


@pytest.fixture
def real_bedrock_client(aws_region):
    """Real Bedrock client for integration tests"""
    return boto3.client('bedrock-runtime', region_name=aws_region)


# =============================================================================
# Mock AWS Clients (for offline/unit tests)
# =============================================================================

@pytest.fixture
def mock_s3_client():
    """Mock S3 client with common responses"""
    client = MagicMock()

    # Default: successful bucket access
    client.head_bucket.return_value = {}

    # Default: some objects in bucket
    client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'data/sample1.jsonl', 'Size': 1024},
            {'Key': 'data/sample2.jsonl', 'Size': 2048},
        ],
        'KeyCount': 2
    }

    return client


@pytest.fixture
def mock_bedrock_client():
    """Mock Bedrock client with successful response"""
    client = MagicMock()

    response_body = MagicMock()
    response_body.read.return_value = json.dumps({
        'content': [{'text': 'Generated code response'}],
        'usage': {'input_tokens': 100, 'output_tokens': 500}
    }).encode()

    client.invoke_model.return_value = {'body': response_body}
    client.meta.events.register = MagicMock()

    return client


# =============================================================================
# Error Response Factories
# =============================================================================

@pytest.fixture
def s3_access_denied_error():
    """Factory for S3 AccessDenied error"""
    def _create():
        return ClientError(
            {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
            'GetObject'
        )
    return _create


@pytest.fixture
def s3_no_such_key_error():
    """Factory for S3 NoSuchKey error"""
    def _create(key='missing-key'):
        return ClientError(
            {'Error': {'Code': 'NoSuchKey', 'Message': f'Key {key} not found'}},
            'GetObject'
        )
    return _create


@pytest.fixture
def bedrock_throttling_error():
    """Factory for Bedrock ThrottlingException"""
    def _create():
        return ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'InvokeModel'
        )
    return _create


@pytest.fixture
def bedrock_validation_error():
    """Factory for Bedrock ValidationException"""
    def _create(message='Invalid model ID'):
        return ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': message}},
            'InvokeModel'
        )
    return _create


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_automotive_code():
    """Sample automotive C code for testing"""
    return {
        'good_tsn_code': '''
/**
 * @brief Initialize TSN frame structure
 * @param frame Pointer to TSN frame
 * @return 0 on success, -1 on error
 */
int tsn_frame_init(tsn_frame_t *frame) {
    if (!frame) return -1;

    frame->priority = 0;
    frame->vlan_id = 0;
    frame->timestamp = 0ULL;

    return 0;
}
''',
        'bad_code_with_goto': '''
void process_data(void) {
    goto error;
    error:
    return;
}
''',
        'code_with_recursion': '''
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
''',
        'empty_code': '',
        'malformed_code': '\xff\xfe Invalid UTF-8',
        'very_long_code': 'int x = 0;\n' * 10000,
        'unicode_code': '''
// Unicode comment: café résumé
int process_café(void) { return 0; }
''',
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts for batch testing"""
    return [
        {'id': 'tsn_001', 'prompt': 'Generate TSN frame initialization code'},
        {'id': 'avb_001', 'prompt': 'Generate AVB stream reservation code'},
        {'id': 'eth_001', 'prompt': 'Generate Ethernet frame parser'},
    ]
