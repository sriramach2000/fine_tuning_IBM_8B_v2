"""
Shared fixtures for Granite-8B Fine-Tuning Test Suite

Provides mock clients, sample data, and test utilities for:
- Bedrock API testing
- S3 operations testing
- Training callback testing
- Data pipeline testing
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Bedrock Fixtures
# =============================================================================

@pytest.fixture
def mock_bedrock_response():
    """Standard successful Bedrock response"""
    return {
        'content': [{'text': 'Generated code response'}],
        'usage': {
            'input_tokens': 100,
            'output_tokens': 500
        },
        'model': 'claude-sonnet-4-5-20250929'
    }


@pytest.fixture
def mock_bedrock_empty_response():
    """Empty Bedrock response (edge case)"""
    return {
        'content': [],
        'usage': {'input_tokens': 0, 'output_tokens': 0}
    }


@pytest.fixture
def mock_bedrock_client(mock_bedrock_response):
    """Mock Bedrock runtime client"""
    client = MagicMock()

    # Create a mock response body
    response_body = MagicMock()
    response_body.read.return_value = json.dumps(mock_bedrock_response).encode()

    client.invoke_model.return_value = {'body': response_body}
    client.meta.events.register = MagicMock()

    return client


@pytest.fixture
def mock_throttling_client():
    """Mock Bedrock client that throttles first N requests"""
    from botocore.exceptions import ClientError

    client = MagicMock()
    call_count = [0]

    def throttle_then_succeed(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # Throttle first 2 calls
            raise ClientError(
                {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                'InvokeModel'
            )
        # Success on 3rd call
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            'content': [{'text': 'Success after retry'}],
            'usage': {'input_tokens': 50, 'output_tokens': 200}
        }).encode()
        return {'body': response_body}

    client.invoke_model.side_effect = throttle_then_succeed
    client.meta.events.register = MagicMock()

    return client


# =============================================================================
# S3 Fixtures
# =============================================================================

@pytest.fixture
def sample_code_files():
    """Sample automotive code files for testing"""
    return {
        'tsn_frame.c': '''
#include <stdint.h>
#include "tsn_frame.h"

int tsn_frame_init(tsn_frame_t *frame) {
    if (!frame) return -1;
    frame->priority = 0;
    frame->vlan_id = 0;
    return 0;
}
''',
        'avb_stream.c': '''
#include "avb_stream.h"

int avb_stream_create(avb_stream_t *stream, uint32_t stream_id) {
    stream->id = stream_id;
    stream->bandwidth = 0;
    return 0;
}
''',
        'empty_file.c': '',
        'malformed.c': b'\xff\xfe Invalid UTF-8'.decode('latin-1'),
    }


@pytest.fixture
def mock_s3_client(sample_code_files):
    """Mock S3 client with sample files"""
    client = MagicMock()

    # Mock list_objects_v2
    objects = [{'Key': f'tsn_data/{name}'} for name in sample_code_files.keys()]
    client.list_objects_v2.return_value = {
        'Contents': objects,
        'KeyCount': len(objects)
    }

    return client


# =============================================================================
# Training Fixtures
# =============================================================================

@pytest.fixture
def mock_trainer_state():
    """Mock HuggingFace TrainerState"""
    state = MagicMock()
    state.global_step = 100
    state.epoch = 1.0
    state.log_history = []
    return state


@pytest.fixture
def mock_trainer_control():
    """Mock HuggingFace TrainerControl"""
    control = MagicMock()
    control.should_training_stop = False
    control.should_save = False
    return control


@pytest.fixture
def eval_metrics_improving():
    """Eval metrics showing improvement"""
    return [
        {'eval_loss': 2.5},
        {'eval_loss': 2.2},
        {'eval_loss': 1.9},
        {'eval_loss': 1.6},
    ]


@pytest.fixture
def eval_metrics_plateau():
    """Eval metrics showing plateau (no improvement)"""
    return [
        {'eval_loss': 2.0},
        {'eval_loss': 2.0},
        {'eval_loss': 2.01},
        {'eval_loss': 2.02},
    ]


@pytest.fixture
def eval_metrics_nan():
    """Eval metrics with NaN"""
    return {'loss': float('nan')}


@pytest.fixture
def eval_metrics_inf():
    """Eval metrics with Inf"""
    return {'loss': float('inf')}


# =============================================================================
# Sample Prompts
# =============================================================================

@pytest.fixture
def sample_prompts():
    """Sample automotive code generation prompts"""
    return [
        {
            'id': 'tsn_001',
            'prompt': 'Generate C code for TSN Time-Aware Shaper initialization.'
        },
        {
            'id': 'avb_001',
            'prompt': 'Generate C code for AVB Stream Reservation Protocol.'
        },
        {
            'id': 'eth_001',
            'prompt': 'Generate C code for Ethernet frame parsing with VLAN tags.'
        },
    ]


@pytest.fixture
def sample_prompts_with_empty():
    """Sample prompts including empty prompt (edge case)"""
    return [
        {'id': 'valid_001', 'prompt': 'Generate TSN code'},
        {'id': 'empty_001', 'prompt': ''},
        {'id': 'valid_002', 'prompt': 'Generate AVB code'},
    ]


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'name': 'ibm-granite/granite-8b-code-instruct-128k',
            'max_seq_length': 4096,
        },
        'training': {
            'num_epochs': 5,
            'per_device_train_batch_size': 2,
            'learning_rate': 1e-4,
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.0,
        },
        'distillation': {
            'teacher_model': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            'max_teacher_tokens': 2048,
        },
        'aws': {
            'region': 'us-east-1',
            's3': {'bucket_name': 'test-bucket'},
        },
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs"""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables"""
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test-key-id')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'test-secret-key')
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    monkeypatch.setenv('AMAZON_BEDROCK_MODEL_API_KEY', 'test-api-key')
    monkeypatch.setenv('HF_TOKEN', 'hf_test_token')
