"""
Tests for Bedrock Teacher Generator

Tests:
1. test_rate_limit_backoff - Verify exponential backoff on throttling
2. test_max_retries_exceeded - Verify failure after max retries
3. test_empty_response_handling - Handle empty API response
4. test_token_counting - Verify input/output token tracking
5. test_authentication_failure - Handle auth errors gracefully
6. test_parallel_workers - ThreadPoolExecutor correctness
"""

import os
import sys
import json
import time
import pytest
from unittest.mock import MagicMock, patch, call
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBedrockGenerator:
    """Test suite for BedrockTeacherGenerator"""

    @pytest.fixture
    def generator_class(self):
        """Import the generator class"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator
        return BedrockTeacherGenerator

    # =========================================================================
    # Test 1: Rate Limit Backoff
    # =========================================================================
    def test_rate_limit_backoff(self, generator_class, mock_env_vars):
        """Verify exponential backoff timing on rate limiting"""
        from botocore.exceptions import ClientError

        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            # Setup mock client
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Track call times
            call_times = []

            def throttle_then_succeed(*args, **kwargs):
                call_times.append(time.time())
                if len(call_times) <= 2:
                    raise ClientError(
                        {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                        'InvokeModel'
                    )
                # Success on 3rd call
                response_body = MagicMock()
                response_body.read.return_value = json.dumps({
                    'content': [{'text': 'Success'}],
                    'usage': {'input_tokens': 50, 'output_tokens': 200}
                }).encode()
                return {'body': response_body}

            mock_client.invoke_model.side_effect = throttle_then_succeed
            mock_client.meta.events.register = MagicMock()

            # Create generator with short delays for testing
            generator = generator_class(
                model_id='test-model',
                base_delay=0.1,  # 100ms for faster tests
                max_retries=5
            )

            # Make request
            with patch('scripts.generate_teacher_outputs.time.sleep') as mock_sleep:
                result = generator.generate_response("Test prompt")

            # Verify backoff was called with increasing delays
            assert result['success'] is True
            # Should have called sleep twice (for 2 throttle errors)
            assert mock_sleep.call_count == 2
            # First call: 0.1 * 2^0 = 0.1
            # Second call: 0.1 * 2^1 = 0.2
            calls = mock_sleep.call_args_list
            assert calls[0][0][0] == pytest.approx(0.1, rel=0.1)
            assert calls[1][0][0] == pytest.approx(0.2, rel=0.1)

    # =========================================================================
    # Test 2: Max Retries Exceeded
    # =========================================================================
    def test_max_retries_exceeded(self, generator_class, mock_env_vars):
        """Verify failure after max retries exhausted"""
        from botocore.exceptions import ClientError

        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Always throttle
            mock_client.invoke_model.side_effect = ClientError(
                {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                'InvokeModel'
            )
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(
                model_id='test-model',
                base_delay=0.01,
                max_retries=3
            )

            with patch('scripts.generate_teacher_outputs.time.sleep'):
                result = generator.generate_response("Test prompt")

            # Should fail after max retries
            assert result['success'] is False
            assert 'Max retries' in result['error']
            assert mock_client.invoke_model.call_count == 3

    # =========================================================================
    # Test 3: Empty Response Handling
    # =========================================================================
    def test_empty_response_handling(self, generator_class, mock_env_vars):
        """Handle empty API response gracefully"""
        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Return empty content
            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                'content': [],
                'usage': {'input_tokens': 10, 'output_tokens': 0}
            }).encode()
            mock_client.invoke_model.return_value = {'body': response_body}
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(model_id='test-model')
            result = generator.generate_response("Test prompt")

            assert result['success'] is False
            assert result['response'] is None
            assert 'Empty response' in result['error']

    # =========================================================================
    # Test 4: Token Counting
    # =========================================================================
    def test_token_counting(self, generator_class, mock_env_vars, sample_prompts):
        """Verify input/output token tracking in batch processing"""
        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Return consistent token counts
            def make_response(*args, **kwargs):
                response_body = MagicMock()
                response_body.read.return_value = json.dumps({
                    'content': [{'text': 'Generated code'}],
                    'usage': {'input_tokens': 100, 'output_tokens': 500}
                }).encode()
                return {'body': response_body}

            mock_client.invoke_model.side_effect = make_response
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(model_id='test-model')

            with patch('scripts.generate_teacher_outputs.time.sleep'):
                results = generator.generate_batch(
                    prompts=sample_prompts,
                    max_workers=1
                )

            # Verify all prompts processed
            assert len(results) == 3

            # Verify token counts recorded
            total_output_tokens = sum(r.get('usage', {}).get('output_tokens', 0) for r in results)
            assert total_output_tokens == 1500  # 3 prompts × 500 tokens

    # =========================================================================
    # Test 5: Authentication Failure
    # =========================================================================
    def test_authentication_failure(self, generator_class, mock_env_vars):
        """Handle auth errors gracefully without retrying"""
        from botocore.exceptions import ClientError

        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Auth error (should not retry)
            mock_client.invoke_model.side_effect = ClientError(
                {'Error': {'Code': 'AccessDeniedException', 'Message': 'Access denied'}},
                'InvokeModel'
            )
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(model_id='test-model', max_retries=5)
            result = generator.generate_response("Test prompt")

            # Should fail immediately without retrying
            assert result['success'] is False
            assert 'AccessDeniedException' in result['error']
            # Only 1 call - no retries for auth errors
            assert mock_client.invoke_model.call_count == 1

    # =========================================================================
    # Test 6: Parallel Workers
    # =========================================================================
    def test_parallel_workers(self, generator_class, mock_env_vars, sample_prompts):
        """Verify ThreadPoolExecutor processes prompts correctly"""
        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            processed_ids = []

            def track_processing(*args, **kwargs):
                # Extract prompt from body to get ID
                response_body = MagicMock()
                response_body.read.return_value = json.dumps({
                    'content': [{'text': 'Response'}],
                    'usage': {'input_tokens': 50, 'output_tokens': 100}
                }).encode()
                return {'body': response_body}

            mock_client.invoke_model.side_effect = track_processing
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(model_id='test-model')

            with patch('scripts.generate_teacher_outputs.time.sleep'):
                results = generator.generate_batch(
                    prompts=sample_prompts,
                    max_workers=3  # All prompts in parallel
                )

            # Verify all prompts processed
            assert len(results) == 3
            assert all(r['success'] for r in results)

            # Verify IDs preserved and sorted
            result_ids = [r['id'] for r in results]
            assert sorted(result_ids) == result_ids

    # =========================================================================
    # Test: API Key Header Registration
    # =========================================================================
    def test_api_key_header_registration(self, generator_class, mock_env_vars):
        """Verify API key header is registered when key is present"""
        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Create generator with API key in env
            generator = generator_class(model_id='test-model')

            # Verify event handler was registered
            mock_client.meta.events.register.assert_called_once()
            call_args = mock_client.meta.events.register.call_args
            assert 'before-send.bedrock-runtime' in call_args[0][0]
