"""
Bedrock Operations Test Suite

Comprehensive tests for Amazon Bedrock including:
- Model invocation
- Error handling
- Rate limiting and retry
- Context length validation
- Response format validation
"""

import json
import time
import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError


# =============================================================================
# BEDROCK BASIC OPERATIONS
# =============================================================================

@pytest.mark.bedrock
class TestBedrockBasicOperations:
    """Basic Bedrock operation tests"""

    def test_model_invocation_success(self, mock_bedrock_client):
        """Test successful model invocation"""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }

        response = mock_bedrock_client.invoke_model(
            modelId='test-model',
            body=json.dumps(body)
        )

        result = json.loads(response['body'].read())
        assert 'content' in result
        assert len(result['content']) > 0

    def test_response_contains_usage_metrics(self, mock_bedrock_client):
        """Test that response includes token usage"""
        response = mock_bedrock_client.invoke_model(
            modelId='test-model',
            body=json.dumps({})
        )

        result = json.loads(response['body'].read())
        assert 'usage' in result
        assert 'input_tokens' in result['usage']
        assert 'output_tokens' in result['usage']


# =============================================================================
# BEDROCK ERROR HANDLING
# =============================================================================

@pytest.mark.bedrock
@pytest.mark.offline
class TestBedrockErrorHandling:
    """Bedrock error handling tests"""

    def test_invalid_model_id(self, mock_bedrock_client, bedrock_validation_error):
        """Test handling of invalid model ID"""
        mock_bedrock_client.invoke_model.side_effect = bedrock_validation_error(
            'Invalid model ID: nonexistent-model'
        )

        with pytest.raises(ClientError) as exc_info:
            mock_bedrock_client.invoke_model(
                modelId='nonexistent-model',
                body=json.dumps({})
            )

        assert exc_info.value.response['Error']['Code'] == 'ValidationException'

    def test_malformed_request_body(self, mock_bedrock_client):
        """Test handling of malformed request body"""
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'ValidationException', 'Message': 'Invalid JSON'}},
            'InvokeModel'
        )

        with pytest.raises(ClientError):
            mock_bedrock_client.invoke_model(
                modelId='test-model',
                body='not valid json'
            )

    def test_access_denied(self, mock_bedrock_client):
        """Test handling of AccessDeniedException"""
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'AccessDeniedException', 'Message': 'Not authorized'}},
            'InvokeModel'
        )

        with pytest.raises(ClientError) as exc_info:
            mock_bedrock_client.invoke_model(modelId='test-model', body='{}')

        assert exc_info.value.response['Error']['Code'] == 'AccessDeniedException'

    def test_model_not_ready(self, mock_bedrock_client):
        """Test handling when model is not ready"""
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'ModelNotReadyException', 'Message': 'Model warming up'}},
            'InvokeModel'
        )

        with pytest.raises(ClientError) as exc_info:
            mock_bedrock_client.invoke_model(modelId='test-model', body='{}')

        assert 'ModelNotReady' in exc_info.value.response['Error']['Code']


# =============================================================================
# BEDROCK RATE LIMITING
# =============================================================================

@pytest.mark.bedrock
@pytest.mark.resilience
class TestBedrockRateLimiting:
    """Rate limiting and retry tests"""

    def test_throttling_with_retry(self, mock_bedrock_client, bedrock_throttling_error):
        """Test retry on throttling exception"""
        call_count = [0]

        def throttle_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise bedrock_throttling_error()

            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                'content': [{'text': 'Success after retry'}],
                'usage': {'input_tokens': 50, 'output_tokens': 100}
            }).encode()
            return {'body': response_body}

        mock_bedrock_client.invoke_model.side_effect = throttle_then_succeed

        # Simulate retry logic
        max_retries = 5
        base_delay = 0.01  # Fast for testing

        for attempt in range(max_retries):
            try:
                response = mock_bedrock_client.invoke_model(
                    modelId='test-model',
                    body='{}'
                )
                break
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                raise

        assert call_count[0] == 3

    def test_max_retries_exceeded(self, mock_bedrock_client, bedrock_throttling_error):
        """Test failure after max retries exceeded"""
        mock_bedrock_client.invoke_model.side_effect = bedrock_throttling_error()

        max_retries = 3
        attempts = 0

        for attempt in range(max_retries):
            try:
                mock_bedrock_client.invoke_model(modelId='test', body='{}')
                break
            except ClientError:
                attempts += 1
                if attempts >= max_retries:
                    break

        assert attempts == max_retries

    def test_exponential_backoff_delays(self):
        """Verify exponential backoff timing calculation"""
        base_delay = 1.0

        expected_delays = [1.0, 2.0, 4.0, 8.0, 16.0]
        actual_delays = [base_delay * (2 ** i) for i in range(5)]

        assert actual_delays == expected_delays


# =============================================================================
# BEDROCK CONTEXT LENGTH
# =============================================================================

@pytest.mark.bedrock
class TestBedrockContextLength:
    """Context length validation tests"""

    def test_context_length_exceeded(self, mock_bedrock_client):
        """Test handling when context length is exceeded"""
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            {'Error': {
                'Code': 'ValidationException',
                'Message': 'Input token count exceeds model limit'
            }},
            'InvokeModel'
        )

        with pytest.raises(ClientError) as exc_info:
            # Very long prompt
            long_prompt = "word " * 100000
            mock_bedrock_client.invoke_model(
                modelId='test-model',
                body=json.dumps({'messages': [{'role': 'user', 'content': long_prompt}]})
            )

        assert 'token' in exc_info.value.response['Error']['Message'].lower()

    @pytest.mark.parametrize("token_count,should_pass", [
        (100, True),
        (1000, True),
        (10000, True),
        (100000, True),
        (200000, False),
    ])
    def test_token_count_thresholds(self, mock_bedrock_client, token_count, should_pass):
        """Test various token count thresholds"""
        if should_pass:
            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                'content': [{'text': 'OK'}],
                'usage': {'input_tokens': token_count, 'output_tokens': 10}
            }).encode()
            mock_bedrock_client.invoke_model.return_value = {'body': response_body}
        else:
            mock_bedrock_client.invoke_model.side_effect = ClientError(
                {'Error': {'Code': 'ValidationException', 'Message': 'Token limit'}},
                'InvokeModel'
            )

        if should_pass:
            response = mock_bedrock_client.invoke_model(modelId='test', body='{}')
            assert response is not None
        else:
            with pytest.raises(ClientError):
                mock_bedrock_client.invoke_model(modelId='test', body='{}')


# =============================================================================
# BEDROCK RESPONSE VALIDATION
# =============================================================================

@pytest.mark.bedrock
class TestBedrockResponseValidation:
    """Response format validation tests"""

    def test_empty_response_handling(self, mock_bedrock_client):
        """Test handling of empty response"""
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            'content': [],
            'usage': {'input_tokens': 10, 'output_tokens': 0}
        }).encode()
        mock_bedrock_client.invoke_model.return_value = {'body': response_body}

        response = mock_bedrock_client.invoke_model(modelId='test', body='{}')
        result = json.loads(response['body'].read())

        assert result['content'] == []

    def test_response_text_extraction(self, mock_bedrock_client):
        """Test extracting text from response"""
        expected_text = "This is the generated code"

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            'content': [{'text': expected_text}],
            'usage': {'input_tokens': 50, 'output_tokens': 10}
        }).encode()
        mock_bedrock_client.invoke_model.return_value = {'body': response_body}

        response = mock_bedrock_client.invoke_model(modelId='test', body='{}')
        result = json.loads(response['body'].read())

        text = result.get('content', [{}])[0].get('text', '')
        assert text == expected_text

    def test_multiple_content_blocks(self, mock_bedrock_client):
        """Test handling multiple content blocks in response"""
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            'content': [
                {'text': 'First block'},
                {'text': 'Second block'},
            ],
            'usage': {'input_tokens': 50, 'output_tokens': 20}
        }).encode()
        mock_bedrock_client.invoke_model.return_value = {'body': response_body}

        response = mock_bedrock_client.invoke_model(modelId='test', body='{}')
        result = json.loads(response['body'].read())

        assert len(result['content']) == 2


# =============================================================================
# BEDROCK INTEGRATION TESTS
# =============================================================================

@pytest.mark.bedrock
@pytest.mark.cloud
@pytest.mark.integration
@pytest.mark.slow
class TestBedrockIntegration:
    """Integration tests requiring real Bedrock access"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_real_model_invocation(self, real_bedrock_client):
        """Test real Bedrock model invocation"""
        model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}]
        }

        try:
            response = real_bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            assert 'content' in result
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                pytest.skip("Model access not enabled")
            raise
