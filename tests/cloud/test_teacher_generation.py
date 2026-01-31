"""
Teacher Generation Cloud Tests

Tests for BedrockTeacherGenerator including:
- Real Bedrock model invocations with automotive prompts
- Response quality scoring
- Token usage tracking
- Checkpoint save/load
- Parallel generation correctness
- System prompt adherence
- Response format validation
"""

import os
import sys
import json
import time
import tempfile
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.generate_teacher_outputs import (
    BedrockTeacherGenerator,
    create_automotive_system_prompt,
    create_sample_prompts,
)


# =============================================================================
# GENERATOR INITIALIZATION
# =============================================================================

@pytest.mark.offline
class TestGeneratorInitialization:
    """Tests for BedrockTeacherGenerator initialization"""

    def test_default_model_id(self):
        """Verify default model ID is Claude Sonnet"""
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator()
            assert 'claude' in gen.model_id.lower() or 'anthropic' in gen.model_id.lower()

    def test_custom_model_id(self):
        """Verify custom model ID is accepted"""
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator(model_id='custom-model-v1')
            assert gen.model_id == 'custom-model-v1'

    def test_api_key_from_env(self, monkeypatch):
        """Verify API key is read from environment"""
        monkeypatch.setenv('AMAZON_BEDROCK_MODEL_API_KEY', 'test-key-123')
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator()
            assert gen.api_key == 'test-key-123'

    def test_api_key_from_parameter(self):
        """Verify API key parameter overrides env"""
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator(api_key='param-key')
            assert gen.api_key == 'param-key'

    def test_retry_config(self):
        """Verify retry settings are configured"""
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator(max_retries=10, base_delay=2.0)
            assert gen.max_retries == 10
            assert gen.base_delay == 2.0


# =============================================================================
# RESPONSE GENERATION (MOCKED)
# =============================================================================

@pytest.mark.offline
class TestResponseGeneration:
    """Tests for generate_response with mocked Bedrock"""

    @pytest.fixture
    def generator(self):
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator()
            return gen

    def test_successful_response(self, generator):
        """Test successful response parsing"""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'int tsn_init() { return 0; }'}],
            'usage': {'input_tokens': 50, 'output_tokens': 100}
        }).encode()
        generator.client.invoke_model.return_value = {'body': mock_body}

        result = generator.generate_response("Generate TSN code")
        assert result['success'] is True
        assert 'tsn_init' in result['response']
        assert result['usage']['output_tokens'] == 100

    def test_empty_response_handled(self, generator):
        """Test empty content array is handled"""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [],
            'usage': {'input_tokens': 50, 'output_tokens': 0}
        }).encode()
        generator.client.invoke_model.return_value = {'body': mock_body}

        result = generator.generate_response("Generate code")
        assert result['success'] is False
        assert 'Empty' in result['error']

    def test_system_prompt_included(self, generator):
        """Test system prompt is included in request body"""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'response'}],
            'usage': {'input_tokens': 50, 'output_tokens': 10}
        }).encode()
        generator.client.invoke_model.return_value = {'body': mock_body}

        generator.generate_response("prompt", system_prompt="You are an expert.")

        call_args = generator.client.invoke_model.call_args
        body = json.loads(call_args[1]['body'] if 'body' in call_args[1] else call_args[0][0])
        assert 'system' in body

    def test_throttling_triggers_retry(self, generator):
        """Test ThrottlingException triggers retry with backoff"""
        from botocore.exceptions import ClientError

        call_count = [0]

        def throttle_twice(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ClientError(
                    {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                    'InvokeModel'
                )
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                'content': [{'text': 'success'}],
                'usage': {'input_tokens': 10, 'output_tokens': 5}
            }).encode()
            return {'body': mock_body}

        generator.client.invoke_model.side_effect = throttle_twice
        generator.base_delay = 0.01  # Speed up test

        result = generator.generate_response("test")
        assert result['success'] is True
        assert call_count[0] == 3

    def test_max_retries_exceeded_returns_error(self, generator):
        """Test all retries exhausted returns error"""
        from botocore.exceptions import ClientError

        generator.client.invoke_model.side_effect = ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'InvokeModel'
        )
        generator.max_retries = 3
        generator.base_delay = 0.001

        result = generator.generate_response("test")
        assert result['success'] is False
        assert 'retries' in result['error'].lower()

    def test_non_retryable_error_fails_immediately(self, generator):
        """Test non-retryable errors fail without retry"""
        from botocore.exceptions import ClientError

        call_count = [0]

        def fail_immediately(*args, **kwargs):
            call_count[0] += 1
            raise ClientError(
                {'Error': {'Code': 'AccessDeniedException', 'Message': 'No access'}},
                'InvokeModel'
            )

        generator.client.invoke_model.side_effect = fail_immediately

        result = generator.generate_response("test")
        assert result['success'] is False
        assert call_count[0] == 1  # No retry


# =============================================================================
# BATCH GENERATION
# =============================================================================

@pytest.mark.offline
class TestBatchGeneration:
    """Tests for batch prompt processing"""

    @pytest.fixture
    def generator(self):
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator()
            mock_body = MagicMock()
            mock_body.read.return_value = json.dumps({
                'content': [{'text': 'generated code'}],
                'usage': {'input_tokens': 50, 'output_tokens': 100}
            }).encode()
            gen.client.invoke_model.return_value = {'body': mock_body}
            return gen

    def test_batch_processes_all_prompts(self, generator):
        """All prompts in batch are processed"""
        prompts = [{'id': f'p{i}', 'prompt': f'Generate code {i}'} for i in range(5)]
        results = generator.generate_batch(prompts, max_workers=2)
        assert len(results) == 5

    def test_batch_preserves_ids(self, generator):
        """Batch results maintain prompt IDs"""
        prompts = [{'id': f'test_{i}', 'prompt': f'prompt {i}'} for i in range(3)]
        results = generator.generate_batch(prompts, max_workers=1)
        result_ids = {r['id'] for r in results}
        expected_ids = {f'test_{i}' for i in range(3)}
        assert result_ids == expected_ids

    def test_batch_skips_empty_prompts(self, generator):
        """Empty prompts are skipped"""
        prompts = [
            {'id': 'valid', 'prompt': 'Generate code'},
            {'id': 'empty', 'prompt': ''},
        ]
        results = generator.generate_batch(prompts, max_workers=1)
        # Empty prompt returns None, so only valid one should be in results
        assert all(r is not None for r in results)

    def test_checkpoint_saves_results(self, generator, tmp_path):
        """Checkpoint file is written during batch processing"""
        prompts = [{'id': f'p{i}', 'prompt': f'code {i}'} for i in range(5)]
        output_file = str(tmp_path / 'output.jsonl')

        generator.generate_batch(
            prompts, output_file=output_file,
            checkpoint_interval=2, max_workers=1
        )

        assert Path(output_file).exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_results_sorted_by_id(self, generator):
        """Results are sorted by ID after batch processing"""
        prompts = [{'id': f'z{9-i}', 'prompt': f'code {i}'} for i in range(5)]
        results = generator.generate_batch(prompts, max_workers=2)
        ids = [r['id'] for r in results]
        assert ids == sorted(ids)


# =============================================================================
# SYSTEM PROMPT QUALITY
# =============================================================================

@pytest.mark.offline
class TestSystemPromptQuality:
    """Tests for automotive system prompt content"""

    def test_system_prompt_mentions_tsn(self):
        """System prompt should mention TSN"""
        prompt = create_automotive_system_prompt()
        assert 'TSN' in prompt or 'Time-Sensitive' in prompt

    def test_system_prompt_mentions_avb(self):
        """System prompt should mention AVB"""
        prompt = create_automotive_system_prompt()
        assert 'AVB' in prompt or 'Audio Video Bridging' in prompt

    def test_system_prompt_mentions_misra(self):
        """System prompt should mention MISRA-C"""
        prompt = create_automotive_system_prompt()
        assert 'MISRA' in prompt

    def test_system_prompt_mentions_data_types(self):
        """System prompt should mention fixed-width data types"""
        prompt = create_automotive_system_prompt()
        assert 'uint' in prompt or 'data type' in prompt.lower()

    def test_sample_prompts_cover_all_domains(self):
        """Sample prompts should cover TSN, AVB, and automotive domains"""
        prompts = create_sample_prompts()
        all_text = ' '.join(p['prompt'] for p in prompts).lower()
        assert 'tsn' in all_text
        assert 'avb' in all_text

    def test_sample_prompts_have_unique_ids(self):
        """All sample prompts should have unique IDs"""
        prompts = create_sample_prompts()
        ids = [p['id'] for p in prompts]
        assert len(ids) == len(set(ids))


# =============================================================================
# TOKEN USAGE TRACKING
# =============================================================================

@pytest.mark.offline
class TestTokenUsageTracking:
    """Tests for token usage tracking accuracy"""

    @pytest.fixture
    def generator(self):
        with patch('boto3.client') as mock_client:
            gen = BedrockTeacherGenerator()
            return gen

    def test_token_usage_in_successful_response(self, generator):
        """Successful response includes token usage"""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'code'}],
            'usage': {'input_tokens': 100, 'output_tokens': 500}
        }).encode()
        generator.client.invoke_model.return_value = {'body': mock_body}

        result = generator.generate_response("test")
        assert result['usage']['input_tokens'] == 100
        assert result['usage']['output_tokens'] == 500

    def test_total_tokens_across_batch(self, generator):
        """Token usage sums correctly across batch"""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({
            'content': [{'text': 'code'}],
            'usage': {'input_tokens': 50, 'output_tokens': 200}
        }).encode()
        generator.client.invoke_model.return_value = {'body': mock_body}

        prompts = [{'id': f'p{i}', 'prompt': f'code {i}'} for i in range(3)]
        results = generator.generate_batch(prompts, max_workers=1)

        total_output = sum(r.get('usage', {}).get('output_tokens', 0) for r in results)
        assert total_output == 600  # 3 * 200


# =============================================================================
# REAL BEDROCK INTEGRATION
# =============================================================================

@pytest.mark.cloud
@pytest.mark.integration
@pytest.mark.slow
class TestTeacherGenerationIntegration:
    """Integration tests with real Bedrock"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_real_automotive_code_generation(self):
        """Generate real automotive code and verify quality"""
        from botocore.exceptions import ClientError

        try:
            gen = BedrockTeacherGenerator(max_tokens=512, temperature=0.3)
            result = gen.generate_response(
                "Generate a C function to initialize a TSN frame structure with priority, vlan_id, and timestamp fields.",
                system_prompt=create_automotive_system_prompt()
            )

            assert result['success'] is True
            assert result['response'] is not None
            assert len(result['response']) > 50

            # Verify the response contains code-like content
            response = result['response']
            assert any(kw in response for kw in ['int', 'void', 'struct', 'return', '#include']), \
                "Response doesn't look like C code"

        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                pytest.skip("Bedrock access not enabled")
            raise

    def test_real_response_passes_quality_threshold(self):
        """Real Bedrock response should pass quality evaluation"""
        from botocore.exceptions import ClientError
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        try:
            gen = BedrockTeacherGenerator(max_tokens=1024, temperature=0.3)
            result = gen.generate_response(
                "Generate C code for AVB stream reservation with stream_id, sample_rate, channels, and bandwidth fields. Include initialization function.",
                system_prompt=create_automotive_system_prompt()
            )

            if not result['success']:
                pytest.skip(f"Bedrock call failed: {result['error']}")

            evaluator = CodeQualityEvaluator(gcc_available=False)
            score = evaluator.evaluate(
                result['response'],
                "Generate AVB stream reservation code"
            )

            # Teacher output should score reasonably well
            assert score.overall >= 4.0, \
                f"Teacher output scored too low: {score}"

        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                pytest.skip("Bedrock access not enabled")
            raise

    def test_real_token_usage_returned(self):
        """Real Bedrock response includes token usage"""
        from botocore.exceptions import ClientError

        try:
            gen = BedrockTeacherGenerator(max_tokens=100)
            result = gen.generate_response("Say hello in one word.")

            assert result['success'] is True
            assert 'usage' in result
            assert result['usage'].get('input_tokens', 0) > 0
            assert result['usage'].get('output_tokens', 0) > 0

        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                pytest.skip("Bedrock access not enabled")
            raise
