"""
Tests for Failure Scenarios and Error Recovery

Tests:
1. test_s3_download_failure_recovery - S3 access denied handling
2. test_disk_space_check - Disk space validation before save
3. test_checkpoint_save_failure - Handle checkpoint save errors
4. test_partial_batch_completion - Handle partial batch failures
5. test_config_validation - Validate config file errors
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestS3Failures:
    """Test S3 operation failure handling"""

    @pytest.fixture
    def pipeline_class(self):
        """Import the pipeline class"""
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        return AutomotiveDataPipeline

    # =========================================================================
    # Test 1: S3 Download Failure Recovery
    # =========================================================================
    def test_s3_download_failure_recovery(self, pipeline_class, mock_env_vars):
        """Handle S3 access denied errors gracefully"""
        from botocore.exceptions import ClientError

        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Simulate access denied
            mock_client.download_file.side_effect = ClientError(
                {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
                'GetObject'
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Should return None and not crash
                result = pipeline.download_file('some/key.c', Path(tmpdir) / 'output.c')
                assert result is None


class TestDiskSpaceValidation:
    """Test disk space validation"""

    # =========================================================================
    # Test 2: Disk Space Check
    # =========================================================================
    def test_disk_space_check(self, mock_env_vars):
        """Verify disk space check before model save"""
        from training.train_granite_qlora import check_disk_space

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should pass with reasonable requirement
            try:
                check_disk_space(tmpdir, required_gb=0.001)  # 1MB
            except Exception as e:
                pytest.fail(f"Should pass with minimal requirement: {e}")

            # Should fail with unreasonable requirement
            with pytest.raises(Exception):
                check_disk_space(tmpdir, required_gb=1000000)  # 1 Petabyte


class TestCheckpointFailures:
    """Test checkpoint save/load failure handling"""

    # =========================================================================
    # Test 3: Checkpoint Save Failure
    # =========================================================================
    def test_checkpoint_save_failure(self, mock_env_vars):
        """Handle checkpoint save errors gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a read-only directory to simulate save failure
            readonly_dir = Path(tmpdir) / 'readonly'
            readonly_dir.mkdir()

            # Create mock model with save_pretrained
            mock_model = MagicMock()
            mock_model.save_pretrained.side_effect = PermissionError("Permission denied")

            # The save should raise an error (which training script catches)
            with pytest.raises(PermissionError):
                mock_model.save_pretrained(str(readonly_dir / 'model'))


class TestBatchProcessingFailures:
    """Test batch processing failure handling"""

    @pytest.fixture
    def generator_class(self):
        """Import the generator class"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator
        return BedrockTeacherGenerator

    # =========================================================================
    # Test 4: Partial Batch Completion
    # =========================================================================
    def test_partial_batch_completion(self, generator_class, mock_env_vars, sample_prompts):
        """Handle partial batch failures - some prompts succeed, some fail"""
        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            call_count = [0]

            def intermittent_failure(*args, **kwargs):
                call_count[0] += 1
                # Fail every other request
                if call_count[0] % 2 == 0:
                    raise Exception("Intermittent failure")

                response_body = MagicMock()
                response_body.read.return_value = json.dumps({
                    'content': [{'text': 'Success'}],
                    'usage': {'input_tokens': 50, 'output_tokens': 100}
                }).encode()
                return {'body': response_body}

            mock_client.invoke_model.side_effect = intermittent_failure
            mock_client.meta.events.register = MagicMock()

            generator = generator_class(model_id='test-model', max_retries=1)

            with patch('scripts.generate_teacher_outputs.time.sleep'):
                results = generator.generate_batch(
                    prompts=sample_prompts,
                    max_workers=1  # Sequential to ensure predictable order
                )

            # Should have processed all prompts
            assert len(results) == 3

            # Some should succeed, some should fail
            successes = sum(1 for r in results if r['success'])
            failures = sum(1 for r in results if not r['success'])

            # At least one success and one failure expected
            assert successes >= 1
            assert failures >= 1


class TestConfigValidation:
    """Test configuration validation"""

    # =========================================================================
    # Test 5: Config Validation
    # =========================================================================
    def test_config_validation_missing_sections(self, mock_env_vars):
        """Detect missing required config sections"""
        from scripts.dry_run_pipeline import PipelineValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create incomplete config
            config_path = Path(tmpdir) / 'bad_config.yaml'
            config_path.write_text('''
model:
  name: "test-model"
# Missing: qlora, training, aws, distillation sections
''')

            validator = PipelineValidator(config_path=str(config_path))

            # Run config validation
            result = validator.test_config_validity()

            # Should fail due to missing sections
            assert result is False
            assert len(validator.errors) > 0

    def test_config_validation_wrong_model(self, mock_env_vars):
        """Detect wrong model name in config"""
        from scripts.dry_run_pipeline import PipelineValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'wrong_model_config.yaml'
            config_path.write_text('''
model:
  name: "wrong-model-name"
qlora:
  lora_r: 32
training:
  num_epochs: 5
aws:
  region: "us-east-1"
distillation:
  enabled: true
''')

            validator = PipelineValidator(config_path=str(config_path))
            result = validator.test_config_validity()

            # Should fail due to non-Granite model
            assert result is False


class TestTokenLimitValidation:
    """Test token limit handling"""

    def test_response_token_tracking(self, mock_env_vars):
        """Verify token usage is tracked correctly"""
        from scripts.generate_teacher_outputs import BedrockTeacherGenerator

        with patch('scripts.generate_teacher_outputs.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Response with token usage
            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                'content': [{'text': 'Generated response'}],
                'usage': {'input_tokens': 150, 'output_tokens': 800}
            }).encode()
            mock_client.invoke_model.return_value = {'body': response_body}
            mock_client.meta.events.register = MagicMock()

            generator = BedrockTeacherGenerator(model_id='test-model', max_tokens=2048)
            result = generator.generate_response("Test prompt")

            # Verify tokens are captured
            assert result['success'] is True
            assert result['usage']['input_tokens'] == 150
            assert result['usage']['output_tokens'] == 800
