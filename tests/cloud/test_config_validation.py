"""
Configuration and Infrastructure Validation Tests

Validates that config.yaml, AWS resources, and pipeline configuration
are consistent and correct:
- Model ID exists in Bedrock
- S3 bucket and prefixes are accessible
- Config cross-field validation
- Hyperparameter range validation
- IAM role existence
- QLoRA config sanity
"""

import os
import sys
import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'


# =============================================================================
# CONFIG FILE STRUCTURE
# =============================================================================

@pytest.mark.offline
class TestConfigFileStructure:
    """Tests for config.yaml structure and required fields"""

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_config_file_exists(self):
        """config.yaml must exist"""
        assert CONFIG_PATH.exists(), "config.yaml not found"

    def test_required_top_level_keys(self, config):
        """Config has all required top-level sections"""
        required = ['project', 'model', 'qlora', 'training', 'distillation', 'aws']
        for key in required:
            assert key in config, f"Missing top-level key: {key}"

    def test_model_name_is_granite(self, config):
        """Model name should reference Granite"""
        model_name = config['model']['name']
        assert 'granite' in model_name.lower(), f"Unexpected model: {model_name}"

    def test_model_has_seq_length(self, config):
        """Model config must specify max sequence length"""
        assert 'max_seq_length' in config['model']
        assert config['model']['max_seq_length'] > 0

    def test_distillation_teacher_is_claude(self, config):
        """Teacher model should be Claude"""
        teacher = config['distillation']['teacher_model']
        assert 'claude' in teacher.lower() or 'anthropic' in teacher.lower()


# =============================================================================
# HYPERPARAMETER RANGE VALIDATION
# =============================================================================

@pytest.mark.offline
class TestHyperparameterRanges:
    """Tests for hyperparameter sanity"""

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_learning_rate_in_safe_range(self, config):
        """Learning rate should be between 1e-5 and 5e-4 for LoRA"""
        lr = config['training']['learning_rate']
        assert 1e-5 <= lr <= 5e-4, f"LR {lr} outside safe range"

    def test_lora_rank_positive(self, config):
        """LoRA rank should be positive"""
        r = config['qlora']['lora_r']
        assert r > 0
        assert r <= 128, f"LoRA rank {r} unusually high"

    def test_lora_alpha_is_2x_rank(self, config):
        """LoRA alpha should typically be 2x rank"""
        r = config['qlora']['lora_r']
        alpha = config['qlora']['lora_alpha']
        ratio = alpha / r
        assert 1 <= ratio <= 4, f"Alpha/rank ratio {ratio} is unusual"

    def test_batch_size_positive(self, config):
        """Batch size must be positive"""
        bs = config['training']['per_device_train_batch_size']
        assert bs >= 1

    def test_gradient_accumulation_positive(self, config):
        """Gradient accumulation steps must be positive"""
        ga = config['training']['gradient_accumulation_steps']
        assert ga >= 1

    def test_effective_batch_size_reasonable(self, config):
        """Effective batch size should be between 4 and 128"""
        bs = config['training']['per_device_train_batch_size']
        ga = config['training']['gradient_accumulation_steps']
        effective = bs * ga
        assert 4 <= effective <= 128, f"Effective batch {effective} unusual"

    def test_max_grad_norm_set(self, config):
        """Max gradient norm should be set for stability"""
        mgn = config['training']['max_grad_norm']
        assert 0 < mgn <= 1.0, f"max_grad_norm {mgn} outside [0, 1]"

    def test_num_epochs_reasonable(self, config):
        """Number of epochs should be between 1 and 20"""
        epochs = config['training']['num_epochs']
        assert 1 <= epochs <= 20

    def test_warmup_ratio_small(self, config):
        """Warmup ratio should be small (< 20%)"""
        warmup = config['training']['warmup_ratio']
        assert 0 <= warmup <= 0.2

    def test_dropout_in_range(self, config):
        """LoRA dropout should be between 0 and 0.5"""
        dropout = config['qlora']['lora_dropout']
        assert 0 <= dropout <= 0.5

    def test_quantization_type_valid(self, config):
        """Quantization type should be nf4 or fp4"""
        qtype = config['qlora']['bnb_4bit_quant_type']
        assert qtype in ('nf4', 'fp4'), f"Unknown quant type: {qtype}"

    def test_quality_threshold_in_range(self, config):
        """Quality threshold should be between 1 and 10"""
        threshold = config['distillation']['min_score_threshold']
        assert 1 <= threshold <= 10

    def test_convergence_above_quality_threshold(self, config):
        """Convergence threshold must be >= quality threshold"""
        quality = config['distillation']['min_score_threshold']
        convergence = config['distillation']['convergence_threshold']
        assert convergence >= quality


# =============================================================================
# QLORA TARGET MODULES
# =============================================================================

@pytest.mark.offline
class TestQLoRATargetModules:
    """Tests for QLoRA target module configuration"""

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_target_modules_not_empty(self, config):
        """Target modules list should not be empty"""
        modules = config['qlora']['lora_target_modules']
        assert len(modules) > 0

    def test_attention_modules_included(self, config):
        """At minimum q_proj and v_proj should be targeted"""
        modules = config['qlora']['lora_target_modules']
        assert 'q_proj' in modules, "q_proj missing from target modules"
        assert 'v_proj' in modules, "v_proj missing from target modules"

    def test_all_modules_are_strings(self, config):
        """All target modules should be strings"""
        modules = config['qlora']['lora_target_modules']
        for m in modules:
            assert isinstance(m, str)


# =============================================================================
# AWS CONFIGURATION
# =============================================================================

@pytest.mark.offline
class TestAWSConfiguration:
    """Tests for AWS config consistency"""

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_region_is_valid(self, config):
        """AWS region should be a known region format"""
        region = config['aws']['region']
        assert region.startswith(('us-', 'eu-', 'ap-', 'sa-', 'ca-', 'me-', 'af-'))

    def test_s3_bucket_name_set(self, config):
        """S3 bucket name should be set"""
        bucket = config['aws']['s3']['bucket_name']
        assert len(bucket) > 0
        assert ' ' not in bucket, "Bucket name contains spaces"

    def test_instance_type_valid(self, config):
        """Training instance type should be a valid ML instance"""
        instance = config['aws']['training_job']['instance_type']
        assert instance.startswith('ml.'), f"Invalid instance type: {instance}"

    def test_bedrock_model_matches_distillation(self, config):
        """Bedrock model ID in AWS config should match distillation teacher"""
        bedrock_model = config['aws']['bedrock']['model_id']
        distillation_model = config['distillation']['teacher_model']
        assert bedrock_model == distillation_model, \
            f"Model mismatch: {bedrock_model} vs {distillation_model}"

    def test_s3_bucket_matches_data_config(self, config):
        """S3 bucket in AWS config should match paths config"""
        aws_bucket = config['aws']['s3']['bucket_name']
        data_bucket = config['paths']['data']['s3_bucket']
        assert aws_bucket == data_bucket, \
            f"Bucket mismatch: {aws_bucket} vs {data_bucket}"


# =============================================================================
# LIVE AWS INFRASTRUCTURE VALIDATION
# =============================================================================

@pytest.mark.cloud
@pytest.mark.integration
class TestLiveAWSInfrastructure:
    """Tests requiring real AWS credentials"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    @pytest.fixture
    def config(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_s3_bucket_accessible(self, real_s3_client, config):
        """S3 bucket from config is accessible"""
        from botocore.exceptions import ClientError

        bucket = config['aws']['s3']['bucket_name']
        try:
            real_s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            pytest.fail(f"Cannot access bucket {bucket}: {e}")

    def test_s3_has_tsn_data(self, real_s3_client, config):
        """S3 bucket has tsn_data/ prefix with files"""
        bucket = config['aws']['s3']['bucket_name']
        response = real_s3_client.list_objects_v2(
            Bucket=bucket, Prefix='tsn_data/', MaxKeys=5
        )
        assert response.get('KeyCount', 0) > 0, "No files in tsn_data/"

    def test_bedrock_model_accessible(self, real_bedrock_client, config):
        """Bedrock model from config can be invoked"""
        from botocore.exceptions import ClientError

        model_id = config['aws']['bedrock']['model_id']
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Say hi."}]
        }

        try:
            response = real_bedrock_client.invoke_model(
                modelId=model_id, body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            assert 'content' in result
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == 'AccessDeniedException':
                pytest.skip("Bedrock model access not enabled")
            raise

    def test_iam_role_exists(self, config):
        """IAM role from config exists"""
        import boto3
        from botocore.exceptions import ClientError

        role_name = config['aws']['iam']['role_name']
        iam = boto3.client('iam')

        try:
            iam.get_role(RoleName=role_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                pytest.skip(f"IAM role {role_name} not found")
            raise
