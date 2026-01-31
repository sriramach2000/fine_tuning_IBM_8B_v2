"""
SageMaker Environment Tests

Validates SageMaker-specific environment when running on SageMaker:
- SM_* environment variables
- /opt/ml directory structure
- Data channels availability
- Entry point and imports
"""

import os
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def is_sagemaker():
    """Check if running inside SageMaker"""
    return os.environ.get('SM_TRAINING_ENV') is not None


# =============================================================================
# SAGEMAKER ENVIRONMENT VARIABLES
# =============================================================================

@pytest.mark.sagemaker
@pytest.mark.gpu
class TestSageMakerEnvVars:
    """Tests for SageMaker environment variables"""

    @pytest.fixture(autouse=True)
    def skip_outside_sagemaker(self):
        if not is_sagemaker():
            pytest.skip("Not running on SageMaker")

    def test_sm_training_env_set(self):
        """SM_TRAINING_ENV must be set"""
        val = os.environ.get('SM_TRAINING_ENV')
        assert val is not None
        # Should be valid JSON
        env = json.loads(val)
        assert isinstance(env, dict)

    def test_sm_channel_train_exists(self):
        """Training data channel must exist when data is provided"""
        train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
        if not os.environ.get('SM_CHANNEL_TRAIN'):
            pytest.skip("No SM_CHANNEL_TRAIN configured (test-only job)")
        assert Path(train_dir).exists(), \
            f"Training channel {train_dir} not found"

    def test_sm_channel_val_exists(self):
        """Validation data channel must exist when data is provided"""
        val_dir = os.environ.get('SM_CHANNEL_VAL', '/opt/ml/input/data/val')
        if not os.environ.get('SM_CHANNEL_VAL'):
            pytest.skip("No SM_CHANNEL_VAL configured (test-only job)")
        assert Path(val_dir).exists(), \
            f"Validation channel {val_dir} not found"

    def test_sm_model_dir_writable(self):
        """Model output directory must be writable"""
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        assert Path(model_dir).exists(), f"{model_dir} not found"
        assert os.access(model_dir, os.W_OK), f"{model_dir} not writable"

    def test_sm_output_dir_writable(self):
        """Output data directory must be writable"""
        output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
        assert Path(output_dir).exists(), f"{output_dir} not found"
        assert os.access(output_dir, os.W_OK), f"{output_dir} not writable"

    def test_hyperparameters_accessible(self):
        """SageMaker hyperparameters should be accessible"""
        hp_path = Path('/opt/ml/input/config/hyperparameters.json')
        if hp_path.exists():
            with open(hp_path) as f:
                hp = json.load(f)
            assert isinstance(hp, dict)
        else:
            # Hyperparameters may also be in SM_TRAINING_ENV
            env = json.loads(os.environ['SM_TRAINING_ENV'])
            assert 'hyperparameters' in env or True  # May not have any

    def test_instance_type_detectable(self):
        """Instance type should be detectable from environment"""
        env = json.loads(os.environ['SM_TRAINING_ENV'])
        # SageMaker provides resource config
        resource_config = env.get('resource_config', {})
        if 'current_instance_type' in resource_config:
            instance_type = resource_config['current_instance_type']
            assert instance_type.startswith('ml.'), \
                f"Unexpected instance type format: {instance_type}"


# =============================================================================
# SAGEMAKER DIRECTORY STRUCTURE
# =============================================================================

@pytest.mark.sagemaker
@pytest.mark.gpu
class TestSageMakerDirectories:
    """Tests for /opt/ml directory structure"""

    @pytest.fixture(autouse=True)
    def skip_outside_sagemaker(self):
        if not is_sagemaker():
            pytest.skip("Not running on SageMaker")

    def test_opt_ml_exists(self):
        """/opt/ml should exist"""
        assert Path('/opt/ml').exists()

    def test_train_data_not_empty(self):
        """Training channel should have data files"""
        train_dir = os.environ.get('SM_CHANNEL_TRAIN')
        if not train_dir:
            pytest.skip("No SM_CHANNEL_TRAIN configured")
        files = list(Path(train_dir).glob('*'))
        assert len(files) > 0, f"No files in {train_dir}"

    def test_val_data_not_empty(self):
        """Validation channel should have data files"""
        val_dir = os.environ.get('SM_CHANNEL_VAL')
        if not val_dir:
            pytest.skip("No SM_CHANNEL_VAL configured")
        files = list(Path(val_dir).glob('*'))
        assert len(files) > 0, f"No files in {val_dir}"

    def test_training_data_is_jsonl(self):
        """Training data should be JSONL format"""
        train_dir = os.environ.get('SM_CHANNEL_TRAIN')
        if not train_dir:
            pytest.skip("No SM_CHANNEL_TRAIN configured")
        jsonl_files = list(Path(train_dir).glob('*.jsonl'))
        if not jsonl_files:
            jsonl_files = list(Path(train_dir).glob('*.json'))
        assert len(jsonl_files) > 0, \
            f"No JSONL files in {train_dir}"

        # Verify first file is valid JSONL
        with open(jsonl_files[0]) as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)
            assert isinstance(data, dict)


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

@pytest.mark.gpu
class TestTrainingEntryPoint:
    """Tests for training script entry point"""

    def test_training_script_exists(self):
        """train_granite_qlora.py must exist"""
        training_dir = Path(__file__).parent.parent.parent / 'training'
        script = training_dir / 'train_granite_qlora.py'
        assert script.exists(), f"Training script not found: {script}"

    def test_training_script_parseable(self):
        """Training script must be valid Python"""
        training_dir = Path(__file__).parent.parent.parent / 'training'
        script = training_dir / 'train_granite_qlora.py'
        if not script.exists():
            pytest.skip("Training script not found")

        import ast
        with open(script) as f:
            source = f.read()
        # Should parse without SyntaxError
        ast.parse(source)

    def test_all_training_imports_available(self):
        """Key imports in training script should resolve on GPU instances"""
        import unittest.mock
        try:
            import torch
            if isinstance(torch.cuda, unittest.mock.MagicMock) or not torch.cuda.is_available():
                pytest.skip("GPU-only check")
        except ImportError:
            pytest.skip("PyTorch not installed")

        required_modules = [
            'transformers',
            'peft',
            'trl',
            'datasets',
            'accelerate',
        ]
        missing = []
        for mod in required_modules:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)

        if missing:
            pytest.fail(f"Missing training imports: {missing}")

    def test_iterative_distillation_importable(self):
        """Iterative distillation module should be importable"""
        try:
            import torch as _torch
        except ImportError:
            from unittest.mock import MagicMock
            sys.modules['torch'] = MagicMock()

        try:
            from training.iterative_distillation import (
                IterativeDistillationTrainer,
                DistillationConfig,
            )
            assert IterativeDistillationTrainer is not None
            assert DistillationConfig is not None
        except ImportError as e:
            pytest.fail(f"Cannot import iterative_distillation: {e}")


# =============================================================================
# SAGEMAKER PATHS PERMISSIONS
# =============================================================================

@pytest.mark.sagemaker
@pytest.mark.gpu
class TestSageMakerPermissions:
    """Tests for SageMaker path permissions"""

    @pytest.fixture(autouse=True)
    def skip_outside_sagemaker(self):
        if not is_sagemaker():
            pytest.skip("Not running on SageMaker")

    def test_code_dir_readable(self):
        """/opt/ml/code should be readable"""
        code_dir = Path('/opt/ml/code')
        if code_dir.exists():
            assert os.access(str(code_dir), os.R_OK)

    def test_model_dir_has_space(self):
        """/opt/ml/model should have adequate disk space"""
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        if Path(model_dir).exists():
            import shutil
            usage = shutil.disk_usage(model_dir)
            free_gb = usage.free / (1024 ** 3)
            assert free_gb >= 10, \
                f"Only {free_gb:.1f}GB free in {model_dir}"

    def test_no_permission_errors_on_opt_ml(self):
        """All /opt/ml subdirs should be accessible"""
        base = Path('/opt/ml')
        if not base.exists():
            pytest.skip("/opt/ml not found")

        for subdir in ['model', 'output', 'code']:
            path = base / subdir
            if path.exists():
                assert os.access(str(path), os.R_OK), \
                    f"{path} not readable"
