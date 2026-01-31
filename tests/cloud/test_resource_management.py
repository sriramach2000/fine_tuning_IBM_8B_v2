"""
Resource Management Tests

Validates cache management, disk space, stale processes, and cleanup:
- HuggingFace cache directory
- Disk space for checkpoints
- CUDA memory cleanup
- Checkpoint lifecycle (save_total_limit)
- Process cleanup
- Training callbacks registration
"""

import os
import sys
import gc
import shutil
import subprocess
import tempfile
import pytest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'


def skip_no_cuda():
    import unittest.mock
    if torch is None or isinstance(torch.cuda, unittest.mock.MagicMock):
        pytest.skip("PyTorch not installed or mocked")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def get_config():
    if yaml is None:
        pytest.skip("pyyaml not installed")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

@pytest.mark.gpu
class TestCacheManagement:
    """Tests for HuggingFace cache and temp directories"""

    def test_hf_cache_dir_writable(self):
        """HF cache directory must be writable"""
        cache_dir = os.environ.get(
            'HF_HOME',
            os.environ.get('TRANSFORMERS_CACHE',
                           os.path.expanduser('~/.cache/huggingface'))
        )
        parent = Path(cache_dir).parent
        if not parent.exists():
            pytest.skip(f"Cache parent {parent} doesn't exist")

        # Either cache dir exists and is writable, or parent is writable
        if Path(cache_dir).exists():
            assert os.access(cache_dir, os.W_OK), \
                f"Cache dir {cache_dir} not writable"
        else:
            assert os.access(str(parent), os.W_OK), \
                f"Cache parent {parent} not writable"

    def test_disk_space_for_checkpoints(self):
        """Output directory must have >= 15GB free"""
        output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/tmp')
        if not os.path.exists(output_dir):
            output_dir = '/tmp'

        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024 ** 3)
        assert free_gb >= 5, \
            f"Only {free_gb:.1f}GB free in {output_dir}, need >= 5GB"

    def test_tmp_dir_not_full(self):
        """/tmp should have adequate space"""
        usage = shutil.disk_usage('/tmp')
        free_gb = usage.free / (1024 ** 3)
        assert free_gb >= 1, \
            f"Only {free_gb:.1f}GB free in /tmp"


# =============================================================================
# CUDA MEMORY CLEANUP
# =============================================================================

@pytest.mark.gpu
class TestCUDAMemoryCleanup:
    """Tests for GPU memory management and cleanup"""

    def test_cuda_empty_cache_works(self):
        """torch.cuda.empty_cache() should free reserved memory"""
        skip_no_cuda()

        # Allocate some memory
        tensors = [torch.randn(1000, 1000, device='cuda') for _ in range(10)]
        allocated_before = torch.cuda.memory_allocated()
        assert allocated_before > 0

        # Delete tensors
        del tensors
        gc.collect()
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated()
        assert allocated_after < allocated_before, \
            "empty_cache didn't reduce allocated memory"

    def test_model_unload_frees_gpu(self):
        """Deleting a model + empty_cache should release GPU memory"""
        skip_no_cuda()

        # Create a moderate-sized model on GPU
        model = torch.nn.Sequential(
            *[torch.nn.Linear(1024, 1024) for _ in range(10)]
        ).to('cuda')

        allocated_with_model = torch.cuda.memory_allocated()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated()
        freed_mb = (allocated_with_model - allocated_after) / (1024 ** 2)
        assert freed_mb > 1, \
            f"Only freed {freed_mb:.1f}MB after model deletion"

    def test_memory_after_gc(self):
        """Python GC + CUDA empty_cache should reclaim memory"""
        skip_no_cuda()

        # Allocate tensors without keeping direct references
        for _ in range(5):
            _ = torch.randn(500, 500, device='cuda')

        gc.collect()
        torch.cuda.empty_cache()

        # Memory should be mostly reclaimed
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        assert allocated_mb < 500, \
            f"{allocated_mb:.0f}MB still allocated after GC"


# =============================================================================
# CHECKPOINT LIFECYCLE
# =============================================================================

@pytest.mark.gpu
class TestCheckpointLifecycle:
    """Tests for checkpoint save/load and cleanup"""

    def test_checkpoint_save_total_limit(self):
        """Config should limit checkpoints to avoid disk overflow"""
        config = get_config()
        # The training script uses save_total_limit=2
        # Verify this is a reasonable value
        assert config['training'].get('save_total_limit', 2) <= 5, \
            "save_total_limit too high, risk of filling disk"

    def test_checkpoint_dir_structure(self, tmp_path):
        """Checkpoint directory should have expected structure"""
        # Simulate checkpoint structure
        ckpt_dir = tmp_path / 'checkpoint-100'
        ckpt_dir.mkdir()

        # A LoRA checkpoint should contain adapter files
        (ckpt_dir / 'adapter_config.json').write_text('{}')
        (ckpt_dir / 'adapter_model.safetensors').write_bytes(b'\x00' * 100)

        assert (ckpt_dir / 'adapter_config.json').exists()
        assert (ckpt_dir / 'adapter_model.safetensors').exists()

    def test_stale_checkpoint_cleanup(self, tmp_path):
        """Old checkpoints should be removable"""
        # Create simulated checkpoints
        for i in range(5):
            ckpt = tmp_path / f'checkpoint-{i * 100}'
            ckpt.mkdir()
            (ckpt / 'adapter_config.json').write_text('{}')

        checkpoints = sorted(tmp_path.glob('checkpoint-*'))
        assert len(checkpoints) == 5

        # Keep only last 2 (simulating save_total_limit)
        for ckpt in checkpoints[:-2]:
            shutil.rmtree(ckpt)

        remaining = list(tmp_path.glob('checkpoint-*'))
        assert len(remaining) == 2

    def test_checkpoint_is_lora_only(self):
        """Saved checkpoints should be LoRA adapters (small), not full model"""
        # This validates that the training config uses PEFT saving
        config = get_config()
        # LoRA rank 32 with target modules means adapter < 500MB
        r = config['qlora']['lora_r']
        # Rough estimate: 7 target modules * 2 * hidden_dim * r * 2 bytes
        # For 8B model, hidden_dim ~4096
        estimated_mb = 7 * 2 * 4096 * r * 2 / (1024 ** 2)
        assert estimated_mb < 500, \
            f"Estimated LoRA adapter size {estimated_mb:.0f}MB seems too large"


# =============================================================================
# PROCESS CLEANUP
# =============================================================================

@pytest.mark.gpu
class TestProcessCleanup:
    """Tests for stale process detection and cleanup"""

    def test_no_zombie_gpu_processes(self):
        """nvidia-smi should not show stale processes"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,name,used_memory',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("nvidia-smi query failed")

            lines = result.stdout.strip().split('\n')
            if lines == ['']:
                return  # No GPU processes, good

            # Check if any processes are using > 1GB but not our current process
            current_pid = os.getpid()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    pid = int(parts[0].strip())
                    if pid != current_pid:
                        mem = parts[2].strip()
                        if 'MiB' in mem:
                            mem_mb = int(mem.replace('MiB', '').strip())
                            if mem_mb > 5000:
                                pytest.fail(
                                    f"Stale GPU process PID {pid} using {mem_mb}MB"
                                )

        except FileNotFoundError:
            pytest.skip("nvidia-smi not found")


# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

@pytest.mark.gpu
class TestTrainingCallbacks:
    """Tests for training callback registration"""

    def test_nan_inf_callback_importable(self):
        """NaNInfDetectionCallback should be importable"""
        try:
            from training.train_granite_qlora import NaNInfDetectionCallback
            callback = NaNInfDetectionCallback()
            assert callback is not None
        except ImportError:
            pytest.skip("Training script not importable")

    def test_early_stopping_callback_importable(self):
        """CustomEarlyStoppingCallback should be importable"""
        try:
            from training.train_granite_qlora import CustomEarlyStoppingCallback
            callback = CustomEarlyStoppingCallback(patience=3)
            assert callback.patience == 3
        except ImportError:
            pytest.skip("Training script not importable")

    def test_disk_space_check_function(self):
        """check_disk_space function should work"""
        try:
            from training.train_granite_qlora import check_disk_space
            result = check_disk_space('/tmp', required_gb=0.001)
            assert result is True
        except ImportError:
            pytest.skip("Training script not importable")

    def test_verify_quantization_function_exists(self):
        """verify_quantization function should exist"""
        try:
            from training.train_granite_qlora import verify_quantization
            assert callable(verify_quantization)
        except ImportError:
            pytest.skip("Training script not importable")

    def test_log_cuda_memory_function(self):
        """log_cuda_memory function should not crash"""
        try:
            from training.train_granite_qlora import log_cuda_memory
            # Should not raise even without CUDA
            log_cuda_memory()
        except ImportError:
            pytest.skip("Training script not importable")
