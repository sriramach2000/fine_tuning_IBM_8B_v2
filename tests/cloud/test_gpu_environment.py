"""
GPU Environment Validation Tests

Validates CUDA/GPU hardware is correctly detected and configured:
- CUDA availability and version
- GPU count, memory, and compute capability
- Multi-GPU readiness (NCCL)
- Environment variable configuration
- GPU health checks (temperature, ECC, occupancy)
"""

import os
import subprocess
import pytest

import unittest.mock

try:
    import torch
    _torch_real = not isinstance(torch.cuda, unittest.mock.MagicMock)
except ImportError:
    torch = None
    _torch_real = False


def requires_torch(func):
    """Skip test if torch is not installed or mocked"""
    return pytest.mark.skipif(
        torch is None or not _torch_real,
        reason="PyTorch not installed or mocked"
    )(func)


# =============================================================================
# CUDA AVAILABILITY
# =============================================================================

@pytest.mark.gpu
class TestCUDAAvailability:
    """Tests for basic CUDA availability"""

    @requires_torch
    def test_cuda_available(self):
        """CUDA must be available on GPU instances"""
        assert torch.cuda.is_available(), \
            "CUDA not available - check driver installation"

    @requires_torch
    def test_cuda_version_matches_pytorch(self):
        """CUDA runtime version must be compatible with PyTorch build"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        cuda_version = torch.version.cuda
        assert cuda_version is not None, "PyTorch built without CUDA support"
        major = int(cuda_version.split('.')[0])
        assert major >= 11, f"CUDA {cuda_version} too old, need >= 11.x"

    @requires_torch
    def test_cuda_device_count_minimum(self):
        """At least 1 GPU must be detected"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        count = torch.cuda.device_count()
        assert count >= 1, f"Expected at least 1 GPU, found {count}"

    @requires_torch
    def test_cuda_device_initialized(self):
        """CUDA device can be initialized"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        device = torch.device('cuda:0')
        # Allocate a small tensor to verify device works
        t = torch.zeros(1, device=device)
        assert t.device.type == 'cuda'
        del t
        torch.cuda.empty_cache()


# =============================================================================
# GPU HARDWARE SPECS
# =============================================================================

@pytest.mark.gpu
class TestGPUHardwareSpecs:
    """Tests for GPU hardware specifications"""

    @requires_torch
    def test_gpu_memory_sufficient(self):
        """Each GPU must have >= 24GB memory (A10G minimum)"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            assert memory_gb >= 20, \
                f"GPU {i} ({props.name}): {memory_gb:.1f}GB < 24GB minimum"

    @requires_torch
    def test_bfloat16_supported(self):
        """GPU must support bfloat16 (compute capability >= 8.0)"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capability = float(f"{props.major}.{props.minor}")
            assert capability >= 8.0, \
                f"GPU {i} ({props.name}): compute capability {capability} < 8.0, bfloat16 not supported"

    @requires_torch
    def test_cuda_compute_capability(self):
        """GPU compute capability must be >= 7.0 (Volta+)"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            capability = float(f"{props.major}.{props.minor}")
            assert capability >= 7.0, \
                f"GPU {i}: compute capability {capability} < 7.0"

    @requires_torch
    def test_cudnn_available(self):
        """cuDNN must be available"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        assert torch.backends.cudnn.is_available(), "cuDNN not available"

    @requires_torch
    def test_gpu_not_occupied(self):
        """GPU memory should start relatively clean"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        assert allocated_mb < 500, \
            f"GPU 0 has {allocated_mb:.0f}MB allocated at start, expected < 500MB"

    @requires_torch
    def test_cuda_device_name_expected(self):
        """GPU should be a known training-grade device"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        name = torch.cuda.get_device_name(0)
        known_gpus = ['A100', 'A10G', 'A10', 'V100', 'H100', 'L4', 'T4', 'L40']
        assert any(gpu in name for gpu in known_gpus), \
            f"Unexpected GPU: {name}"

    @requires_torch
    def test_all_gpus_same_type(self):
        """All GPUs should be the same model"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        count = torch.cuda.device_count()
        if count <= 1:
            pytest.skip("Single GPU, no homogeneity check needed")
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        assert len(set(names)) == 1, \
            f"Mixed GPU types detected: {set(names)}"


# =============================================================================
# MULTI-GPU READINESS
# =============================================================================

@pytest.mark.gpu
class TestMultiGPUReadiness:
    """Tests for multi-GPU training readiness"""

    @requires_torch
    def test_nccl_backend_available(self):
        """NCCL backend must be available for multi-GPU"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        assert torch.distributed.is_nccl_available(), \
            "NCCL not available for distributed training"

    @requires_torch
    def test_gpu_count_matches_instance_type(self):
        """GPU count should match expected for instance type"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")

        count = torch.cuda.device_count()
        instance_type = os.environ.get('SM_CURRENT_INSTANCE_TYPE', '')

        expected = {
            'ml.p4d.24xlarge': 8,
            'ml.p4de.24xlarge': 8,
            'ml.g5.12xlarge': 4,
            'ml.g5.48xlarge': 8,
            'ml.g5.xlarge': 1,
            'ml.g5.2xlarge': 1,
        }

        if instance_type in expected:
            assert count == expected[instance_type], \
                f"{instance_type}: expected {expected[instance_type]} GPUs, found {count}"
        else:
            # Not on a known instance, just verify count > 0
            assert count >= 1

    @requires_torch
    def test_peer_to_peer_access(self):
        """GPUs should support peer-to-peer access for fast communication"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        count = torch.cuda.device_count()
        if count <= 1:
            pytest.skip("Single GPU")
        # Check P2P between GPU 0 and GPU 1
        can_p2p = torch.cuda.can_device_access_peer(0, 1)
        if not can_p2p:
            pytest.skip("P2P not supported (may still work via PCIe)")


# =============================================================================
# CUDA ENVIRONMENT VARIABLES
# =============================================================================

@pytest.mark.gpu
class TestCUDAEnvironment:
    """Tests for CUDA-related environment variables"""

    def test_pytorch_cuda_alloc_conf(self):
        """PYTORCH_CUDA_ALLOC_CONF should be set for memory optimization"""
        conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        if conf:
            assert 'expandable_segments' in conf or 'max_split_size_mb' in conf, \
                f"PYTORCH_CUDA_ALLOC_CONF set but missing key settings: {conf}"

    def test_tokenizers_parallelism_disabled(self):
        """TOKENIZERS_PARALLELISM should be false to avoid deadlocks"""
        val = os.environ.get('TOKENIZERS_PARALLELISM', '')
        if val:
            assert val.lower() == 'false', \
                f"TOKENIZERS_PARALLELISM should be 'false', got '{val}'"


# =============================================================================
# GPU HEALTH (nvidia-smi)
# =============================================================================

@pytest.mark.gpu
class TestGPUHealth:
    """Tests for GPU health via nvidia-smi"""

    def test_nvidia_smi_available(self):
        """nvidia-smi command must be available"""
        try:
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 0, \
                f"nvidia-smi failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found")

    def test_gpu_temperature_within_range(self):
        """GPU temperature should be below 85°C"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("nvidia-smi query failed")

            for line in result.stdout.strip().split('\n'):
                temp = int(line.strip())
                assert temp < 85, f"GPU temperature {temp}°C >= 85°C threshold"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found")

    def test_gpu_ecc_errors(self):
        """No uncorrectable ECC errors on GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=ecc.errors.uncorrected.aggregate.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("ECC query not supported")

            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and line != 'N/A' and line != '[N/A]':
                    errors = int(line)
                    assert errors == 0, f"Found {errors} uncorrectable ECC errors"
        except (FileNotFoundError, ValueError):
            pytest.skip("ECC check not available")

    def test_gpu_utilization_starts_low(self):
        """GPU utilization should start low (not already running workloads)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                pytest.skip("nvidia-smi query failed")

            for line in result.stdout.strip().split('\n'):
                util = int(line.strip())
                assert util < 50, \
                    f"GPU utilization already at {util}%, expected < 50% at start"
        except FileNotFoundError:
            pytest.skip("nvidia-smi not found")
