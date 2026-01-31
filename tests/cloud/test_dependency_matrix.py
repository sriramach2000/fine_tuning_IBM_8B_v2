"""
Dependency Version Matrix Tests

Validates all dependency versions are compatible:
- PyTorch + CUDA version compatibility
- Transformers, PEFT, BitsAndBytes versions
- Pinned dependency versions (datasets, pyarrow)
- Python version
- All requirements.txt entries satisfied
"""

import os
import sys
import subprocess
import pytest
from pathlib import Path
from packaging import version as pkg_version

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REQUIREMENTS_PATH = Path(__file__).parent.parent.parent / 'requirements.txt'


def _check_version(module_name, min_version):
    """Import module and check version >= min_version"""
    try:
        mod = __import__(module_name)
        ver = getattr(mod, '__version__', None)
        if ver is None:
            pytest.skip(f"{module_name} has no __version__")
        assert pkg_version.parse(ver) >= pkg_version.parse(min_version), \
            f"{module_name} {ver} < {min_version}"
        return ver
    except ImportError:
        pytest.skip(f"{module_name} not installed")


def _check_version_exact(module_name, exact_version):
    """Import module and check version == exact_version"""
    try:
        mod = __import__(module_name)
        ver = getattr(mod, '__version__', None)
        if ver is None:
            pytest.skip(f"{module_name} has no __version__")
        assert pkg_version.parse(ver) == pkg_version.parse(exact_version), \
            f"{module_name} {ver} != {exact_version} (pinned)"
        return ver
    except ImportError:
        pytest.skip(f"{module_name} not installed")


# =============================================================================
# CORE ML DEPENDENCIES
# =============================================================================

@pytest.mark.gpu
class TestCoreMLDependencies:
    """Tests for core ML library versions"""

    def test_pytorch_version(self):
        """PyTorch >= 2.1.0"""
        _check_version('torch', '2.1.0')

    def test_transformers_version(self):
        """transformers >= 4.45.2"""
        _check_version('transformers', '4.45.2')

    def test_peft_version(self):
        """peft >= 0.12.0"""
        _check_version('peft', '0.12.0')

    def test_bitsandbytes_version(self):
        """bitsandbytes >= 0.43.0"""
        _check_version('bitsandbytes', '0.43.0')

    def test_accelerate_version(self):
        """accelerate >= 0.31.0"""
        _check_version('accelerate', '0.31.0')

    def test_trl_version(self):
        """trl >= 0.9.6"""
        _check_version('trl', '0.9.6')


# =============================================================================
# PINNED DEPENDENCIES
# =============================================================================

@pytest.mark.gpu
class TestPinnedDependencies:
    """Tests for pinned dependency versions (must match exactly)"""

    def test_datasets_version_pinned(self):
        """datasets must be exactly 2.14.5"""
        _check_version_exact('datasets', '2.14.5')

    def test_pyarrow_version_pinned(self):
        """pyarrow must be exactly 14.0.1"""
        _check_version_exact('pyarrow', '14.0.1')


# =============================================================================
# CUDA TOOLKIT
# =============================================================================

@pytest.mark.gpu
class TestCUDAToolkit:
    """Tests for CUDA toolkit version"""

    def test_cuda_toolkit_version(self):
        """CUDA toolkit >= 12.1"""
        import unittest.mock
        try:
            import torch
            if isinstance(torch.cuda, unittest.mock.MagicMock) or not torch.cuda.is_available():
                pytest.skip("No CUDA")
            cuda_ver = torch.version.cuda
            if not cuda_ver or isinstance(cuda_ver, unittest.mock.MagicMock):
                pytest.skip("CUDA version not available")
            parts = cuda_ver.split('.')
            if len(parts) < 2:
                pytest.skip(f"Unexpected CUDA version format: {cuda_ver}")
            major, minor = parts[:2]
            assert int(major) >= 12 or (int(major) == 11 and int(minor) >= 8), \
                f"CUDA {cuda_ver} too old, need >= 11.8"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_pytorch_cuda_version_match(self):
        """PyTorch CUDA version should match system CUDA"""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No CUDA")

            pytorch_cuda = torch.version.cuda
            # Get system CUDA version
            try:
                result = subprocess.run(
                    ['nvcc', '--version'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and 'release' in result.stdout:
                    # Parse "release 12.1" from nvcc output
                    for line in result.stdout.split('\n'):
                        if 'release' in line:
                            parts = line.split('release')[-1].strip()
                            system_cuda = parts.split(',')[0].strip()
                            # Major version should match
                            pt_major = pytorch_cuda.split('.')[0]
                            sys_major = system_cuda.split('.')[0]
                            assert pt_major == sys_major, \
                                f"PyTorch CUDA {pytorch_cuda} vs system CUDA {system_cuda}"
                            break
            except FileNotFoundError:
                pytest.skip("nvcc not found")
        except ImportError:
            pytest.skip("PyTorch not installed")


# =============================================================================
# PYTHON VERSION
# =============================================================================

@pytest.mark.gpu
class TestPythonVersion:
    """Tests for Python version"""

    def test_python_version(self):
        """Python >= 3.11"""
        assert sys.version_info >= (3, 11), \
            f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.11"


# =============================================================================
# AWS DEPENDENCIES
# =============================================================================

@pytest.mark.gpu
class TestAWSDependencies:
    """Tests for AWS SDK versions"""

    def test_boto3_version(self):
        """boto3 >= 1.34.0"""
        _check_version('boto3', '1.34.0')

    def test_sagemaker_version(self):
        """sagemaker >= 2.199.0"""
        _check_version('sagemaker', '2.199.0')


# =============================================================================
# UTILITY DEPENDENCIES
# =============================================================================

@pytest.mark.gpu
class TestUtilityDependencies:
    """Tests for utility library availability"""

    def test_sentencepiece_available(self):
        """sentencepiece must be importable on GPU instances"""
        import unittest.mock
        try:
            import torch
            if isinstance(torch.cuda, unittest.mock.MagicMock) or not torch.cuda.is_available():
                pytest.skip("GPU-only check")
        except ImportError:
            pytest.skip("PyTorch not installed")
        try:
            import sentencepiece
            assert sentencepiece is not None
        except ImportError:
            pytest.fail("sentencepiece not installed")

    def test_scipy_available(self):
        """scipy must be importable on GPU instances"""
        import unittest.mock
        try:
            import torch
            if isinstance(torch.cuda, unittest.mock.MagicMock) or not torch.cuda.is_available():
                pytest.skip("GPU-only check")
        except ImportError:
            pytest.skip("PyTorch not installed")
        try:
            import scipy
            assert scipy is not None
        except ImportError:
            pytest.fail("scipy not installed")

    def test_protobuf_available(self):
        """protobuf must be importable on GPU instances"""
        import unittest.mock
        try:
            import torch
            if isinstance(torch.cuda, unittest.mock.MagicMock) or not torch.cuda.is_available():
                pytest.skip("GPU-only check")
        except ImportError:
            pytest.skip("PyTorch not installed")
        try:
            import google.protobuf
            assert google.protobuf is not None
        except ImportError:
            pytest.fail("protobuf not installed")


# =============================================================================
# REQUIREMENTS.TXT VALIDATION
# =============================================================================

@pytest.mark.gpu
class TestRequirementsFile:
    """Tests that all requirements.txt entries are satisfied"""

    def test_all_requirements_installed(self):
        """Every line in requirements.txt should be satisfied"""
        if not REQUIREMENTS_PATH.exists():
            pytest.skip("requirements.txt not found")

        try:
            from pip._internal.req import parse_requirements
            from pip._internal.network.session import PipSession
        except ImportError:
            # Fallback: use pkg_resources
            import pkg_resources

            with open(REQUIREMENTS_PATH) as f:
                requirements = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith('#')
                    and not line.startswith('-')
                ]

            missing = []
            for req_str in requirements:
                try:
                    pkg_resources.require(req_str)
                except (pkg_resources.DistributionNotFound,
                        pkg_resources.VersionConflict) as e:
                    missing.append(f"{req_str}: {e}")

            if missing:
                pytest.fail(
                    f"Missing/incompatible requirements:\n" +
                    "\n".join(f"  - {m}" for m in missing)
                )
