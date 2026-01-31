#!/usr/bin/env python3
"""
GPU Hardware Test Runner - SageMaker Entry Point

This script runs inside a SageMaker GPU training container.
It executes the pytest-based GPU/hardware test suite and reports results.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime


def install_test_deps():
    """Install test dependencies"""
    print("[SETUP] Installing test dependencies...")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install',
         'pytest', 'pytest-timeout', 'pyyaml', 'packaging',
         'bitsandbytes>=0.43.0', 'peft>=0.12.0', 'trl>=0.9.6',
         'accelerate>=0.31.0'],
        check=True,
        capture_output=True,
    )
    print("[SETUP] Dependencies installed.")


def print_environment():
    """Print environment info for debugging"""
    print("\n" + "=" * 70)
    print("GPU HARDWARE TEST ENVIRONMENT")
    print("=" * 70)

    # Python
    print(f"\nPython: {sys.version}")

    # PyTorch + CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                print(f"  GPU {i}: {props.name} ({mem_gb:.1f}GB)")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"\nnvidia-smi:\n{result.stdout.strip()}")
    except FileNotFoundError:
        print("\nnvidia-smi: NOT FOUND")

    # SageMaker env
    sm_env = os.environ.get('SM_TRAINING_ENV')
    if sm_env:
        env_data = json.loads(sm_env)
        print(f"\nSageMaker instance: {env_data.get('resource_config', {}).get('current_instance_type', 'unknown')}")

    # Key packages
    for pkg in ['transformers', 'peft', 'bitsandbytes', 'accelerate', 'trl', 'datasets']:
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

    print("=" * 70 + "\n")


def run_tests(marker='gpu'):
    """Run pytest with specified marker"""
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'test_results.xml')
    json_results = os.path.join(output_dir, 'test_results.json')

    # Run pytest
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/cloud/',
        '-m', marker,
        '-v',
        '--tb=long',
        f'--junitxml={results_file}',
        '--timeout=600',
        '--no-header',
    ]

    print(f"\n[RUN] Executing: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output to CloudWatch logs
        text=True,
    )

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'marker': marker,
        'exit_code': result.returncode,
        'passed': result.returncode == 0,
        'instance_type': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'unknown'),
    }

    with open(json_results, 'w') as f:
        json.dump(summary, f, indent=2)

    # Also save to model dir so it persists
    with open(os.path.join(model_dir, 'gpu_test_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    # SageMaker passes hyperparameters as CLI args
    parser.add_argument('--test-marker', type=str, default='gpu')
    parser.add_argument('--s3-bucket', type=str, default='')
    parser.add_argument('--region', type=str, default='us-east-1')
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))

    args, _ = parser.parse_known_args()

    install_test_deps()
    print_environment()

    exit_code = run_tests(marker=args.test_marker)

    if exit_code != 0:
        print(f"\n[RESULT] Some tests failed (exit code {exit_code})")
        # Don't sys.exit(1) — let SageMaker see the results
    else:
        print("\n[RESULT] All GPU hardware tests passed!")


if __name__ == "__main__":
    main()
