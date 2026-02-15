#!/usr/bin/env python3
"""
SageMaker Processing Job Script for Automotive Data Pipeline

Optimized for ml.c5.xlarge (4 vCPU, 8 GB RAM):
- Parallel file processing with multiprocessing.Pool(4)
- Memory-efficient streaming with imap_unordered
- AWS CLI for fast S3 bulk downloads

Run via SageMaker Processing Job (see launch_processing_job.py)
or locally for testing:
    python scripts/sagemaker_processing.py --workers 4 --output-dir ./data/splits

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import random
import re
import hashlib
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

# ============================================================================
# DATA SOURCES (same as prepare_automotive_data.py)
# ============================================================================
DATA_SOURCES = [
    {"prefix": "tsn_data/", "domain": "tsn", "contexts": [
        "Time-Sensitive Networking (TSN)", "IEEE 802.1Qbv Time-Aware Shaper",
        "IEEE 802.1Qav Credit-Based Shaper", "IEEE 802.1AS Precision Time Protocol",
    ]},
    {"prefix": "avb_data/", "domain": "avb", "contexts": [
        "Audio Video Bridging (AVB)", "AVB Stream Reservation Protocol",
        "AVTP timestamp processing", "gPTP synchronization",
    ]},
    {"prefix": "advanced_academic/carla_autonomous_driving_simulator/", "domain": "carla", "contexts": [
        "CARLA autonomous driving simulator", "vehicle control system",
        "autonomous vehicle perception",
    ]},
    {"prefix": "advanced_embedded/", "domain": "embedded", "contexts": [
        "embedded automotive systems", "bare-metal firmware",
        "hardware abstraction layer", "peripheral driver development",
    ]},
    {"prefix": "phase_2_embedded/", "domain": "embedded_p2", "contexts": [
        "embedded automotive firmware", "MCU peripheral drivers",
        "low-level hardware interface",
    ]},
    {"prefix": "phase_3_embedded/", "domain": "embedded_p3", "contexts": [
        "production embedded firmware", "automotive ECU software",
        "embedded C safety-critical code",
    ]},
    {"prefix": "advanced_rtos/", "domain": "rtos", "contexts": [
        "real-time operating system", "FreeRTOS task management",
        "RTOS scheduling algorithms",
    ]},
    {"prefix": "phase_2_rtos/", "domain": "rtos_p2", "contexts": [
        "FreeRTOS automotive application", "RTOS queue management",
        "real-time interrupt handling",
    ]},
    {"prefix": "phase_3_rtos/", "domain": "rtos_p3", "contexts": [
        "production RTOS firmware", "automotive real-time scheduler",
    ]},
    {"prefix": "nxp_automotive_freertos/", "domain": "nxp_freertos", "contexts": [
        "NXP automotive FreeRTOS", "NXP S32K MCU firmware",
    ]},
    {"prefix": "nxp_s32k_freertos_bsp/", "domain": "nxp_bsp", "contexts": [
        "NXP S32K board support package", "NXP FreeRTOS BSP drivers",
    ]},
    {"prefix": "car_freertos_example/", "domain": "car_freertos", "contexts": [
        "automotive FreeRTOS example", "vehicle ECU FreeRTOS application",
    ]},
    {"prefix": "advanced_middleware/", "domain": "middleware", "contexts": [
        "automotive middleware", "SOME/IP communication",
        "CommonAPI service framework",
    ]},
    {"prefix": "phase_2_middleware/", "domain": "middleware_p2", "contexts": [
        "automotive middleware stack", "inter-ECU communication",
    ]},
    {"prefix": "covesa_commonapi_core_tools/", "domain": "covesa", "contexts": [
        "COVESA CommonAPI tools", "GENIVI middleware framework",
    ]},
    {"prefix": "genivi_candevstudio/", "domain": "genivi", "contexts": [
        "GENIVI CAN development studio", "CAN bus protocol tools",
    ]},
    {"prefix": "advanced_safety/", "domain": "safety", "contexts": [
        "functional safety ISO 26262", "ASIL-compliant code",
        "safety-critical automotive software",
    ]},
    {"prefix": "phase_3_safety/", "domain": "safety_p3", "contexts": [
        "production safety-critical code", "ISO 26262 compliant implementation",
    ]},
    {"prefix": "functional_safety_examples/", "domain": "func_safety", "contexts": [
        "functional safety examples", "safety pattern implementation",
    ]},
    {"prefix": "autosar_learning_project/", "domain": "autosar", "contexts": [
        "AUTOSAR Classic Platform", "AUTOSAR software component",
    ]},
    {"prefix": "phase_2_academic/", "domain": "academic_p2", "contexts": [
        "automotive research code", "vehicle systems academic implementation",
    ]},
    {"prefix": "phase_3_academic/", "domain": "academic_p3", "contexts": [
        "advanced automotive research", "vehicle systems algorithm",
    ]},
    {"prefix": "awesome_vehicle_security/", "domain": "security", "contexts": [
        "vehicle security", "automotive cybersecurity", "CAN bus security",
    ]},
]

CODE_EXTENSIONS = ('.c', '.h', '.cpp', '.cc', '.cxx', '.txt')


# ============================================================================
# FILE PROCESSING FUNCTIONS (run in parallel workers)
# ============================================================================

def read_code_file(file_path: str) -> Optional[str]:
    """Read code file with encoding handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def extract_functions(code: str, language: str = 'c') -> List[Dict]:
    """Extract functions from code for training examples."""
    functions = []
    if language not in ['c', 'cpp', 'h']:
        return functions

    pattern = r'(?:static\s+)?(?:inline\s+)?(?:const\s+)?(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{'

    for match in re.finditer(pattern, code):
        return_type = match.group(1).strip()
        func_name = match.group(2)
        params = match.group(3).strip()

        start = match.end() - 1
        brace_count = 1
        end = start + 1

        while end < len(code) and brace_count > 0:
            if code[end] == '{':
                brace_count += 1
            elif code[end] == '}':
                brace_count -= 1
            end += 1

        func_body = code[match.start():end]

        if len(func_body) < 50:
            continue

        functions.append({
            'name': func_name,
            'return_type': return_type,
            'params': params,
            'body': func_body,
            'lines': func_body.count('\n')
        })

    return functions


def generate_prompt_from_function(func: Dict, context: str = "") -> Dict:
    """Generate a training prompt from a function."""
    func_name = func['name']
    return_type = func['return_type']
    params = func['params']
    body = func['body']

    prompt_templates = [
        f"Generate a C function named '{func_name}' that returns {return_type} with parameters ({params})",
        f"Implement the function '{func_name}' for automotive embedded systems",
        f"Write C code for the '{func_name}' function used in {context}",
        f"Create an implementation of '{func_name}' following automotive coding standards"
    ]

    prompt = random.choice(prompt_templates)
    if context:
        prompt += f". Context: {context}"

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": body}
        ],
        "metadata": {
            "function_name": func_name,
            "context": context,
            "source": "automotive_codebase"
        }
    }


def generate_code_completion_prompt(code: str, split_ratio: float = 0.5) -> Optional[Dict]:
    """Generate code completion training example."""
    lines = code.strip().split('\n')
    split_point = int(len(lines) * split_ratio)

    if split_point < 3 or len(lines) - split_point < 3:
        return None

    prefix = '\n'.join(lines[:split_point])
    suffix = '\n'.join(lines[split_point:])

    prompt = f"Complete the following automotive code:\n\n```c\n{prefix}\n```\n\nProvide the continuation:"

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```c\n{suffix}\n```"}
        ],
        "metadata": {
            "type": "code_completion",
            "source": "automotive_codebase"
        }
    }


def process_single_file(args: Tuple[str, str, List[str]]) -> List[Dict]:
    """
    Process a single file and return training examples.
    This function runs in a worker process.

    Args:
        args: Tuple of (file_path, domain, contexts)

    Returns:
        List of training examples
    """
    file_path, domain, contexts = args
    examples = []

    code = read_code_file(file_path)
    if not code:
        return examples

    lang = 'cpp' if file_path.endswith(('.cpp', '.cc', '.cxx')) else 'c'
    functions = extract_functions(code, language=lang)

    for func in functions:
        context = random.choice(contexts)
        example = generate_prompt_from_function(func, context)
        example['metadata']['domain'] = domain
        examples.append(example)

    if len(code) > 200:
        completion = generate_code_completion_prompt(code)
        if completion:
            completion['metadata']['domain'] = domain
            examples.append(completion)

    return examples


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def sync_from_s3(s3_bucket: str, local_dir: Path, region: str) -> bool:
    """Download data from S3 using AWS CLI (fast bulk sync)."""
    s3_uri = f"s3://{s3_bucket}/"

    include_args = []
    for ext in CODE_EXTENSIONS:
        include_args.extend(['--include', f'*{ext}'])

    cmd = [
        'aws', 's3', 'sync',
        s3_uri,
        str(local_dir),
        '--exclude', '*',
        *include_args,
        '--region', region,
    ]

    print(f"\n[SageMaker] Syncing s3://{s3_bucket}/ -> {local_dir}")
    print(f"[SageMaker] Extensions: {', '.join(CODE_EXTENSIONS)}")
    print("-" * 70)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        file_count = 0
        for line in process.stdout:
            line = line.strip()
            if line and line.startswith('download:'):
                file_count += 1
                if file_count % 5000 == 0:
                    print(f"[SageMaker] Downloaded {file_count:,} files...")

        process.wait()
        print(f"\n[SageMaker] Download complete: {file_count:,} files")
        return process.returncode == 0

    except FileNotFoundError:
        print("[SageMaker] ERROR: AWS CLI not found")
        return False
    except Exception as e:
        print(f"[SageMaker] ERROR: {e}")
        return False


def get_files_for_source(local_dir: Path, source: Dict) -> List[str]:
    """Get list of local files for a given data source."""
    prefix = source['prefix'].rstrip('/')
    source_dir = local_dir / prefix

    if not source_dir.exists():
        return []

    files = []
    for ext in CODE_EXTENSIONS:
        files.extend(str(p) for p in source_dir.rglob(f'*{ext}'))

    return files


def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate examples based on content hash."""
    seen = set()
    unique = []

    for example in examples:
        content = json.dumps(example['messages'], sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(example)

    print(f"[SageMaker] Deduplicated: {len(examples):,} -> {len(unique):,}")
    return unique


def save_jsonl(examples: List[Dict], output_path: Path):
    """Save examples to JSONL file."""
    with open(output_path, 'w') as f:
        for example in examples:
            output = {"messages": example["messages"]}
            f.write(json.dumps(output) + '\n')

    size_mb = output_path.stat().st_size / 1e6
    print(f"[SageMaker] Saved {len(examples):,} examples to {output_path} ({size_mb:.1f} MB)")


def run_pipeline(
    s3_bucket: str,
    region: str,
    output_dir: Path,
    workers: int,
    train_ratio: float = 0.9,
    skip_download: bool = False,
):
    """
    Run the full pipeline with parallel processing and memory-efficient streaming.

    Memory optimization: Instead of accumulating all examples in RAM (~200K examples),
    we stream processed examples to disk, then deduplicate via streaming.
    This keeps RAM usage under ~1 GB even for 9 GB of raw data.

    Flow:
    1. Download files to /tmp (disk) via AWS CLI
    2. Process files in parallel, stream examples directly to disk
    3. Deduplicate by streaming (only hashes in memory, ~50 MB for 200K examples)
    4. Shuffle indices and split to train/val
    """
    pipeline_start = time.time()

    # Directories
    local_data_dir = Path('/tmp/automotive_data')
    temp_dir = Path('/tmp/processing_temp')
    local_data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("[SageMaker] AUTOMOTIVE DATA PIPELINE (Memory-Optimized)")
    print(f"[SageMaker] Workers: {workers} (vCPUs)")
    print(f"[SageMaker] Data sources: {len(DATA_SOURCES)}")
    print(f"[SageMaker] Output: {output_dir}")
    print("=" * 70)

    # Step 1: Download from S3
    if not skip_download:
        success = sync_from_s3(s3_bucket, local_data_dir, region)
        if not success:
            print("[SageMaker] WARNING: S3 sync had issues, continuing...")

    # Step 2: Collect all files to process
    print("\n[SageMaker] Collecting files to process...")
    all_file_args = []
    source_file_counts = {}

    for source in DATA_SOURCES:
        domain = source['domain']
        contexts = source['contexts']
        files = get_files_for_source(local_data_dir, source)
        source_file_counts[domain] = len(files)

        for f in files:
            all_file_args.append((f, domain, contexts))

    total_files = len(all_file_args)
    print(f"[SageMaker] Total files to process: {total_files:,}")

    for domain, count in sorted(source_file_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {domain:25s} {count:>8,}")

    # Step 3: Process files in parallel, STREAM to disk (not RAM)
    print(f"\n[SageMaker] Processing with {workers} workers (streaming to disk)...")
    process_start = time.time()

    raw_output = temp_dir / "raw_examples.jsonl"
    completed = 0
    total_examples = 0

    with open(raw_output, 'w') as out_f:
        with Pool(workers) as pool:
            for examples in pool.imap_unordered(process_single_file, all_file_args, chunksize=100):
                # Stream each batch directly to disk - NOT accumulating in RAM
                for example in examples:
                    out_f.write(json.dumps({"messages": example["messages"]}) + '\n')
                    total_examples += 1

                completed += 1
                if completed % 10000 == 0:
                    print(f"[SageMaker] Processed {completed:,}/{total_files:,} files, {total_examples:,} examples...")

    process_elapsed = time.time() - process_start
    raw_size_mb = raw_output.stat().st_size / 1e6
    print(f"[SageMaker] Processing complete: {total_examples:,} examples in {process_elapsed:.1f}s ({raw_size_mb:.1f} MB)")

    # Step 4: Deduplicate by streaming (only hashes in memory ~50MB)
    print("\n[SageMaker] Deduplicating (streaming)...")
    seen_hashes = set()
    dedup_output = temp_dir / "dedup_examples.jsonl"
    unique_count = 0

    with open(raw_output, 'r') as in_f, open(dedup_output, 'w') as out_f:
        for line in in_f:
            content_hash = hashlib.md5(line.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                out_f.write(line)
                unique_count += 1

    print(f"[SageMaker] Deduplicated: {total_examples:,} -> {unique_count:,}")

    # Clean up raw file to save disk space
    raw_output.unlink()
    del seen_hashes  # Free memory

    # Step 5: Shuffle and split (only indices in memory)
    print("\n[SageMaker] Shuffling and splitting...")

    random.seed(42)
    indices = list(range(unique_count))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = set(indices[:split_idx])

    # Single pass: write to train or val based on shuffled assignment
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(dedup_output, 'r') as in_f, \
         open(train_path, 'w') as train_f, \
         open(val_path, 'w') as val_f:

        for i, line in enumerate(in_f):
            if i in train_indices:
                train_f.write(line)
            else:
                val_f.write(line)

    # Clean up dedup file
    dedup_output.unlink()

    train_count = len(train_indices)
    val_count = unique_count - train_count
    train_size_mb = train_path.stat().st_size / 1e6
    val_size_mb = val_path.stat().st_size / 1e6

    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("[SageMaker] PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total files: {total_files:,}")
    print(f"  Total examples: {unique_count:,}")
    print(f"  Train: {train_count:,} ({train_size_mb:.1f} MB)")
    print(f"  Val: {val_count:,} ({val_size_mb:.1f} MB)")
    print(f"  Duration: {pipeline_elapsed / 60:.1f} minutes")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker Processing Job for Automotive Data")
    parser.add_argument('--s3-bucket', type=str, default='granite-8b-unified-automotive-data',
                        help='Source S3 bucket')
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_REGION', 'us-east-1'),
                        help='AWS region')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output',
                        help='Output directory for splits')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (match vCPUs)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Train/val split ratio')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip S3 download (use existing local data)')

    args = parser.parse_args()

    run_pipeline(
        s3_bucket=args.s3_bucket,
        region=args.region,
        output_dir=Path(args.output_dir),
        workers=args.workers,
        train_ratio=args.train_ratio,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
