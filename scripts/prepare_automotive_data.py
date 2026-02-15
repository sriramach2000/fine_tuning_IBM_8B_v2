#!/usr/bin/env python3
"""
Automotive Data Pipeline for AVB/TSN Code Generation

This script processes automotive code data from S3 for fine-tuning Granite-8B:
1. Downloads data from S3 using AWS CLI (fast bulk sync)
2. Parses and extracts code patterns
3. Generates training prompts
4. Creates train/val splits in JSONL format
5. Uploads processed splits to S3 for Colab consumption

Usage:
    # Full pipeline with S3 upload (run once locally, then use Colab for training)
    python scripts/prepare_automotive_data.py --upload-splits

    # Test with small sample
    python scripts/prepare_automotive_data.py --sample-only

    # Skip download (process already-downloaded data)
    python scripts/prepare_automotive_data.py --no-download --upload-splits

Target: Embedded Automotive Code Generation (AVB/TSN)

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
from datetime import datetime
from collections import defaultdict

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class AutomotiveDataPipeline:
    """
    Process automotive code data for Granite-8B fine-tuning.

    Handles TSN, AVB, and CARLA autonomous driving code.
    Uses AWS CLI for fast bulk downloads.
    """

    # All automotive data sources with their S3 prefixes, domain labels, and contexts
    DATA_SOURCES = [
        # TSN / AVB core
        {"prefix": "tsn_data/", "domain": "tsn", "contexts": [
            "Time-Sensitive Networking (TSN)", "IEEE 802.1Qbv Time-Aware Shaper",
            "IEEE 802.1Qav Credit-Based Shaper", "IEEE 802.1AS Precision Time Protocol",
            "TSN stream scheduling",
        ]},
        {"prefix": "avb_data/", "domain": "avb", "contexts": [
            "Audio Video Bridging (AVB)", "AVB Stream Reservation Protocol",
            "AVB audio streaming", "AVTP timestamp processing", "gPTP synchronization",
        ]},
        # CARLA simulator
        {"prefix": "advanced_academic/carla_autonomous_driving_simulator/", "domain": "carla", "contexts": [
            "CARLA autonomous driving simulator", "vehicle control system",
            "sensor data processing", "autonomous vehicle perception",
            "vehicle physics simulation",
        ]},
        # Embedded systems
        {"prefix": "advanced_embedded/", "domain": "embedded", "contexts": [
            "embedded automotive systems", "bare-metal firmware",
            "hardware abstraction layer", "peripheral driver development",
            "embedded C real-time systems",
        ]},
        {"prefix": "phase_2_embedded/", "domain": "embedded_p2", "contexts": [
            "embedded automotive firmware", "MCU peripheral drivers",
            "low-level hardware interface", "embedded systems programming",
        ]},
        {"prefix": "phase_3_embedded/", "domain": "embedded_p3", "contexts": [
            "production embedded firmware", "automotive ECU software",
            "embedded C safety-critical code", "vehicle embedded controller",
        ]},
        # RTOS
        {"prefix": "advanced_rtos/", "domain": "rtos", "contexts": [
            "real-time operating system", "FreeRTOS task management",
            "RTOS scheduling algorithms", "automotive RTOS development",
            "real-time task synchronization",
        ]},
        {"prefix": "phase_2_rtos/", "domain": "rtos_p2", "contexts": [
            "FreeRTOS automotive application", "RTOS queue management",
            "real-time interrupt handling", "RTOS memory management",
        ]},
        {"prefix": "phase_3_rtos/", "domain": "rtos_p3", "contexts": [
            "production RTOS firmware", "automotive real-time scheduler",
            "RTOS semaphore and mutex patterns",
        ]},
        {"prefix": "nxp_automotive_freertos/", "domain": "nxp_freertos", "contexts": [
            "NXP automotive FreeRTOS", "NXP S32K MCU firmware",
            "NXP automotive BSP", "NXP real-time driver",
        ]},
        {"prefix": "nxp_s32k_freertos_bsp/", "domain": "nxp_bsp", "contexts": [
            "NXP S32K board support package", "NXP FreeRTOS BSP drivers",
            "NXP automotive peripheral initialization", "NXP MCU startup code",
        ]},
        {"prefix": "car_freertos_example/", "domain": "car_freertos", "contexts": [
            "automotive FreeRTOS example", "vehicle ECU FreeRTOS application",
        ]},
        # Middleware
        {"prefix": "advanced_middleware/", "domain": "middleware", "contexts": [
            "automotive middleware", "SOME/IP communication",
            "CommonAPI service framework", "vehicle service-oriented architecture",
        ]},
        {"prefix": "phase_2_middleware/", "domain": "middleware_p2", "contexts": [
            "automotive middleware stack", "inter-ECU communication",
            "vehicle middleware services", "automotive IPC framework",
        ]},
        {"prefix": "covesa_commonapi_core_tools/", "domain": "covesa", "contexts": [
            "COVESA CommonAPI tools", "GENIVI middleware framework",
            "automotive service interface", "CommonAPI code generation",
        ]},
        {"prefix": "genivi_candevstudio/", "domain": "genivi", "contexts": [
            "GENIVI CAN development studio", "CAN bus protocol tools",
            "automotive CAN interface", "vehicle network diagnostics",
        ]},
        # Safety
        {"prefix": "advanced_safety/", "domain": "safety", "contexts": [
            "functional safety ISO 26262", "ASIL-compliant code",
            "safety-critical automotive software", "MISRA C compliance",
        ]},
        {"prefix": "phase_3_safety/", "domain": "safety_p3", "contexts": [
            "production safety-critical code", "ISO 26262 compliant implementation",
            "automotive safety monitoring", "fault-tolerant vehicle software",
        ]},
        {"prefix": "functional_safety_examples/", "domain": "func_safety", "contexts": [
            "functional safety examples", "safety pattern implementation",
            "redundancy and fault detection",
        ]},
        # AUTOSAR
        {"prefix": "autosar_learning_project/", "domain": "autosar", "contexts": [
            "AUTOSAR Classic Platform", "AUTOSAR software component",
            "AUTOSAR RTE interface", "AUTOSAR BSW module",
        ]},
        # Academic / research
        {"prefix": "phase_2_academic/", "domain": "academic_p2", "contexts": [
            "automotive research code", "vehicle systems academic implementation",
            "automotive protocol reference",
        ]},
        {"prefix": "phase_3_academic/", "domain": "academic_p3", "contexts": [
            "advanced automotive research", "vehicle systems algorithm",
            "automotive academic reference implementation",
        ]},
        # Vehicle security
        {"prefix": "awesome_vehicle_security/", "domain": "security", "contexts": [
            "vehicle security", "automotive cybersecurity",
            "CAN bus security", "vehicle intrusion detection",
        ]},
    ]

    CODE_EXTENSIONS = ('.c', '.h', '.cpp', '.cc', '.cxx', '.txt')

    def __init__(
        self,
        s3_bucket: str = "granite-8b-unified-automotive-data",
        region: str = "us-east-1",
        local_data_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        splits_dir: str = "./data/splits",
    ):
        self.s3_bucket = s3_bucket
        self.region = region
        self.local_data_dir = Path(local_data_dir)
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)

        # Create directories
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # S3 client for non-bulk operations
        self.s3_client = boto3.client('s3', region_name=region)

        print(f"[DataPipeline] S3 Bucket: {s3_bucket}")
        print(f"[DataPipeline] Region: {region}")
        print(f"[DataPipeline] Local data dir: {self.local_data_dir.absolute()}")

    def sync_from_s3(self, prefix: str = "") -> bool:
        """
        Download data from S3 using AWS CLI 's3 sync' (much faster than boto3).
        Includes only code file extensions.
        """
        s3_uri = f"s3://{self.s3_bucket}/{prefix}"
        local_dir = self.local_data_dir / prefix.rstrip('/')

        # Build include patterns for code files
        include_args = []
        for ext in self.CODE_EXTENSIONS:
            include_args.extend(['--include', f'*{ext}'])

        cmd = [
            'aws', 's3', 'sync',
            s3_uri,
            str(local_dir),
            '--exclude', '*',  # Exclude everything first
            *include_args,     # Then include only code files
            '--region', self.region,
        ]

        print(f"\n[DataPipeline] Syncing {s3_uri} -> {local_dir}")
        print(f"[DataPipeline] Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"[DataPipeline] WARNING: aws s3 sync returned {result.returncode}")
                return False
            return True
        except FileNotFoundError:
            print("[DataPipeline] ERROR: AWS CLI not found. Install with: pip install awscli")
            return False
        except Exception as e:
            print(f"[DataPipeline] ERROR: {e}")
            return False

    def sync_all_sources(self) -> bool:
        """Sync all data sources from S3 using AWS CLI."""
        print("\n" + "=" * 70)
        print("[DataPipeline] DOWNLOADING DATA FROM S3 (using AWS CLI)")
        print("=" * 70)

        # Sync entire bucket with code file filter
        s3_uri = f"s3://{self.s3_bucket}/"

        include_args = []
        for ext in self.CODE_EXTENSIONS:
            include_args.extend(['--include', f'*{ext}'])

        cmd = [
            'aws', 's3', 'sync',
            s3_uri,
            str(self.local_data_dir),
            '--exclude', '*',
            *include_args,
            '--region', self.region,
        ]

        print(f"[DataPipeline] Syncing entire bucket (code files only)...")
        print(f"[DataPipeline] Source: {s3_uri}")
        print(f"[DataPipeline] Destination: {self.local_data_dir.absolute()}")
        print(f"[DataPipeline] Extensions: {', '.join(self.CODE_EXTENSIONS)}")
        print("-" * 70)

        try:
            # Run with output streaming to show progress
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
                if line:
                    # Count downloaded files
                    if line.startswith('download:'):
                        file_count += 1
                        if file_count % 1000 == 0:
                            print(f"[DataPipeline] Downloaded {file_count:,} files...")
                    elif not line.startswith('Completed'):
                        print(line)

            process.wait()

            if process.returncode != 0:
                print(f"[DataPipeline] WARNING: aws s3 sync returned {process.returncode}")
                return False

            print(f"\n[DataPipeline] Download complete: {file_count:,} files")
            return True

        except FileNotFoundError:
            print("[DataPipeline] ERROR: AWS CLI not found.")
            print("  Install with: pip install awscli")
            print("  Or: brew install awscli")
            return False
        except Exception as e:
            print(f"[DataPipeline] ERROR: {e}")
            return False

    def get_local_files_for_source(self, source: Dict) -> List[str]:
        """Get list of local files for a given data source."""
        prefix = source['prefix'].rstrip('/')
        local_dir = self.local_data_dir / prefix

        if not local_dir.exists():
            return []

        files = []
        for ext in self.CODE_EXTENSIONS:
            files.extend(str(p) for p in local_dir.rglob(f'*{ext}'))

        return files

    def read_code_file(self, file_path: str) -> Optional[str]:
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

    def extract_functions(self, code: str, language: str = 'c') -> List[Dict]:
        """Extract functions from code for training examples."""
        functions = []

        if language in ['c', 'cpp', 'h']:
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

    def generate_prompt_from_function(self, func: Dict, context: str = "") -> Dict:
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

    def generate_code_completion_prompt(self, code: str, split_ratio: float = 0.5) -> Optional[Dict]:
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

    def process_embedded_data(self, files: List[str], domain: str, contexts: List[str]) -> List[Dict]:
        """Generic processor for embedded/automotive code files."""
        examples = []

        for file_path in tqdm(files, desc=f"Processing {domain}", leave=False):
            code = self.read_code_file(file_path)
            if not code:
                continue

            lang = 'cpp' if file_path.endswith(('.cpp', '.cc', '.cxx')) else 'c'
            functions = self.extract_functions(code, language=lang)

            for func in functions:
                context = random.choice(contexts)
                example = self.generate_prompt_from_function(func, context)
                example['metadata']['domain'] = domain
                examples.append(example)

            if len(code) > 200:
                completion = self.generate_code_completion_prompt(code)
                if completion:
                    completion['metadata']['domain'] = domain
                    examples.append(completion)

        return examples

    def deduplicate(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples based on content hash."""
        seen = set()
        unique = []

        for example in examples:
            content = json.dumps(example['messages'], sort_keys=True)
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(example)

        print(f"[DataPipeline] Deduplicated: {len(examples):,} -> {len(unique):,}")
        return unique

    def create_splits(
        self,
        examples: List[Dict],
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Create train/val splits."""
        random.seed(seed)
        random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        train = examples[:split_idx]
        val = examples[split_idx:]

        return train, val

    def save_jsonl(self, examples: List[Dict], output_path: Path):
        """Save examples to JSONL file."""
        with open(output_path, 'w') as f:
            for example in examples:
                output = {"messages": example["messages"]}
                f.write(json.dumps(output) + '\n')

        size_mb = output_path.stat().st_size / 1e6
        print(f"[DataPipeline] Saved {len(examples):,} examples to {output_path} ({size_mb:.1f} MB)")

    def run_pipeline(
        self,
        download_data: bool = True,
        max_files_per_type: int = 0,
        train_ratio: float = 0.9
    ):
        """Run the full data pipeline across all automotive data sources."""
        pipeline_start = time.time()

        print("\n" + "=" * 70)
        print("[DataPipeline] AUTOMOTIVE DATA PIPELINE")
        print(f"[DataPipeline] Processing {len(self.DATA_SOURCES)} data sources")
        print(f"[DataPipeline] Max files per type: {'unlimited' if max_files_per_type == 0 else max_files_per_type}")
        print("=" * 70)

        # Step 1: Download from S3 using AWS CLI
        if download_data:
            success = self.sync_all_sources()
            if not success:
                print("[DataPipeline] WARNING: S3 sync had issues, continuing with available files...")

        # Step 2: Process each data source
        print("\n" + "=" * 70)
        print("[DataPipeline] PROCESSING DOWNLOADED DATA")
        print("=" * 70)

        all_examples = []
        source_stats = {}
        total_files = 0

        for i, source in enumerate(self.DATA_SOURCES, 1):
            domain = source['domain']
            contexts = source['contexts']

            files = self.get_local_files_for_source(source)

            if max_files_per_type > 0:
                files = files[:max_files_per_type]

            total_files += len(files)

            if not files:
                source_stats[domain] = 0
                continue

            print(f"\n[{i}/{len(self.DATA_SOURCES)}] {domain}: {len(files):,} files")
            examples = self.process_embedded_data(files, domain, contexts)
            all_examples.extend(examples)
            source_stats[domain] = len(examples)

        # Print summary by source
        print("\n" + "-" * 70)
        print("[DataPipeline] Examples by source:")
        for domain, count in sorted(source_stats.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {domain:25s} {count:>8,}")
        print(f"  {'TOTAL (before dedup)':25s} {len(all_examples):>8,}")
        print("-" * 70)

        # Deduplicate
        all_examples = self.deduplicate(all_examples)

        # Create splits
        train_examples, val_examples = self.create_splits(all_examples, train_ratio)

        # Save
        self.save_jsonl(train_examples, self.splits_dir / "train.jsonl")
        self.save_jsonl(val_examples, self.splits_dir / "val.jsonl")

        pipeline_elapsed = time.time() - pipeline_start

        print("\n" + "=" * 70)
        print("[DataPipeline] PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Total files processed: {total_files:,}")
        print(f"  Total examples: {len(all_examples):,}")
        print(f"  Train examples: {len(train_examples):,}")
        print(f"  Val examples: {len(val_examples):,}")
        print(f"  Output directory: {self.splits_dir}")
        print(f"  Duration: {pipeline_elapsed / 60:.1f} minutes")
        print("=" * 70)

        return train_examples, val_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare automotive data for Granite-8B fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with S3 upload (run once, then use Colab for training)
  python scripts/prepare_automotive_data.py --upload-splits

  # Test with small sample first
  python scripts/prepare_automotive_data.py --sample-only

  # Skip download (process already-downloaded data)
  python scripts/prepare_automotive_data.py --no-download --upload-splits

  # Full pipeline without upload (just generate local splits)
  python scripts/prepare_automotive_data.py
        """
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default='granite-8b-unified-automotive-data',
        help='S3 bucket name (source data)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=os.environ.get('AWS_REGION', 'us-east-1'),
        help='AWS region'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=0,
        help='Max files per data type (0 = no limit)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Train/val split ratio'
    )
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Only process a small sample for testing (10 files per type)'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip S3 download (use existing local files)'
    )
    parser.add_argument(
        '--upload-splits',
        action='store_true',
        help='Upload processed train.jsonl/val.jsonl to S3 output bucket'
    )
    parser.add_argument(
        '--output-bucket',
        type=str,
        default='granite-8b-training-outputs',
        help='S3 bucket for uploading processed splits'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='runs/data/splits',
        help='S3 prefix for uploaded splits'
    )

    args = parser.parse_args()

    max_files = 10 if args.sample_only else args.max_files

    print("\n" + "=" * 70)
    print("AUTOMOTIVE DATA PIPELINE - PRE-PROCESSING")
    print("=" * 70)
    print(f"Source bucket: s3://{args.s3_bucket}")
    print(f"Output bucket: s3://{args.output_bucket}/{args.output_prefix}/")
    print(f"Max files/type: {'10 (sample)' if args.sample_only else ('unlimited' if max_files == 0 else max_files)}")
    print(f"Download from S3: {not args.no_download}")
    print(f"Upload to S3: {args.upload_splits}")
    print("=" * 70)

    pipeline = AutomotiveDataPipeline(
        s3_bucket=args.s3_bucket,
        region=args.region,
    )

    pipeline.run_pipeline(
        download_data=not args.no_download,
        max_files_per_type=max_files,
        train_ratio=args.train_ratio
    )

    # Upload splits to S3 if requested
    if args.upload_splits:
        print("\n" + "=" * 70)
        print("[DataPipeline] UPLOADING SPLITS TO S3")
        print("=" * 70)

        for split_name in ['train.jsonl', 'val.jsonl']:
            local_path = pipeline.splits_dir / split_name
            if local_path.exists():
                s3_uri = f"s3://{args.output_bucket}/{args.output_prefix}/{split_name}"
                size_mb = local_path.stat().st_size / 1e6
                print(f"Uploading {split_name} ({size_mb:.1f} MB) -> {s3_uri}")

                cmd = ['aws', 's3', 'cp', str(local_path), s3_uri, '--region', args.region]
                subprocess.run(cmd, check=True)
            else:
                print(f"[WARNING] {local_path} not found, skipping upload")

        print(f"\nSplits uploaded to s3://{args.output_bucket}/{args.output_prefix}/")
        print("\n" + "=" * 70)
        print("DONE! Colab notebook will now use the fast path:")
        print("  - Downloads cached splits in seconds")
        print("  - Skips the full download/processing step")
        print("=" * 70)


if __name__ == "__main__":
    main()
