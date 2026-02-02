#!/usr/bin/env python3
"""
Automotive Data Pipeline for AVB/TSN Code Generation

This script processes automotive code data from S3 for fine-tuning Granite-8B:
1. Downloads TSN/AVB/CARLA code from S3
2. Parses and extracts code patterns
3. Generates training prompts
4. Creates train/val splits in JSONL format

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
    pass  # python-dotenv not installed, environment variables must be set externally


class AutomotiveDataPipeline:
    """
    Process automotive code data for Granite-8B fine-tuning.

    Handles TSN, AVB, and CARLA autonomous driving code.
    """

    def __init__(
        self,
        s3_bucket: str = "granite-8b-unified-automotive-data",
        region: str = "us-east-1",
        local_data_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        splits_dir: str = "./data/splits"
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

        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=region)

        print(f"[DataPipeline] S3 Bucket: {s3_bucket}")
        print(f"[DataPipeline] Region: {region}")

    def list_s3_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict]:
        """List objects in S3 bucket with given prefix"""
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
                if len(objects) >= max_keys:
                    break
        except ClientError as e:
            print(f"[DataPipeline] Error listing S3: {e}")

        return objects

    def download_file(self, s3_key: str, local_path: Optional[Path] = None) -> Optional[str]:
        """Download a single file from S3"""
        if local_path is None:
            local_path = self.local_data_dir / s3_key.replace('/', '_')

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            return str(local_path)
        except ClientError as e:
            print(f"[DataPipeline] Error downloading {s3_key}: {e}")
            return None

    def download_prefix(self, prefix: str, max_files: int = 100) -> List[str]:
        """Download files with given prefix from S3"""
        objects = self.list_s3_objects(prefix, max_keys=max_files)
        downloaded = []

        print(f"[DataPipeline] Downloading {len(objects)} files from {prefix}...")

        for obj in tqdm(objects, desc=f"Downloading {prefix}"):
            key = obj['key']
            # Skip directories
            if key.endswith('/'):
                continue

            local_path = self.download_file(key)
            if local_path:
                downloaded.append(local_path)

        return downloaded

    def read_code_file(self, file_path: str) -> Optional[str]:
        """Read code file with encoding handling"""
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
        """Extract functions from code for training examples"""
        functions = []

        if language in ['c', 'cpp', 'h']:
            # C/C++ function pattern
            pattern = r'(?:static\s+)?(?:inline\s+)?(?:const\s+)?(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{'

            for match in re.finditer(pattern, code):
                return_type = match.group(1).strip()
                func_name = match.group(2)
                params = match.group(3).strip()

                # Find function body
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

                # Skip very short functions
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
        """Generate a training prompt from a function"""
        func_name = func['name']
        return_type = func['return_type']
        params = func['params']
        body = func['body']

        # Generate descriptive prompt
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

    def generate_code_completion_prompt(self, code: str, split_ratio: float = 0.5) -> Dict:
        """Generate code completion training example"""
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

    def process_tsn_data(self, files: List[str]) -> List[Dict]:
        """Process TSN protocol files"""
        examples = []

        tsn_contexts = [
            "Time-Sensitive Networking (TSN)",
            "IEEE 802.1Qbv Time-Aware Shaper",
            "IEEE 802.1Qav Credit-Based Shaper",
            "IEEE 802.1AS Precision Time Protocol",
            "TSN stream scheduling"
        ]

        for file_path in tqdm(files, desc="Processing TSN"):
            code = self.read_code_file(file_path)
            if not code:
                continue

            # Extract functions
            functions = self.extract_functions(code)

            for func in functions:
                context = random.choice(tsn_contexts)
                example = self.generate_prompt_from_function(func, context)
                example['metadata']['domain'] = 'tsn'
                examples.append(example)

            # Generate code completion examples
            if len(code) > 200:
                completion = self.generate_code_completion_prompt(code)
                if completion:
                    completion['metadata']['domain'] = 'tsn'
                    examples.append(completion)

        return examples

    def process_avb_data(self, files: List[str]) -> List[Dict]:
        """Process AVB protocol files"""
        examples = []

        avb_contexts = [
            "Audio Video Bridging (AVB)",
            "AVB Stream Reservation Protocol",
            "AVB audio streaming",
            "AVTP timestamp processing",
            "gPTP synchronization"
        ]

        for file_path in tqdm(files, desc="Processing AVB"):
            code = self.read_code_file(file_path)
            if not code:
                continue

            functions = self.extract_functions(code)

            for func in functions:
                context = random.choice(avb_contexts)
                example = self.generate_prompt_from_function(func, context)
                example['metadata']['domain'] = 'avb'
                examples.append(example)

        return examples

    def process_carla_data(self, files: List[str]) -> List[Dict]:
        """Process CARLA autonomous driving files"""
        examples = []

        carla_contexts = [
            "CARLA autonomous driving simulator",
            "vehicle control system",
            "sensor data processing",
            "autonomous vehicle perception",
            "vehicle physics simulation"
        ]

        for file_path in tqdm(files, desc="Processing CARLA"):
            code = self.read_code_file(file_path)
            if not code:
                continue

            # CARLA uses C++
            functions = self.extract_functions(code, language='cpp')

            for func in functions:
                context = random.choice(carla_contexts)
                example = self.generate_prompt_from_function(func, context)
                example['metadata']['domain'] = 'carla'
                examples.append(example)

        return examples

    def process_embedded_data(self, files: List[str], domain: str, contexts: List[str]) -> List[Dict]:
        """Generic processor for embedded/automotive code files."""
        examples = []

        for file_path in tqdm(files, desc=f"Processing {domain}"):
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
        """Remove duplicate examples based on content hash"""
        seen = set()
        unique = []

        for example in examples:
            content = json.dumps(example['messages'], sort_keys=True)
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(example)

        print(f"[DataPipeline] Deduplicated: {len(examples)} -> {len(unique)}")
        return unique

    def create_splits(
        self,
        examples: List[Dict],
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """Create train/val splits"""
        random.seed(seed)
        random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        train = examples[:split_idx]
        val = examples[split_idx:]

        return train, val

    def save_jsonl(self, examples: List[Dict], output_path: Path):
        """Save examples to JSONL file"""
        with open(output_path, 'w') as f:
            for example in examples:
                # Only save messages for training
                output = {"messages": example["messages"]}
                f.write(json.dumps(output) + '\n')

        print(f"[DataPipeline] Saved {len(examples)} examples to {output_path}")

    def _download_and_filter(self, prefix: str, extensions: tuple, max_files: int) -> List[str]:
        """Download files from S3 prefix, filtered by extension."""
        max_keys = max_files if max_files > 0 else 100000
        objects = self.list_s3_objects(prefix, max_keys=max_keys)
        if max_files > 0:
            objects = objects[:max_files]

        # Filter by extension first, then report
        matching = [obj for obj in objects if obj['key'].endswith(extensions)]
        total_size_mb = sum(obj['size'] for obj in matching) / 1e6
        print(f"[DataPipeline] Found {len(matching)} files ({total_size_mb:.1f} MB) in {prefix}")

        files = []
        for obj in tqdm(matching, desc=f"Downloading {prefix}"):
            local = self.download_file(obj['key'])
            if local:
                files.append(local)
        return files

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

    def run_pipeline(
        self,
        download_data: bool = True,
        max_files_per_type: int = 100,
        train_ratio: float = 0.9
    ):
        """Run the full data pipeline across all automotive data sources."""
        print("\n" + "="*70)
        print("[DataPipeline] Starting Automotive Data Pipeline")
        print(f"[DataPipeline] Processing {len(self.DATA_SOURCES)} data sources")
        print("="*70)

        all_examples = []
        source_stats = {}

        if download_data:
            code_extensions = ('.c', '.h', '.cpp', '.cc', '.cxx', '.txt')

            for source in self.DATA_SOURCES:
                prefix = source['prefix']
                domain = source['domain']
                contexts = source['contexts']

                print(f"\n[DataPipeline] [{domain}] Downloading from {prefix}...")
                files = self._download_and_filter(prefix, code_extensions, max_files_per_type)

                if not files:
                    print(f"[DataPipeline] [{domain}] No files found, skipping")
                    source_stats[domain] = 0
                    continue

                examples = self.process_embedded_data(files, domain, contexts)
                all_examples.extend(examples)
                source_stats[domain] = len(examples)
                print(f"[DataPipeline] [{domain}] {len(examples)} examples from {len(files)} files")

        # Print summary by source
        print("\n" + "-"*70)
        print("[DataPipeline] Examples by source:")
        for domain, count in sorted(source_stats.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {domain:25s} {count:>6,}")
        print(f"  {'TOTAL (before dedup)':25s} {len(all_examples):>6,}")
        print("-"*70)

        # Deduplicate
        all_examples = self.deduplicate(all_examples)

        # Create splits
        train_examples, val_examples = self.create_splits(all_examples, train_ratio)

        # Save
        self.save_jsonl(train_examples, self.splits_dir / "train.jsonl")
        self.save_jsonl(val_examples, self.splits_dir / "val.jsonl")

        print("\n" + "="*70)
        print("[DataPipeline] Pipeline Complete")
        print("="*70)
        print(f"  Total examples: {len(all_examples)}")
        print(f"  Train examples: {len(train_examples)}")
        print(f"  Val examples: {len(val_examples)}")
        print(f"  Output: {self.splits_dir}")
        print("="*70)

        return train_examples, val_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare automotive data for Granite-8B fine-tuning"
    )
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default='granite-8b-unified-automotive-data',
        help='S3 bucket name'
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
        default=100,
        help='Max files per data type'
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
        help='Only process a small sample for testing'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip S3 download (use existing local files)'
    )

    args = parser.parse_args()

    max_files = 10 if args.sample_only else args.max_files

    pipeline = AutomotiveDataPipeline(
        s3_bucket=args.s3_bucket,
        region=args.region
    )

    pipeline.run_pipeline(
        download_data=not args.no_download,
        max_files_per_type=max_files,
        train_ratio=args.train_ratio
    )


if __name__ == "__main__":
    main()
