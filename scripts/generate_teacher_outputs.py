#!/usr/bin/env python3
"""
Generate Teacher Outputs using Amazon Bedrock Claude

This script generates high-quality code examples using Claude via Amazon Bedrock
for knowledge distillation into Granite-8B for automotive code generation.

Features:
- Batch processing of prompts with rate limiting
- Automatic retries with exponential backoff
- Progress tracking and checkpointing
- Outputs saved in JSONL format for training

Target: Embedded Automotive Code Generation (AVB/TSN)

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BedrockTeacherGenerator:
    """
    Generate teacher outputs using Amazon Bedrock Claude.

    Uses Claude Sonnet for high-quality automotive code generation
    with automatic rate limiting and retry logic.

    Authentication:
    - Primary: AWS IAM credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - Secondary: Bedrock API Key (AMAZON_BEDROCK_MODEL_API_KEY) for additional auth
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region: str = "us-east-1",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_retries: int = 5,
        base_delay: float = 1.0,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Get Bedrock API key from parameter or environment
        self.api_key = api_key or os.environ.get('AMAZON_BEDROCK_MODEL_API_KEY')

        # Configure boto3 with retry settings
        config = Config(
            region_name=region,
            retries={
                'max_attempts': max_retries,
                'mode': 'adaptive'
            }
        )

        self.client = boto3.client('bedrock-runtime', config=config)

        # Register event handler to add API key header if available
        if self.api_key:
            self.client.meta.events.register(
                'before-send.bedrock-runtime.*',
                self._add_api_key_header
            )
            print(f"[Bedrock] API Key authentication: ENABLED")

        print(f"[Bedrock] Initialized with model: {model_id}")
        print(f"[Bedrock] Region: {region}")
        print(f"[Bedrock] Max tokens: {max_tokens}")
        print(f"[Bedrock] IAM credentials: {os.environ.get('AWS_ACCESS_KEY_ID', 'N/A')[:10]}...")

    def _add_api_key_header(self, request, **kwargs):
        """Add Bedrock API key to request headers for additional authentication"""
        if self.api_key:
            request.headers['x-amz-bedrock-api-key'] = self.api_key

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate a single response from Claude via Bedrock.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context

        Returns:
            Dict with 'success', 'response', and 'error' fields
        """
        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }

        if system_prompt:
            body["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )

                result = json.loads(response['body'].read())
                content = result.get('content', [])

                if content and len(content) > 0:
                    return {
                        'success': True,
                        'response': content[0].get('text', ''),
                        'usage': result.get('usage', {}),
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'response': None,
                        'error': 'Empty response from Bedrock'
                    }

            except ClientError as e:
                error_code = e.response['Error']['Code']

                if error_code == 'ThrottlingException':
                    # Rate limited - exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    print(f"[Bedrock] Rate limited, waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                elif error_code == 'ModelStreamErrorException':
                    # Transient error - retry
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

                else:
                    return {
                        'success': False,
                        'response': None,
                        'error': f"Bedrock error: {error_code} - {str(e)}"
                    }

            except Exception as e:
                return {
                    'success': False,
                    'response': None,
                    'error': f"Unexpected error: {str(e)}"
                }

        return {
            'success': False,
            'response': None,
            'error': f"Max retries ({self.max_retries}) exceeded"
        }

    def generate_batch(
        self,
        prompts: List[Dict],
        system_prompt: Optional[str] = None,
        output_file: Optional[str] = None,
        checkpoint_interval: int = 10,
        max_workers: int = 10
    ) -> List[Dict]:
        """
        Generate responses for a batch of prompts with parallel processing.

        Args:
            prompts: List of dicts with 'id' and 'prompt' fields
            system_prompt: Optional system prompt
            output_file: Optional file to save results incrementally
            checkpoint_interval: Save checkpoint every N prompts
            max_workers: Number of parallel workers (default: 10, max recommended: 50)

        Returns:
            List of results with prompt, response, and metadata
        """
        results = []
        failed_count = 0
        results_lock = __import__('threading').Lock()

        print(f"\n[Bedrock] Processing {len(prompts)} prompts with {max_workers} parallel workers...")
        print(f"[Bedrock] Estimated time: ~{len(prompts) / max_workers / 60 * 3:.1f} minutes")

        def process_prompt(prompt_data, idx):
            """Process a single prompt"""
            prompt_id = prompt_data.get('id', idx)
            prompt_text = prompt_data.get('prompt', '')

            if not prompt_text:
                return None

            result = self.generate_response(prompt_text, system_prompt)

            return {
                'id': prompt_id,
                'prompt': prompt_text,
                'response': result.get('response'),
                'success': result.get('success'),
                'error': result.get('error'),
                'usage': result.get('usage', {}),
                'timestamp': datetime.now().isoformat()
            }

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_prompt, prompt_data, i): i
                for i, prompt_data in enumerate(prompts)
            }

            completed = 0
            for future in tqdm(as_completed(futures), total=len(prompts), desc="Generating"):
                output = future.result()
                if output:
                    with results_lock:
                        results.append(output)
                        if not output['success']:
                            failed_count += 1
                        completed += 1

                        # Checkpoint save
                        if output_file and completed % checkpoint_interval == 0:
                            self._save_checkpoint(results, output_file)

        # Sort results by ID to maintain order
        results.sort(key=lambda x: str(x['id']))

        # Final save
        if output_file:
            self._save_checkpoint(results, output_file)

        print(f"\n[Bedrock] Completed: {len(results) - failed_count}/{len(results)} successful")

        return results

    def _save_checkpoint(self, results: List[Dict], output_file: str):
        """Save results to JSONL file"""
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')


def create_automotive_system_prompt() -> str:
    """Create system prompt for automotive code generation"""
    return """You are an expert embedded systems engineer specializing in automotive software development.
Your expertise includes:
- Time-Sensitive Networking (TSN) protocols (IEEE 802.1Qbv, 802.1Qav, 802.1AS)
- Audio Video Bridging (AVB) for automotive infotainment
- Real-time embedded C/C++ programming
- Safety-critical code following MISRA-C guidelines
- Automotive Ethernet protocols

When generating code:
1. Follow MISRA-C/C++ guidelines for safety-critical code
2. Include proper error handling and return codes
3. Add clear, concise comments explaining the logic
4. Use appropriate data types (uint8_t, uint32_t, etc.)
5. Consider real-time constraints and deterministic execution
6. Include necessary header guards and includes

Generate production-quality code that could be used in real automotive systems."""


def create_sample_prompts() -> List[Dict]:
    """Create sample automotive code prompts for testing"""
    prompts = [
        {
            "id": "tsn_tas_001",
            "prompt": "Generate C code for a TSN Time-Aware Shaper (802.1Qbv) implementation with 8 priority queues for automotive Ethernet. Include the gate control list scheduling logic."
        },
        {
            "id": "avb_srp_001",
            "prompt": "Generate C code for AVB Stream Reservation Protocol (SRP) talker advertisement for an automotive audio stream with 48kHz sample rate, 24-bit depth, and 8 channels."
        },
        {
            "id": "tsn_ptp_001",
            "prompt": "Generate C code for IEEE 802.1AS Precision Time Protocol (PTP) timestamp processing for automotive time synchronization. Include nanosecond-precision timestamp handling."
        },
        {
            "id": "eth_frame_001",
            "prompt": "Generate C code for parsing and constructing Ethernet frames with VLAN tags (802.1Q) for automotive networking. Include CRC calculation."
        },
        {
            "id": "sensor_fusion_001",
            "prompt": "Generate C++ code for a sensor data fusion module that combines LIDAR, camera, and radar data for autonomous driving perception using Kalman filtering."
        }
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher outputs using Amazon Bedrock Claude"
    )

    parser.add_argument(
        '--input-file',
        type=str,
        help='Input JSONL file with prompts (each line: {"id": ..., "prompt": ...})'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='./data/teacher_outputs/bedrock_outputs.jsonl',
        help='Output JSONL file for teacher responses'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        help='Bedrock inference profile ID (default: Claude Sonnet 4.5)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default=os.environ.get('AWS_REGION', 'us-east-1'),
        help='AWS region'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens per response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run with sample prompts for testing'
    )
    parser.add_argument(
        '--prompts',
        type=int,
        default=5,
        help='Number of test prompts (with --test-mode)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10, max recommended: 50)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = BedrockTeacherGenerator(
        model_id=args.model_id,
        region=args.region,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Get prompts
    if args.test_mode:
        print("\n[Bedrock] Running in TEST MODE with sample prompts...")
        prompts = create_sample_prompts()[:args.prompts]
    elif args.input_file:
        print(f"\n[Bedrock] Loading prompts from: {args.input_file}")
        prompts = []
        with open(args.input_file, 'r') as f:
            for line in f:
                prompts.append(json.loads(line))
        print(f"[Bedrock] Loaded {len(prompts)} prompts")
    else:
        print("ERROR: Provide --input-file or use --test-mode")
        sys.exit(1)

    # Generate outputs
    system_prompt = create_automotive_system_prompt()

    results = generator.generate_batch(
        prompts=prompts,
        system_prompt=system_prompt,
        output_file=args.output_file,
        checkpoint_interval=10,
        max_workers=args.workers
    )

    # Print summary
    successful = sum(1 for r in results if r['success'])
    total_tokens = sum(r.get('usage', {}).get('output_tokens', 0) for r in results)

    print("\n" + "="*70)
    print("[Bedrock] Generation Complete")
    print("="*70)
    print(f"  Total prompts: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    print(f"  Total output tokens: {total_tokens:,}")
    print(f"  Output file: {args.output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
