#!/usr/bin/env python3
"""
Run Iterative Teacher-Student Distillation Pipeline

Cloud-Native Design:
- Runs inside SageMaker Training Job container
- All data accessed via S3 channels
- Teacher model calls via Amazon Bedrock
- Checkpoints saved to S3 via SM_MODEL_DIR

Usage:
    # Local testing
    python scripts/run_iterative_pipeline.py --local --max-epochs 2

    # Launch as SageMaker Training Job
    python scripts/launch_cloud_training.py

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables must be set externally


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run iterative teacher-student distillation pipeline"
    )

    # Mode selection
    parser.add_argument('--local', action='store_true',
                        help='Run locally instead of in SageMaker')

    # Training settings
    parser.add_argument('--max-epochs', type=int, default=5,
                        help='Maximum training epochs')
    parser.add_argument('--quality-threshold', type=float, default=7.0,
                        help='Minimum quality score (outputs below this get corrected)')
    parser.add_argument('--convergence-threshold', type=float, default=8.0,
                        help='Target average score for convergence')
    parser.add_argument('--max-corrections-per-epoch', type=int, default=500,
                        help='Limit teacher API calls per epoch')
    parser.add_argument('--eval-samples', type=int, default=200,
                        help='Number of samples to evaluate each epoch')

    # Model settings
    parser.add_argument('--model-name', type=str,
                        default='ibm-granite/granite-8b-code-instruct-128k',
                        help='HuggingFace model name')
    parser.add_argument('--teacher-model', type=str,
                        default='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        help='Bedrock model ID for teacher')

    # SageMaker paths (auto-populated in cloud)
    parser.add_argument('--train-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--eval-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_EVAL', './data/eval'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './models'))

    # Resumption
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint directory')

    # Parallel workers
    parser.add_argument('--teacher-workers', type=int, default=10,
                        help='Parallel teacher API calls')

    return parser.parse_args()


def setup_directories(args):
    """Create output directories"""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/corrections").mkdir(parents=True, exist_ok=True)


def load_datasets(args):
    """
    Load training and evaluation datasets.

    In SageMaker, data is mounted at SM_CHANNEL_* paths.
    Locally, reads from specified directories.
    """
    from datasets import load_dataset

    # Load training data
    train_file = Path(args.train_dir) / 'train.jsonl'
    if train_file.exists():
        train_dataset = load_dataset('json', data_files=str(train_file), split='train')
        print(f"[Data] Loaded {len(train_dataset)} training examples")
    else:
        # Try accumulated corrections from previous run
        corrections_file = Path(args.output_dir) / 'corrections/accumulated_corrections.jsonl'
        if corrections_file.exists():
            train_dataset = load_dataset('json', data_files=str(corrections_file), split='train')
            print(f"[Data] Loaded {len(train_dataset)} examples from corrections")
        else:
            raise FileNotFoundError(f"No training data found at {train_file}")

    # Load evaluation prompts
    eval_file = Path(args.eval_dir) / 'eval_prompts.jsonl'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_prompts = [json.loads(line) for line in f]
        print(f"[Data] Loaded {len(eval_prompts)} evaluation prompts")
    else:
        # Use sample prompts for testing
        print("[Data] No eval prompts found, using sample prompts")
        eval_prompts = create_sample_eval_prompts()

    return train_dataset, eval_prompts


def create_sample_eval_prompts():
    """Create sample evaluation prompts for testing"""
    return [
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
            "prompt": "Generate C code for parsing and constructing Ethernet frames with VLAN tags (802.1Q) for automotive networking. Include proper byte ordering."
        },
        {
            "id": "tsn_cbs_001",
            "prompt": "Generate C code for Credit-Based Shaper (802.1Qav) implementation for AVB traffic class management in automotive Ethernet."
        },
    ]


def load_model_and_tokenizer(args):
    """
    Load the student model with QLoRA configuration.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

    print(f"\n[Model] Loading {args.model_name}...")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] Device: {device}")

    # Quantization config
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        model = prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("[Model] QLoRA adapters configured")
    else:
        # CPU fallback for testing
        print("[Model] WARNING: Running on CPU - limited functionality")
        model = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def initialize_components(args, model, tokenizer):
    """Initialize teacher generator and quality evaluator"""
    from scripts.generate_teacher_outputs import BedrockTeacherGenerator
    from evaluation.code_quality_metrics import CodeQualityEvaluator
    from training.iterative_distillation import DistillationConfig

    # Teacher generator (Bedrock)
    teacher = BedrockTeacherGenerator(
        model_id=args.teacher_model,
        region=os.environ.get('AWS_REGION', 'us-east-1'),
        max_tokens=2048,
        temperature=0.7
    )

    # Quality evaluator
    evaluator = CodeQualityEvaluator(
        strict_mode=True,
        quality_threshold=args.quality_threshold
    )

    # Distillation config
    config = DistillationConfig(
        quality_threshold=args.quality_threshold,
        convergence_threshold=args.convergence_threshold,
        max_corrections_per_epoch=args.max_corrections_per_epoch,
        max_parallel_teacher_calls=args.teacher_workers,
        eval_samples_per_epoch=args.eval_samples,
        teacher_model=args.teacher_model,
        train_data_dir=args.train_dir,
        eval_data_dir=args.eval_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )

    return teacher, evaluator, config


def run_training_loop(args, trainer, train_dataset, eval_prompts):
    """
    Execute the iterative training loop.

    Each epoch:
    1. Train student
    2. Evaluate outputs
    3. Get teacher corrections for poor outputs
    4. Add corrections to training data
    5. Check convergence
    """
    print("\n" + "="*70)
    print("ITERATIVE TEACHER-STUDENT DISTILLATION")
    print("="*70)
    print(f"Max epochs: {args.max_epochs}")
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Convergence target: {args.convergence_threshold}")
    print(f"Eval samples/epoch: {args.eval_samples}")
    print(f"Max corrections/epoch: {args.max_corrections_per_epoch}")
    print("="*70)

    for epoch in range(1, args.max_epochs + 1):
        # Limit eval prompts
        epoch_eval = eval_prompts[:args.eval_samples]

        # Run epoch
        metrics = trainer.train_epoch(
            train_dataset=train_dataset,
            eval_prompts=epoch_eval,
            epoch_num=epoch
        )

        # Print epoch summary
        print(f"\n[Epoch {epoch}] Summary:")
        print(f"  Train loss: {metrics.train_loss:.4f}")
        print(f"  Avg student score: {metrics.avg_student_score:.2f}/10")
        print(f"  Poor outputs: {metrics.num_poor_outputs}/{len(epoch_eval)}")
        print(f"  Corrections: {metrics.num_corrections}")
        print(f"  Correction rate: {metrics.correction_rate:.1%}")

        # Check convergence
        converged, reason = trainer.check_convergence()
        if converged:
            print(f"\n[CONVERGED] {reason}")
            break

    return trainer.get_training_summary()


def save_final_model(model, tokenizer, args):
    """Save the final trained model"""
    if model is not None:
        print(f"\n[Save] Saving model to {args.model_dir}...")
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        print("[Save] Model saved successfully")


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("Iterative Teacher-Student Distillation Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    # Detect environment
    in_sagemaker = os.environ.get('SM_TRAINING_ENV') is not None
    print(f"Environment: {'SageMaker' if in_sagemaker else 'Local'}")

    # Setup
    setup_directories(args)

    # Load data
    train_dataset, eval_prompts = load_datasets(args)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args)

    # Initialize components
    teacher, evaluator, config = initialize_components(args, model, tokenizer)

    # Create trainer
    from training.iterative_distillation import IterativeDistillationTrainer

    trainer = IterativeDistillationTrainer(
        student_model=model,
        student_tokenizer=tokenizer,
        teacher_generator=teacher,
        quality_evaluator=evaluator,
        config=config,
    )

    # Run training
    summary = run_training_loop(args, trainer, train_dataset, eval_prompts)

    # Save model
    save_final_model(model, tokenizer, args)

    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total epochs: {summary.get('total_epochs', 0)}")
    print(f"Final avg score: {summary.get('final_avg_score', 0):.2f}/10")
    print(f"Best avg score: {summary.get('best_avg_score', 0):.2f}/10")
    print(f"Total corrections: {summary.get('total_corrections', 0)}")
    print(f"Initial correction rate: {summary.get('initial_correction_rate', 0):.1%}")
    print(f"Final correction rate: {summary.get('final_correction_rate', 0):.1%}")
    print(f"Converged: {summary.get('converged', False)}")
    if summary.get('convergence_reason'):
        print(f"Reason: {summary['convergence_reason']}")
    print("="*70)

    # Save summary
    summary_file = Path(args.output_dir) / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
