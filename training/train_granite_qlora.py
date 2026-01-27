#!/usr/bin/env python3
"""
QLoRA Fine-tuning script for IBM Granite-8B-Code-Instruct-128K

This script runs as a SageMaker Training Job to:
1. Load Granite-8B with 4-bit quantization (QLoRA)
2. Apply LoRA adapters
3. Train on TSN/AVB automotive code data
4. Save fine-tuned model and adapters

Target: Embedded Automotive Code Generation (AVB/TSN)

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import argparse
import torch
import shutil
import math
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer


class NaNInfDetectionCallback(TrainerCallback):
    """Callback to detect NaN/Inf in loss and halt training"""

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                print("\n" + "="*70)
                print("CRITICAL ERROR: NaN or Inf detected in loss!")
                print("="*70)
                print(f"Loss value: {loss}")
                print(f"Step: {state.global_step}")

                # Log CUDA memory state
                if torch.cuda.is_available():
                    print(f"\nCUDA Memory:")
                    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

                print("\nStopping training to prevent further corruption.")
                print("="*70 + "\n")

                control.should_training_stop = True

        return control


class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Custom early stopping that doesn't require load_best_model_at_end.
    Works around PEFT/Transformers version conflicts.
    """

    def __init__(self, patience: int = 3, threshold: float = 0.0, save_best: bool = True):
        self.patience = patience
        self.threshold = threshold
        self.save_best = save_best
        self.best_loss = float('inf')
        self.best_step = 0
        self.wait_count = 0
        self.best_model_checkpoint = None

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return control

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        if eval_loss < self.best_loss - self.threshold:
            self.best_loss = eval_loss
            self.best_step = state.global_step
            self.wait_count = 0
            print(f"\n[Granite-8B] New best eval_loss: {eval_loss:.4f} at step {state.global_step}")

            if self.save_best and args.output_dir:
                self.best_model_checkpoint = f"{args.output_dir}/checkpoint-{state.global_step}"
        else:
            self.wait_count += 1
            print(f"\n[Granite-8B] No improvement. Best: {self.best_loss:.4f} at step {self.best_step}. "
                  f"Patience: {self.wait_count}/{self.patience}")

            if self.wait_count >= self.patience:
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING: No improvement for {self.patience} evaluations")
                print(f"Best eval_loss: {self.best_loss:.4f} at step {self.best_step}")
                print(f"{'='*70}\n")
                control.should_training_stop = True

        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\n[Granite-8B] Training Summary:")
        print(f"   Best eval_loss: {self.best_loss:.4f}")
        print(f"   Best step: {self.best_step}")
        if self.best_model_checkpoint:
            print(f"   Best checkpoint: {self.best_model_checkpoint}")
        return control


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Granite-8B QLoRA Fine-tuning for Automotive Code")

    # Model arguments - UPDATED FOR GRANITE-8B
    parser.add_argument('--model-name', type=str,
                        default='ibm-granite/granite-8b-code-instruct-128k',
                        help='HuggingFace model name')
    parser.add_argument('--max-seq-length', type=int, default=4096,
                        help='Maximum sequence length for training')

    # QLoRA arguments - OPTIMIZED FOR GRANITE
    parser.add_argument('--load-in-4bit', type=bool, default=True)
    parser.add_argument('--bnb-4bit-compute-dtype', type=str, default='bfloat16',
                        help='Compute dtype (bfloat16 recommended for Granite)')
    parser.add_argument('--bnb-4bit-quant-type', type=str, default='nf4')
    parser.add_argument('--bnb-4bit-use-double-quant', type=bool, default=True)

    # LoRA arguments - INCREASED FOR 8B MODEL
    parser.add_argument('--lora-r', type=int, default=32,
                        help='LoRA rank (increased for 8B model)')
    parser.add_argument('--lora-alpha', type=int, default=64,
                        help='LoRA alpha (2x rank)')
    parser.add_argument('--lora-dropout', type=float, default=0.05)

    # Training arguments - ADJUSTED FOR 8B MODEL
    parser.add_argument('--num-train-epochs', type=int, default=5)
    parser.add_argument('--per-device-train-batch-size', type=int, default=2,
                        help='Batch size (reduced for 8B model)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Gradient accumulation (increased for effective batch size 16)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (recommended for Granite fine-tuning)')
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.3)
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit')

    # SageMaker paths
    parser.add_argument('--train-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--val-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_VAL', '/opt/ml/input/data/val'))
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    # Logging
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--save-steps', type=int, default=100)

    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=3)
    parser.add_argument('--early-stopping-threshold', type=float, default=0.0)

    return parser.parse_args()


def format_chat_template(example, tokenizer):
    """
    Format messages using Granite's chat template.

    Granite uses a Llama-style chat format:
    <|user|>
    {user_message}
    <|assistant|>
    {assistant_message}
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback format for Granite (Llama-style)
        messages = example["messages"]
        text = ""
        for msg in messages:
            if msg["role"] == "user":
                text += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                text += f"<|assistant|>\n{msg['content']}\n"
            elif msg["role"] == "system":
                text += f"<|system|>\n{msg['content']}\n"
        return text


def print_trainable_parameters(model):
    """Print number of trainable parameters"""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"[Granite-8B] Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable %: {100 * trainable_params / all_param:.2f}%")


def validate_datasets(train_dataset, val_dataset):
    """Validate dataset structure and content"""
    print("\n[Granite-8B] Validating datasets...")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty!")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty!")

    if len(val_dataset) < 3:
        print(f"  WARNING: Validation set is small ({len(val_dataset)} examples)")

    # Validate structure
    first_train = train_dataset[0]

    if "messages" not in first_train:
        raise ValueError("Training dataset missing 'messages' field!")

    if not isinstance(first_train["messages"], list) or len(first_train["messages"]) == 0:
        raise ValueError("'messages' field must be a non-empty list!")

    first_msg = first_train["messages"][0]
    if "role" not in first_msg or "content" not in first_msg:
        raise ValueError("Messages must have 'role' and 'content' fields!")

    print(f"  [OK] Train examples: {len(train_dataset)}")
    print(f"  [OK] Val examples: {len(val_dataset)}")
    print(f"  [OK] Message format: Valid")


def verify_quantization(model):
    """Verify model is properly quantized"""
    print("\n[Granite-8B] Verifying quantization...")
    print(f"  Model dtype: {model.dtype}")

    quantized_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'quant_state'):
            quantized_layers.append(name)

    if len(quantized_layers) > 0:
        print(f"  [OK] Found {len(quantized_layers)} quantized layers")
        return True
    else:
        print("  WARNING: No quantized layers found - may use more VRAM")
        return False


def log_cuda_memory():
    """Log CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n[Granite-8B] CUDA Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    else:
        print("\n[Granite-8B] No CUDA device available")


def check_disk_space(path, required_gb=15):
    """Check available disk space (increased for 8B model)"""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        print(f"\n[Granite-8B] Disk Space ({path}): {free_gb:.2f} GB free")

        if free_gb < required_gb:
            print(f"  WARNING: Low disk space! Need at least {required_gb} GB")
            return False
        return True
    except Exception as e:
        print(f"  Could not check disk space: {str(e)}")
        return True


def main():
    args = parse_args()

    print("="*70)
    print("IBM Granite-8B QLoRA Fine-tuning")
    print("Target: Embedded Automotive Code Generation (AVB/TSN)")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Training epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Compute dtype: {args.bnb_4bit_compute_dtype}")
    print("="*70)

    # Load datasets
    print("\n[Granite-8B] Loading datasets...")
    try:
        train_jsonl = os.path.join(args.train_dir, 'train.jsonl')
        val_jsonl = os.path.join(args.val_dir, 'val.jsonl')

        train_dataset = load_dataset('json', data_files=train_jsonl, split='train')
        val_dataset = load_dataset('json', data_files=val_jsonl, split='train')

        print(f"  [OK] Train: {len(train_dataset)} examples")
        print(f"  [OK] Val: {len(val_dataset)} examples")

    except Exception as e:
        print(f"\n  ERROR loading datasets: {str(e)}")
        sys.exit(1)

    validate_datasets(train_dataset, val_dataset)

    # Configure 4-bit quantization
    print("\n[Granite-8B] Configuring 4-bit quantization...")

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    print(f"  Quantization: {args.bnb_4bit_quant_type}")
    print(f"  Compute dtype: {args.bnb_4bit_compute_dtype}")
    print(f"  Double quantization: {args.bnb_4bit_use_double_quant}")

    # Load model
    print("\n[Granite-8B] Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,  # Granite doesn't need this
            torch_dtype=compute_dtype,
            attn_implementation="eager"  # Default attention
        )

        print("  [OK] Model loaded with 4-bit quantization")
        verify_quantization(model)
        log_cuda_memory()

    except Exception as e:
        print(f"\n  ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  [OK] Tokenizer loaded")

    # Prepare for k-bit training
    print("\n[Granite-8B] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    print("  [OK] Model prepared")

    # Configure LoRA
    print("\n[Granite-8B] Configuring LoRA adapters...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("  [OK] LoRA adapters configured")
    print_trainable_parameters(model)

    # Training arguments
    print("\n[Granite-8B] Configuring training...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,

        # Mixed precision - BF16 for Granite
        fp16=False,
        bf16=True if args.bnb_4bit_compute_dtype == 'bfloat16' else False,

        # Memory optimization
        gradient_checkpointing=True,

        # Logging
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        logging_strategy="steps",

        # Evaluation
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,

        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,

        # Other
        report_to=[],
        disable_tqdm=False,
    )

    print("  [OK] Training arguments configured")

    # Initialize trainer
    print("\n[Granite-8B] Initializing SFTTrainer...")

    def formatting_func(example):
        return format_chat_template(example, tokenizer)

    early_stopping = CustomEarlyStoppingCallback(
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
        save_best=True
    )

    nan_detection = NaNInfDetectionCallback()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        callbacks=[nan_detection, early_stopping],
    )

    print(f"  [OK] SFTTrainer initialized")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  NaN/Inf detection: enabled")

    # Train
    print("\n" + "="*70)
    print("[Granite-8B] Starting training...")
    print("="*70 + "\n")

    try:
        trainer.train()

        print("\n" + "="*70)
        print("[Granite-8B] Training complete!")
        print("="*70 + "\n")

        log_cuda_memory()

    except Exception as e:
        print("\n" + "="*70)
        print("[Granite-8B] Training failed!")
        print("="*70)
        print(f"\n  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save model
    print("[Granite-8B] Saving fine-tuned model...")

    check_disk_space(args.model_dir, required_gb=15)

    try:
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)

        print(f"  [OK] Model saved to: {args.model_dir}")

        model_files = list(Path(args.model_dir).glob("*"))
        print(f"  [OK] Saved {len(model_files)} files")

    except Exception as e:
        print(f"\n  ERROR saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save metrics
    metrics = trainer.state.log_history
    metrics_file = os.path.join(args.output_dir, 'training_metrics.json')

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  [OK] Metrics saved to: {metrics_file}")

    print("\n" + "="*70)
    print("[Granite-8B] Fine-tuning Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
