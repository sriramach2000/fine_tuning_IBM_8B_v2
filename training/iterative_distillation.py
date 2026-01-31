#!/usr/bin/env python3
"""
Iterative Teacher-Student Distillation Training Loop

Cloud-Native Architecture:
- Designed to run inside SageMaker Training/Processing Jobs
- All data accessed via S3 (no local downloads)
- Teacher model calls via Amazon Bedrock
- Checkpoints saved to S3

The student model generates outputs, which are evaluated by quality metrics.
Poor outputs trigger a call to the teacher model (Bedrock Claude) for correction.
Corrections are added to the training set for the next epoch.

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import json
import math
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DistillationConfig:
    """
    Configuration for iterative distillation.

    Cloud-Native Settings:
    - All paths default to SageMaker conventions
    - S3 URIs supported for data locations
    - Bedrock model IDs for teacher
    """
    # Quality thresholds
    quality_threshold: float = 7.0          # Min score to pass without correction
    convergence_threshold: float = 8.0      # Target average score for convergence
    convergence_patience: int = 3           # Epochs at threshold before stopping

    # Teacher model settings (Bedrock)
    teacher_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    teacher_max_tokens: int = 2048
    teacher_temperature: float = 0.7

    # Rate limiting
    max_corrections_per_epoch: int = 500    # Limit API costs
    max_parallel_teacher_calls: int = 10    # Parallel Bedrock requests

    # Evaluation
    eval_batch_size: int = 32
    eval_samples_per_epoch: int = 200

    # Cloud-native paths (SageMaker conventions)
    train_data_dir: str = field(default_factory=lambda: os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    eval_data_dir: str = field(default_factory=lambda: os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    output_dir: str = field(default_factory=lambda: os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    model_dir: str = field(default_factory=lambda: os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    corrections_dir: str = field(default_factory=lambda: os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output') + '/corrections')

    def __post_init__(self):
        """Create directories if they don't exist"""
        Path(self.corrections_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class EpochMetrics:
    """Metrics collected during a single epoch"""
    epoch: int
    train_loss: float
    avg_student_score: float
    num_eval_samples: int
    num_poor_outputs: int
    num_corrections: int
    correction_rate: float
    scores_distribution: Dict[str, int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'avg_student_score': self.avg_student_score,
            'num_eval_samples': self.num_eval_samples,
            'num_poor_outputs': self.num_poor_outputs,
            'num_corrections': self.num_corrections,
            'correction_rate': self.correction_rate,
            'scores_distribution': self.scores_distribution,
            'timestamp': self.timestamp,
        }


class IterativeDistillationTrainer:
    """
    Training loop with iterative teacher feedback.

    Cloud-Native Design:
    - All operations designed for SageMaker environment
    - No local file dependencies
    - S3-compatible checkpoint saving
    - Bedrock-based teacher model

    Each epoch:
    1. Train on current dataset (original + corrections)
    2. Evaluate student on validation prompts
    3. Identify poor outputs (score < threshold)
    4. Get teacher corrections for poor outputs
    5. Add corrections to training set
    6. Repeat until convergence
    """

    def __init__(
        self,
        student_model,
        student_tokenizer,
        teacher_generator,  # BedrockTeacherGenerator
        quality_evaluator,  # CodeQualityEvaluator
        config: DistillationConfig,
        trainer=None  # SFTTrainer or similar
    ):
        """
        Initialize the iterative distillation trainer.

        Args:
            student_model: The student model to train (Granite-8B)
            student_tokenizer: Tokenizer for the student model
            teacher_generator: BedrockTeacherGenerator for corrections
            quality_evaluator: CodeQualityEvaluator for scoring
            config: DistillationConfig with settings
            trainer: Optional pre-configured trainer (SFTTrainer)
        """
        self.student = student_model
        self.tokenizer = student_tokenizer
        self.teacher = teacher_generator
        self.evaluator = quality_evaluator
        self.config = config
        self.trainer = trainer

        # History tracking
        self.metrics_history: List[EpochMetrics] = []
        self.all_corrections: List[Dict] = []

        # Convergence tracking
        self.epochs_at_threshold = 0
        self.best_avg_score = 0.0

        print(f"[Distillation] Initialized iterative trainer")
        print(f"[Distillation] Quality threshold: {config.quality_threshold}")
        print(f"[Distillation] Convergence target: {config.convergence_threshold}")
        print(f"[Distillation] Max corrections/epoch: {config.max_corrections_per_epoch}")

    def train_epoch(
        self,
        train_dataset,
        eval_prompts: List[Dict],
        epoch_num: int
    ) -> EpochMetrics:
        """
        Execute a single training epoch with teacher feedback.

        Args:
            train_dataset: Current training dataset (original + corrections)
            eval_prompts: Prompts to evaluate student on
            epoch_num: Current epoch number (1-indexed)

        Returns:
            EpochMetrics with all collected metrics
        """
        print(f"\n{'='*60}")
        print(f"[Distillation] Epoch {epoch_num}")
        print(f"{'='*60}")

        # Step 1: Standard training on current dataset
        print(f"\n[Step 1/4] Training student model...")
        train_metrics = self._train_step(train_dataset)
        train_loss = train_metrics.get('loss', 0.0)
        print(f"  Train loss: {train_loss:.4f}")

        # Step 2: Generate student outputs on evaluation prompts
        print(f"\n[Step 2/4] Generating student outputs on {len(eval_prompts)} prompts...")
        student_outputs = self._generate_student_outputs(eval_prompts)

        # Step 3: Evaluate quality of each output
        print(f"\n[Step 3/4] Evaluating output quality...")
        scores, quality_scores = self._evaluate_outputs(student_outputs, eval_prompts)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"  Average score: {avg_score:.2f}/10")

        # Calculate score distribution
        distribution = {
            'below_5': sum(1 for s in scores if s < 5),
            '5_to_7': sum(1 for s in scores if 5 <= s < 7),
            '7_to_9': sum(1 for s in scores if 7 <= s < 9),
            'above_9': sum(1 for s in scores if s >= 9),
        }
        print(f"  Distribution: {distribution}")

        # Step 4: Identify outputs needing correction
        poor_outputs = self._identify_poor_outputs(eval_prompts, student_outputs, scores)
        print(f"\n[Step 4/4] Identified {len(poor_outputs)} outputs needing correction")

        # Step 5: Get teacher corrections for poor outputs
        corrections = []
        if poor_outputs:
            corrections = self._get_teacher_corrections(poor_outputs, epoch_num)
            print(f"  Received {len(corrections)} corrections from teacher")

            # Add corrections to training dataset for next epoch
            self._add_corrections_to_dataset(corrections, train_dataset)

        # Calculate metrics
        correction_rate = len(poor_outputs) / len(eval_prompts) if eval_prompts else 0.0

        metrics = EpochMetrics(
            epoch=epoch_num,
            train_loss=train_loss,
            avg_student_score=avg_score,
            num_eval_samples=len(eval_prompts),
            num_poor_outputs=len(poor_outputs),
            num_corrections=len(corrections),
            correction_rate=correction_rate,
            scores_distribution=distribution,
        )

        self.metrics_history.append(metrics)
        self._save_epoch_metrics(metrics)

        # Update convergence tracking
        if avg_score >= self.config.convergence_threshold:
            self.epochs_at_threshold += 1
        else:
            self.epochs_at_threshold = 0

        if avg_score > self.best_avg_score:
            self.best_avg_score = avg_score

        return metrics

    def _train_step(self, train_dataset) -> Dict[str, float]:
        """
        Execute standard training step.

        Returns dict with training metrics including 'loss'.
        """
        if self.trainer is not None:
            # Use pre-configured trainer
            train_result = self.trainer.train()
            return {'loss': train_result.training_loss}
        else:
            # Placeholder for custom training loop
            return {'loss': 0.0}

    def _generate_student_outputs(
        self,
        prompts: List[Dict],
        max_new_tokens: int = 1024
    ) -> List[str]:
        """
        Generate outputs from the student model.

        Args:
            prompts: List of prompt dicts with 'prompt' key
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated code strings
        """
        outputs = []
        self.student.eval()

        with torch.no_grad():
            for prompt_data in tqdm(prompts, desc="Student generating"):
                prompt_text = prompt_data.get('prompt', '')

                # Format as chat message
                messages = [{"role": "user", "content": prompt_text}]

                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted = f"<|user|>\n{prompt_text}\n<|assistant|>\n"

                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                ).to(self.student.device)

                output_ids = self.student.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Decode only the generated part
                generated = self.tokenizer.decode(
                    output_ids[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                outputs.append(generated)

        return outputs

    def _evaluate_outputs(
        self,
        outputs: List[str],
        prompts: List[Dict]
    ) -> Tuple[List[float], List[Any]]:
        """
        Evaluate quality of student outputs.

        Returns:
            Tuple of (overall_scores, quality_score_objects)
        """
        results = self.evaluator.evaluate_batch(outputs, prompts)
        scores = [r[0] for r in results]
        quality_scores = [r[1] for r in results]
        return scores, quality_scores

    def _identify_poor_outputs(
        self,
        prompts: List[Dict],
        outputs: List[str],
        scores: List[float]
    ) -> List[Tuple[Dict, str, float]]:
        """
        Identify outputs that need teacher correction.

        Returns list of (prompt, output, score) tuples for poor outputs.
        """
        poor = []
        for prompt, output, score in zip(prompts, outputs, scores):
            if score < self.config.quality_threshold:
                poor.append((prompt, output, score))

        # Sort by score (worst first) and limit
        poor.sort(key=lambda x: x[2])
        return poor[:self.config.max_corrections_per_epoch]

    def _get_teacher_corrections(
        self,
        poor_outputs: List[Tuple[Dict, str, float]],
        epoch_num: int
    ) -> List[Dict]:
        """
        Get teacher corrections for poor student outputs.

        Uses parallel requests to Bedrock for efficiency.

        Args:
            poor_outputs: List of (prompt, student_output, score) tuples
            epoch_num: Current epoch for tracking

        Returns:
            List of correction dicts ready for training
        """
        corrections = []

        def process_single(item: Tuple[Dict, str, float]) -> Optional[Dict]:
            prompt_data, student_output, score = item
            prompt_text = prompt_data.get('prompt', '')
            prompt_id = prompt_data.get('id', 'unknown')

            correction_prompt = self._create_correction_prompt(
                prompt_text, student_output, score
            )

            result = self.teacher.generate_response(
                prompt=correction_prompt,
                system_prompt=self._get_correction_system_prompt()
            )

            if result.get('success') and result.get('response'):
                return {
                    'id': f"{prompt_id}_correction_e{epoch_num}",
                    'original_prompt': prompt_text,
                    'student_output': student_output,
                    'student_score': score,
                    'teacher_correction': result['response'],
                    'epoch': epoch_num,
                    'messages': [
                        {'role': 'user', 'content': prompt_text},
                        {'role': 'assistant', 'content': result['response']}
                    ],
                    'usage': result.get('usage', {}),
                }
            return None

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_teacher_calls) as executor:
            futures = {executor.submit(process_single, item): item for item in poor_outputs}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Teacher correcting"):
                result = future.result()
                if result:
                    corrections.append(result)
                    self.all_corrections.append(result)

        # Save corrections for this epoch
        self._save_epoch_corrections(corrections, epoch_num)

        return corrections

    def _create_correction_prompt(
        self,
        original_prompt: str,
        student_output: str,
        score: float
    ) -> str:
        """Create prompt asking teacher to correct student's output"""
        return f"""A student model was asked to generate automotive embedded code for the following prompt:

<prompt>
{original_prompt}
</prompt>

The student produced this output (scored {score:.1f}/10):

<student_output>
{student_output}
</student_output>

Please provide a corrected, production-quality version that:
1. Fixes any errors or bugs in the student's code
2. Follows MISRA-C guidelines for safety-critical automotive code
3. Includes proper error handling and return codes
4. Uses correct data types (uint8_t, uint32_t, etc.)
5. Is suitable for embedded automotive systems (TSN/AVB)

Provide ONLY the corrected code with appropriate comments. No explanations outside the code."""

    def _get_correction_system_prompt(self) -> str:
        """System prompt for teacher when correcting student outputs"""
        return """You are a senior embedded systems engineer specializing in automotive software.
You are reviewing code from a junior developer learning automotive protocols (TSN/AVB).

Your task is to correct their code while:
- Maintaining the original intent and structure where possible
- Fixing bugs, safety issues, and protocol compliance problems
- Following MISRA-C guidelines for safety-critical code
- Ensuring TSN/AVB protocol compliance
- Adding proper error handling
- Using appropriate data types (uint8_t, uint32_t, etc.)

Output ONLY the corrected code with comments. No explanations outside the code."""

    def _add_corrections_to_dataset(
        self,
        corrections: List[Dict],
        train_dataset
    ):
        """
        Add teacher corrections to the training dataset.

        The corrections become part of the training data for the next epoch,
        allowing the student to learn from the teacher's improvements.
        """
        # For HuggingFace datasets
        if hasattr(train_dataset, 'add_item'):
            for correction in corrections:
                train_dataset.add_item(correction)
        else:
            # Fallback: Save to file for next epoch
            corrections_file = Path(self.config.corrections_dir) / 'accumulated_corrections.jsonl'
            with open(corrections_file, 'a') as f:
                for correction in corrections:
                    f.write(json.dumps(correction) + '\n')

    def _save_epoch_metrics(self, metrics: EpochMetrics):
        """Save epoch metrics to output directory"""
        metrics_file = Path(self.config.output_dir) / 'metrics_history.jsonl'
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')

    def _save_epoch_corrections(self, corrections: List[Dict], epoch_num: int):
        """Save corrections for a specific epoch"""
        corrections_file = Path(self.config.corrections_dir) / f'epoch_{epoch_num}_corrections.jsonl'
        with open(corrections_file, 'w') as f:
            for correction in corrections:
                f.write(json.dumps(correction) + '\n')

    def check_convergence(self) -> Tuple[bool, str]:
        """
        Check if training has converged.

        Convergence criteria:
        1. Average student score >= convergence_threshold for N consecutive epochs
        2. OR correction rate drops below 10%

        Returns:
            Tuple of (converged: bool, reason: str)
        """
        if len(self.metrics_history) < self.config.convergence_patience:
            return False, "Not enough epochs"

        # Check consistent high quality
        if self.epochs_at_threshold >= self.config.convergence_patience:
            return True, f"Student quality >= {self.config.convergence_threshold} for {self.epochs_at_threshold} epochs"

        # Check minimal corrections needed
        recent = self.metrics_history[-self.config.convergence_patience:]
        if all(m.correction_rate < 0.10 for m in recent):
            return True, "Correction rate < 10% for consecutive epochs"

        return False, "Training in progress"

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of the entire training run"""
        if not self.metrics_history:
            return {}

        return {
            'total_epochs': len(self.metrics_history),
            'final_avg_score': self.metrics_history[-1].avg_student_score,
            'best_avg_score': self.best_avg_score,
            'total_corrections': len(self.all_corrections),
            'initial_correction_rate': self.metrics_history[0].correction_rate if self.metrics_history else 0,
            'final_correction_rate': self.metrics_history[-1].correction_rate if self.metrics_history else 0,
            'converged': self.check_convergence()[0],
            'convergence_reason': self.check_convergence()[1],
        }


def create_distillation_trainer(
    student_model,
    student_tokenizer,
    bedrock_generator,
    quality_evaluator,
    config: Optional[DistillationConfig] = None,
    trainer=None
) -> IterativeDistillationTrainer:
    """
    Factory function to create a configured distillation trainer.

    Args:
        student_model: Granite-8B model
        student_tokenizer: Model tokenizer
        bedrock_generator: BedrockTeacherGenerator instance
        quality_evaluator: CodeQualityEvaluator instance
        config: Optional custom configuration
        trainer: Optional pre-configured SFTTrainer

    Returns:
        Configured IterativeDistillationTrainer
    """
    if config is None:
        config = DistillationConfig()

    return IterativeDistillationTrainer(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_generator=bedrock_generator,
        quality_evaluator=quality_evaluator,
        config=config,
        trainer=trainer
    )
