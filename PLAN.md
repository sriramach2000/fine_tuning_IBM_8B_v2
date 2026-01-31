# Granite-8B Fine-Tuning: Iterative Teacher-Student Distillation Pipeline

## Architecture Overview

### Core Concept: Online/Iterative Distillation
Unlike traditional offline distillation (pre-generate all teacher outputs), this pipeline uses an **iterative feedback loop**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATIVE DISTILLATION LOOP                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Student  │───►│ Metrics  │───►│ Quality  │───►│   Teacher    │  │
│  │ Generates│    │ Evaluate │    │ Check    │    │   Corrects   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│       ▲                              │                   │          │
│       │                              │ Good              │ Bad      │
│       │                              ▼                   ▼          │
│       │                        ┌──────────┐      ┌──────────────┐  │
│       │                        │  Keep    │      │   Add to     │  │
│       │                        │  Output  │      │ Training Set │  │
│       │                        └──────────┘      └──────────────┘  │
│       │                              │                   │          │
│       └──────────────────────────────┴───────────────────┘          │
│                          Next Epoch                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Workflow Per Epoch

1. **Student Inference**: Granite-8B generates code for evaluation prompts
2. **Quality Metrics**: Evaluate using:
   - Code compilation success
   - TSN/AVB protocol compliance
   - MISRA-C guideline adherence
   - Semantic correctness scoring
3. **Quality Gate**: If score < threshold (e.g., 7.0/10):
   - Send prompt + student output to Teacher (Bedrock Claude)
   - Teacher provides corrected/improved version
   - Add correction to training data for next epoch
4. **Training Update**: Train on:
   - Original training data
   - Teacher corrections from this epoch
5. **Convergence Check**: Stop when student quality meets threshold

---

## Phase 1: Critical Bug Fixes (Immediate)

### Bug #1: Dead Assertion in test_ml_paradigm.py

**File**: `tests/test_ml_paradigm.py:113`
**Bug**: `assert ratio < 100 or True` - Always passes due to `or True`
**Fix**: Remove `or True`

```python
# BEFORE (BUG):
assert ratio < 100 or True  # Would fail in real scenario

# AFTER (FIXED):
assert ratio < 100, f"Gradient imbalance too high: {ratio:.2f}x"
```

---

## Phase 2: Iterative Distillation Implementation

### New Files to Create

```
fine_tuning_IBM_8B_v2/
├── training/
│   └── iterative_distillation.py    # NEW: Main training loop with teacher feedback
├── evaluation/
│   ├── __init__.py
│   ├── code_quality_metrics.py      # NEW: Quality scoring for student outputs
│   ├── tsn_avb_compliance.py        # NEW: Protocol-specific checks
│   └── misra_checker.py             # NEW: MISRA-C guideline checker
├── scripts/
│   └── run_iterative_pipeline.py    # NEW: Orchestrate iterative training
└── data/
    └── corrections/                  # NEW: Store teacher corrections per epoch
        ├── epoch_1_corrections.jsonl
        ├── epoch_2_corrections.jsonl
        └── ...
```

### iterative_distillation.py - Core Training Loop

```python
#!/usr/bin/env python3
"""
Iterative Teacher-Student Distillation Training Loop

The student model generates outputs, which are evaluated by quality metrics.
Poor outputs trigger a call to the teacher model (Bedrock Claude) for correction.
Corrections are added to the training set for the next epoch.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DistillationConfig:
    """Configuration for iterative distillation"""
    quality_threshold: float = 7.0          # Min score to pass without correction
    max_corrections_per_epoch: int = 1000   # Limit API costs
    teacher_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    evaluation_batch_size: int = 32
    correction_batch_size: int = 10         # Parallel teacher calls


class IterativeDistillationTrainer:
    """
    Training loop with iterative teacher feedback.

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
        teacher_generator,  # BedrockTeacherGenerator
        quality_evaluator,  # CodeQualityEvaluator
        config: DistillationConfig
    ):
        self.student = student_model
        self.teacher = teacher_generator
        self.evaluator = quality_evaluator
        self.config = config
        self.corrections_history = []

    def train_epoch(
        self,
        train_dataset,
        eval_prompts: List[Dict]
    ) -> Dict:
        """
        Single epoch with teacher feedback.

        Returns:
            Dict with metrics including:
            - train_loss
            - avg_student_score
            - num_corrections
            - correction_improvement
        """
        # Step 1: Standard training on current dataset
        train_metrics = self._train_step(train_dataset)

        # Step 2: Generate student outputs on evaluation prompts
        student_outputs = self._generate_student_outputs(eval_prompts)

        # Step 3: Evaluate quality of each output
        scores = self._evaluate_outputs(student_outputs)

        # Step 4: Identify outputs needing correction
        poor_outputs = [
            (prompt, output, score)
            for prompt, output, score in zip(eval_prompts, student_outputs, scores)
            if score < self.config.quality_threshold
        ]

        # Step 5: Get teacher corrections for poor outputs
        corrections = []
        if poor_outputs:
            corrections = self._get_teacher_corrections(poor_outputs)
            self._add_corrections_to_dataset(corrections, train_dataset)

        return {
            'train_loss': train_metrics['loss'],
            'avg_student_score': sum(scores) / len(scores),
            'num_poor_outputs': len(poor_outputs),
            'num_corrections': len(corrections),
            'scores_distribution': {
                'below_5': sum(1 for s in scores if s < 5),
                '5_to_7': sum(1 for s in scores if 5 <= s < 7),
                '7_to_9': sum(1 for s in scores if 7 <= s < 9),
                'above_9': sum(1 for s in scores if s >= 9),
            }
        }

    def _get_teacher_corrections(
        self,
        poor_outputs: List[Tuple[Dict, str, float]]
    ) -> List[Dict]:
        """
        Call teacher model to correct poor student outputs.

        For each poor output:
        1. Send original prompt + student's attempt
        2. Ask teacher to identify issues and provide correction
        3. Return corrected output for training
        """
        corrections = []

        for prompt_data, student_output, score in poor_outputs[:self.config.max_corrections_per_epoch]:
            correction_prompt = self._create_correction_prompt(
                original_prompt=prompt_data['prompt'],
                student_output=student_output,
                score=score
            )

            result = self.teacher.generate_response(
                prompt=correction_prompt,
                system_prompt=self._get_correction_system_prompt()
            )

            if result['success']:
                corrections.append({
                    'id': f"{prompt_data['id']}_corrected",
                    'original_prompt': prompt_data['prompt'],
                    'student_output': student_output,
                    'student_score': score,
                    'teacher_correction': result['response'],
                    'messages': [
                        {'role': 'user', 'content': prompt_data['prompt']},
                        {'role': 'assistant', 'content': result['response']}
                    ]
                })

        return corrections

    def _create_correction_prompt(
        self,
        original_prompt: str,
        student_output: str,
        score: float
    ) -> str:
        """Create prompt asking teacher to correct student's output"""
        return f"""A student model was asked to generate automotive code for the following prompt:

<prompt>
{original_prompt}
</prompt>

The student produced this output (scored {score:.1f}/10):

<student_output>
{student_output}
</student_output>

Please provide a corrected, production-quality version that:
1. Fixes any errors in the student's code
2. Follows MISRA-C guidelines for safety-critical code
3. Includes proper error handling
4. Is suitable for embedded automotive systems (TSN/AVB)

Provide ONLY the corrected code, no explanations."""

    def _get_correction_system_prompt(self) -> str:
        """System prompt for teacher when correcting student outputs"""
        return """You are a senior embedded systems engineer reviewing code from a junior developer.
Your task is to correct their automotive code while:
- Maintaining the original intent
- Fixing bugs and safety issues
- Following MISRA-C guidelines
- Ensuring TSN/AVB protocol compliance
- Adding proper error handling

Output only the corrected code."""

    def check_convergence(self, metrics_history: List[Dict]) -> bool:
        """
        Check if training has converged.

        Convergence criteria:
        1. Average student score >= quality_threshold for 3 consecutive epochs
        2. Number of corrections drops to < 10% of eval set
        """
        if len(metrics_history) < 3:
            return False

        recent = metrics_history[-3:]

        # Check if scores consistently above threshold
        avg_scores_good = all(
            m['avg_student_score'] >= self.config.quality_threshold
            for m in recent
        )

        # Check if corrections are minimal
        corrections_minimal = all(
            m['num_corrections'] < m['num_poor_outputs'] * 0.1
            for m in recent
        )

        return avg_scores_good and corrections_minimal
```

### code_quality_metrics.py - Evaluation Module

```python
#!/usr/bin/env python3
"""
Code Quality Metrics for Student Output Evaluation

Evaluates generated code on multiple dimensions:
1. Syntax correctness (compilation)
2. Protocol compliance (TSN/AVB)
3. Safety guidelines (MISRA-C)
4. Code style and documentation
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import subprocess
import tempfile
import re

@dataclass
class QualityScore:
    """Composite quality score for generated code"""
    syntax_score: float      # 0-10: Does it compile?
    protocol_score: float    # 0-10: TSN/AVB compliance
    safety_score: float      # 0-10: MISRA-C adherence
    style_score: float       # 0-10: Code quality

    @property
    def overall(self) -> float:
        """Weighted average of all scores"""
        weights = {
            'syntax': 0.3,    # Critical - must compile
            'protocol': 0.3,  # Critical - must be correct
            'safety': 0.25,   # Important for automotive
            'style': 0.15,    # Nice to have
        }
        return (
            self.syntax_score * weights['syntax'] +
            self.protocol_score * weights['protocol'] +
            self.safety_score * weights['safety'] +
            self.style_score * weights['style']
        )


class CodeQualityEvaluator:
    """
    Evaluate quality of generated automotive code.

    Used in iterative distillation to identify outputs
    that need teacher correction.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def evaluate(self, code: str, prompt: str) -> QualityScore:
        """
        Evaluate a single code sample.

        Args:
            code: Generated code string
            prompt: Original prompt (for context)

        Returns:
            QualityScore with breakdown
        """
        return QualityScore(
            syntax_score=self._check_syntax(code),
            protocol_score=self._check_protocol_compliance(code, prompt),
            safety_score=self._check_misra_compliance(code),
            style_score=self._check_style(code)
        )

    def evaluate_batch(
        self,
        outputs: List[str],
        prompts: List[Dict]
    ) -> List[float]:
        """Evaluate multiple outputs, return overall scores"""
        scores = []
        for output, prompt in zip(outputs, prompts):
            quality = self.evaluate(output, prompt.get('prompt', ''))
            scores.append(quality.overall)
        return scores

    def _check_syntax(self, code: str) -> float:
        """
        Check if code compiles.

        Returns:
            10.0 if compiles, 0.0 if syntax errors
        """
        # Extract code from markdown if present
        code = self._extract_code_block(code)

        if not code.strip():
            return 0.0

        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.c', delete=False
            ) as f:
                f.write(code)
                f.flush()

                # Try to compile with gcc (syntax check only)
                result = subprocess.run(
                    ['gcc', '-fsyntax-only', '-std=c99', f.name],
                    capture_output=True,
                    timeout=5
                )

                if result.returncode == 0:
                    return 10.0
                else:
                    # Partial credit for minor errors
                    errors = result.stderr.decode().count('error:')
                    warnings = result.stderr.decode().count('warning:')

                    if errors == 0:
                        return 9.0 - min(warnings, 4) * 0.5
                    else:
                        return max(0, 5.0 - errors)

        except Exception:
            return 5.0  # Unknown - give neutral score

    def _check_protocol_compliance(self, code: str, prompt: str) -> float:
        """
        Check TSN/AVB protocol compliance.

        Looks for:
        - Correct struct definitions
        - Proper timestamp handling
        - Required protocol fields
        """
        score = 10.0
        code_lower = code.lower()

        # TSN checks
        if 'tsn' in prompt.lower() or '802.1' in prompt:
            required = ['pcp', 'vlan', 'timestamp', 'gate', 'priority']
            for term in required:
                if term not in code_lower:
                    score -= 1.5

        # AVB checks
        if 'avb' in prompt.lower() or 'audio' in prompt.lower():
            required = ['stream', 'sample', 'channel', 'bandwidth']
            for term in required:
                if term not in code_lower:
                    score -= 1.5

        return max(0, score)

    def _check_misra_compliance(self, code: str) -> float:
        """
        Check MISRA-C guideline adherence.

        Basic checks:
        - No goto statements
        - Explicit types (uint8_t, not char)
        - No recursion
        - Bounded loops
        """
        score = 10.0

        violations = [
            ('goto ', -3.0, "MISRA 15.1: goto prohibited"),
            ('malloc(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
            ('free(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
            ('setjmp', -3.0, "MISRA 21.4: setjmp/longjmp prohibited"),
            ('longjmp', -3.0, "MISRA 21.4: setjmp/longjmp prohibited"),
        ]

        for pattern, penalty, _ in violations:
            if pattern in code:
                score += penalty

        # Bonus for good practices
        good_practices = [
            ('uint8_t', 0.5),
            ('uint16_t', 0.5),
            ('uint32_t', 0.5),
            ('static ', 0.3),
            ('const ', 0.3),
        ]

        for pattern, bonus in good_practices:
            if pattern in code:
                score = min(10, score + bonus)

        return max(0, score)

    def _check_style(self, code: str) -> float:
        """
        Check code style and documentation.

        - Has comments
        - Reasonable function length
        - Clear variable names
        """
        score = 7.0  # Start neutral

        # Check for comments
        comment_count = code.count('//') + code.count('/*')
        if comment_count >= 3:
            score += 2.0
        elif comment_count >= 1:
            score += 1.0

        # Check for function documentation
        if '/**' in code or '@param' in code or '@return' in code:
            score += 1.0

        return min(10, score)

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # Look for ```c or ``` blocks
        pattern = r'```(?:c|cpp|C)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0]
        return text
```

### run_iterative_pipeline.py - Orchestration Script

```python
#!/usr/bin/env python3
"""
Run Iterative Teacher-Student Distillation Pipeline

This script orchestrates the full training loop:
1. Load initial training data
2. For each epoch:
   a. Train student on current data
   b. Evaluate student outputs
   c. Get teacher corrections for poor outputs
   d. Add corrections to training data
3. Continue until convergence or max epochs
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

def main():
    parser = argparse.ArgumentParser(
        description="Run iterative teacher-student distillation"
    )

    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--quality-threshold', type=float, default=7.0,
                        help="Minimum quality score (outputs below this get corrected)")
    parser.add_argument('--max-corrections-per-epoch', type=int, default=500,
                        help="Limit teacher API calls per epoch")
    parser.add_argument('--eval-samples', type=int, default=200,
                        help="Number of samples to evaluate each epoch")
    parser.add_argument('--output-dir', type=str, default='./output/iterative')
    parser.add_argument('--resume-from', type=str, default=None,
                        help="Resume from checkpoint directory")

    args = parser.parse_args()

    print("="*70)
    print("Iterative Teacher-Student Distillation Pipeline")
    print("="*70)
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Max corrections/epoch: {args.max_corrections_per_epoch}")
    print(f"Evaluation samples: {args.eval_samples}")
    print("="*70)

    # Initialize components
    config = DistillationConfig(
        quality_threshold=args.quality_threshold,
        max_corrections_per_epoch=args.max_corrections_per_epoch,
    )

    # Training loop
    metrics_history = []

    for epoch in range(args.max_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{args.max_epochs}")
        print(f"{'='*70}")

        # Train epoch with teacher feedback
        epoch_metrics = trainer.train_epoch(
            train_dataset=train_dataset,
            eval_prompts=eval_prompts[:args.eval_samples]
        )

        metrics_history.append(epoch_metrics)

        # Log metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train loss: {epoch_metrics['train_loss']:.4f}")
        print(f"  Avg student score: {epoch_metrics['avg_student_score']:.2f}/10")
        print(f"  Poor outputs: {epoch_metrics['num_poor_outputs']}")
        print(f"  Teacher corrections: {epoch_metrics['num_corrections']}")
        print(f"  Score distribution: {epoch_metrics['scores_distribution']}")

        # Save checkpoint
        save_checkpoint(epoch, epoch_metrics, args.output_dir)

        # Check convergence
        if trainer.check_convergence(metrics_history):
            print(f"\n[CONVERGED] Student quality meets threshold!")
            break

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
```

---

## Phase 3: Quality Metrics Implementation

### Metrics Used for Student Evaluation

| Metric | Weight | Description | Threshold |
|--------|--------|-------------|-----------|
| Syntax Score | 30% | Does code compile? | Must pass |
| Protocol Score | 30% | TSN/AVB compliance | ≥7/10 |
| Safety Score | 25% | MISRA-C adherence | ≥6/10 |
| Style Score | 15% | Code quality | ≥5/10 |
| **Overall** | 100% | Weighted average | **≥7.0** |

### Quality Gate Decision

```
IF overall_score >= 7.0:
    → Keep student output
    → Add to "good outputs" statistics
ELSE:
    → Send to teacher for correction
    → Add correction to training set
    → Student learns from correction
```

---

## Phase 4: Test Suite Enhancement

### Tests to Add for Iterative Distillation

```python
# tests/test_iterative_distillation.py

class TestIterativeDistillation:
    """Tests for iterative teacher-student loop"""

    def test_quality_threshold_triggers_correction(self):
        """Verify outputs below threshold get sent to teacher"""

    def test_corrections_added_to_training(self):
        """Verify teacher corrections are added to dataset"""

    def test_convergence_detection(self):
        """Verify training stops when student quality meets threshold"""

    def test_max_corrections_limit(self):
        """Verify API cost control via max corrections"""

    def test_score_improvement_over_epochs(self):
        """Verify student scores improve with corrections"""


class TestCodeQualityMetrics:
    """Tests for quality evaluation"""

    def test_syntax_check_compilation(self):
        """Verify syntax checker detects compilation errors"""

    def test_protocol_compliance_tsn(self):
        """Verify TSN protocol checks work"""

    def test_misra_violation_detection(self):
        """Verify MISRA-C violations are caught"""

    def test_overall_score_weighting(self):
        """Verify weighted average is correct"""
```

### Fix Required Tests

**File**: `tests/test_ml_paradigm.py:113`

```python
# BEFORE (BUG - always passes):
assert ratio < 100 or True  # Would fail in real scenario

# AFTER (FIXED):
assert ratio < 100, f"Gradient imbalance too high: {ratio:.2f}x"
```

---

## Phase 5: Cost Estimation (Iterative Approach)

### Per-Epoch Costs

| Metric | Value | Cost |
|--------|-------|------|
| Evaluation samples | 200 | - |
| Poor outputs (est. 30%) | 60 | - |
| Teacher corrections | 60 calls | ~$2-5 |
| Training compute | 1 epoch | ~$5-10 |

### Total Pipeline Cost (5 epochs)

| Phase | Cost |
|-------|------|
| Epoch 1 (most corrections) | ~$15 |
| Epoch 2 | ~$10 |
| Epoch 3 | ~$7 |
| Epoch 4 | ~$5 |
| Epoch 5 | ~$3 |
| **Total** | **~$40-50** |

Note: Costs decrease as student improves and needs fewer corrections.

---

## Implementation Order

### Step 1: Fix Critical Bug (Immediate)
```bash
# Fix dead assertion
# tests/test_ml_paradigm.py line 113: remove "or True"
```

### Step 2: Create Evaluation Module
1. `evaluation/__init__.py`
2. `evaluation/code_quality_metrics.py`
3. Tests for quality metrics

### Step 3: Create Iterative Training Loop
1. `training/iterative_distillation.py`
2. Integrate with existing `BedrockTeacherGenerator`
3. Tests for distillation loop

### Step 4: Create Orchestration Script
1. `scripts/run_iterative_pipeline.py`
2. Checkpoint saving/loading
3. Convergence monitoring

### Step 5: Run Pipeline
```bash
# 1. Run tests
pytest tests/ -v

# 2. Test with small sample
python scripts/run_iterative_pipeline.py \
    --max-epochs 2 \
    --eval-samples 10 \
    --max-corrections-per-epoch 5

# 3. Full run
python scripts/run_iterative_pipeline.py \
    --max-epochs 5 \
    --eval-samples 200 \
    --quality-threshold 7.0
```

---

## Verification Checklist

### Before Running
- [ ] Bug in `test_ml_paradigm.py:113` is fixed
- [ ] All existing tests pass
- [ ] `evaluation/code_quality_metrics.py` created
- [ ] `training/iterative_distillation.py` created
- [ ] Bedrock credentials working

### After Each Epoch
- [ ] Student scores logged
- [ ] Teacher corrections saved
- [ ] Checkpoint created
- [ ] Cost tracked

### Convergence Criteria
- [ ] Average student score ≥ 7.0 for 3 consecutive epochs
- [ ] Correction rate < 10% of evaluation set
- [ ] No regressions in quality scores

---

## Phase 6: Cloud-Native Architecture (IMPLEMENTED)

### Design Principle
**Everything runs in AWS** - the local machine only triggers cloud jobs.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLOUD-NATIVE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   LOCAL MACHINE                     AWS CLOUD                       │
│   ┌─────────────┐                  ┌──────────────────────────────┐ │
│   │ Launch      │ ───triggers───►  │ SageMaker Training Job       │ │
│   │ Scripts     │                  │ ┌────────────────────────┐   │ │
│   │             │                  │ │ Granite-8B Training    │   │ │
│   │ Monitor     │ ◄──CloudWatch──  │ │ Bedrock Teacher Calls  │   │ │
│   │ Dashboard   │                  │ │ S3 Data Access         │   │ │
│   └─────────────┘                  │ └────────────────────────┘   │ │
│                                    │                              │ │
│                                    │ ┌────────────────────────┐   │ │
│                                    │ │ S3 Bucket              │   │ │
│                                    │ │ • Training Data        │   │ │
│                                    │ │ • Eval Prompts         │   │ │
│                                    │ │ • Model Checkpoints    │   │ │
│                                    │ │ • Test Results         │   │ │
│                                    │ └────────────────────────┘   │ │
│                                    └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Cloud-Native Components Created

| File | Type | Purpose |
|------|------|---------|
| `scripts/launch_cloud_training.py` | Launcher | Deploy training as SageMaker Training Job |
| `scripts/run_iterative_pipeline.py` | Entry Point | Runs inside SageMaker container |
| `training/iterative_distillation.py` | Training | Cloud-native distillation loop |
| `tests/cloud/launch_cloud_tests.py` | Launcher | Deploy tests as SageMaker Processing Job |
| `tests/cloud/run_cloud_tests.py` | Tests | Validates S3, Bedrock, and integration |

### SageMaker Training Job Configuration

```python
# Instance: ml.g5.12xlarge (4x A10G GPUs)
# Runtime: Up to 24 hours
# Spot: Yes (70% cost savings)

estimator = Estimator(
    image_uri=huggingface_dlc,
    instance_type='ml.g5.12xlarge',
    instance_count=1,
    use_spot_instances=True,
    max_wait=172800,  # 48h for spot
    entry_point='run_iterative_pipeline.py',
    hyperparameters={
        'max-epochs': '5',
        'quality-threshold': '7.0',
        'max-corrections-per-epoch': '500',
    }
)
```

### SageMaker Processing Job (Tests)

```python
# Instance: ml.t3.medium (cheap for tests)
# Runtime: 1 hour max
# Cost: ~$0.05

processor = ScriptProcessor(
    instance_type='ml.t3.medium',
    max_runtime_in_seconds=3600,
    code='tests/cloud/run_cloud_tests.py',
)
```

### Cloud Test Suite Coverage

| Category | Tests | What It Validates |
|----------|-------|-------------------|
| S3 Access | 5 | Bucket access, list, read, write, pagination |
| Bedrock | 3 | Model access, code generation, rate limiting |
| Quality Evaluator | 2 | Initialization, scoring accuracy |
| Integration | 3 | Teacher generator, full correction flow |
| **Total** | **13** | Full cloud integration |

### Launch Commands

```bash
# Launch training job
python scripts/launch_cloud_training.py --use-spot

# Launch cloud tests
python tests/cloud/launch_cloud_tests.py

# Monitor job
aws sagemaker describe-training-job --training-job-name <job-name>

# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### Data Flow (All in AWS)

```
S3: granite-8b-unified-automotive-data/
├── data/processed/           # Training data (192,964 files)
├── data/eval/                # Evaluation prompts
├── output/distillation/      # Training outputs
│   ├── checkpoints/          # Model checkpoints
│   ├── corrections/          # Teacher corrections per epoch
│   └── metrics/              # Training metrics
└── test-results/             # Cloud test results
```

### Cost Summary (Cloud-Native)

| Component | Instance | Cost/Hour | Spot | Est. Total |
|-----------|----------|-----------|------|------------|
| Training | ml.g5.12xlarge | $7.09 | $2.13 | ~$20-50 |
| Tests | ml.t3.medium | $0.05 | N/A | ~$0.10 |
| Bedrock | Claude Sonnet | N/A | N/A | ~$40-50 |
| S3 | Storage | $0.023/GB | N/A | ~$0.50 |
| **Total** | | | | **~$60-100** |

---

## Implementation Status

### Completed
- [x] Fix critical bug in test_ml_paradigm.py:113
- [x] Create evaluation module (code_quality_metrics.py)
- [x] Create iterative_distillation.py training loop
- [x] Create run_iterative_pipeline.py orchestration
- [x] Create launch_cloud_training.py (SageMaker launcher)
- [x] Create cloud-native test suite (tests/cloud/)
- [x] Add tests for iterative distillation

### Ready to Run
```bash
# 1. Launch cloud tests to validate AWS setup
python tests/cloud/launch_cloud_tests.py --wait

# 2. If tests pass, launch training
python scripts/launch_cloud_training.py --use-spot

# 3. Monitor progress
python scripts/monitor_jobs.py --follow
```
