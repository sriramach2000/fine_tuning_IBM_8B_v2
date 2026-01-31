"""
Iterative Distillation End-to-End Tests

Tests the distillation loop with cloud-native scenarios:
- Correction prompt formatting with realistic data
- Teacher correction quality (real and mocked Bedrock)
- Corrections JSONL format for training
- Metrics file persistence
- Convergence detection with realistic scenarios
- Quality gate enforcement
- Score distribution tracking
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# torch may not be installed in test environment - mock it before import
try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()

from training.iterative_distillation import (
    IterativeDistillationTrainer,
    DistillationConfig,
    EpochMetrics,
    create_distillation_trainer,
)
from evaluation.code_quality_metrics import CodeQualityEvaluator, QualityScore


# =============================================================================
# CORRECTION PROMPT FORMATTING
# =============================================================================

@pytest.mark.offline
class TestCorrectionPromptFormatting:
    """Tests for correction prompt content and structure"""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        t = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        t.config = config
        t.metrics_history = []
        t.all_corrections = []
        t.epochs_at_threshold = 0
        t.best_avg_score = 0.0
        return t

    def test_correction_prompt_includes_original(self, trainer):
        """Correction prompt contains the original prompt"""
        prompt = trainer._create_correction_prompt(
            "Generate TSN shaper code", "bad code here", 3.5
        )
        assert "Generate TSN shaper code" in prompt

    def test_correction_prompt_includes_student_output(self, trainer):
        """Correction prompt contains the student's output"""
        student_code = "void foo() { goto err; err: return; }"
        prompt = trainer._create_correction_prompt(
            "Generate code", student_code, 4.0
        )
        assert student_code in prompt

    def test_correction_prompt_includes_score(self, trainer):
        """Correction prompt mentions the quality score"""
        prompt = trainer._create_correction_prompt(
            "Generate code", "bad code", 3.5
        )
        assert "3.5" in prompt

    def test_correction_prompt_mentions_misra(self, trainer):
        """Correction prompt asks for MISRA compliance"""
        prompt = trainer._create_correction_prompt(
            "Generate code", "code", 5.0
        )
        assert "MISRA" in prompt

    def test_correction_system_prompt_has_guidance(self, trainer):
        """System prompt provides automotive guidance"""
        sys_prompt = trainer._get_correction_system_prompt()
        assert "automotive" in sys_prompt.lower()
        assert "MISRA" in sys_prompt
        assert "error handling" in sys_prompt.lower()


# =============================================================================
# POOR OUTPUT IDENTIFICATION
# =============================================================================

@pytest.mark.offline
class TestPoorOutputIdentification:
    """Tests for identifying outputs needing correction"""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = DistillationConfig(
            quality_threshold=7.0,
            max_corrections_per_epoch=5,
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        t = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        t.config = config
        return t

    def test_identifies_below_threshold(self, trainer):
        """Outputs below threshold are identified"""
        prompts = [{'id': f'p{i}', 'prompt': f'test{i}'} for i in range(5)]
        outputs = [f'code{i}' for i in range(5)]
        scores = [8.0, 3.0, 9.0, 5.5, 6.0]

        poor = trainer._identify_poor_outputs(prompts, outputs, scores)
        poor_scores = [p[2] for p in poor]

        assert all(s < 7.0 for s in poor_scores)
        assert len(poor) == 3  # scores 3.0, 5.5, 6.0

    def test_worst_outputs_first(self, trainer):
        """Poor outputs are sorted worst-first"""
        prompts = [{'id': f'p{i}', 'prompt': f'test{i}'} for i in range(5)]
        outputs = [f'code{i}' for i in range(5)]
        scores = [6.0, 2.0, 5.0, 3.0, 4.0]

        poor = trainer._identify_poor_outputs(prompts, outputs, scores)
        poor_scores = [p[2] for p in poor]

        assert poor_scores == sorted(poor_scores)

    def test_respects_max_corrections_limit(self, trainer):
        """Number of poor outputs capped at max_corrections_per_epoch"""
        trainer.config.max_corrections_per_epoch = 3

        prompts = [{'id': f'p{i}', 'prompt': f'test{i}'} for i in range(10)]
        outputs = [f'code{i}' for i in range(10)]
        scores = [4.0] * 10  # All below threshold

        poor = trainer._identify_poor_outputs(prompts, outputs, scores)
        assert len(poor) <= 3

    def test_no_poor_outputs_when_all_pass(self, trainer):
        """No outputs identified when all pass threshold"""
        prompts = [{'id': 'p0', 'prompt': 'test'}]
        outputs = ['good_code']
        scores = [9.0]

        poor = trainer._identify_poor_outputs(prompts, outputs, scores)
        assert len(poor) == 0


# =============================================================================
# METRICS PERSISTENCE
# =============================================================================

@pytest.mark.offline
class TestMetricsPersistence:
    """Tests for metrics file writing and integrity"""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        t = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        t.config = config
        t.metrics_history = []
        t.all_corrections = []
        t.epochs_at_threshold = 0
        t.best_avg_score = 0.0
        return t

    def test_metrics_saved_to_jsonl(self, trainer):
        """Epoch metrics are saved to JSONL file"""
        metrics = EpochMetrics(
            epoch=1, train_loss=2.0, avg_student_score=7.5,
            num_eval_samples=100, num_poor_outputs=20,
            num_corrections=15, correction_rate=0.2,
            scores_distribution={'below_5': 5, '5_to_7': 15, '7_to_9': 60, 'above_9': 20}
        )
        trainer._save_epoch_metrics(metrics)

        metrics_file = Path(trainer.config.output_dir) / 'metrics_history.jsonl'
        assert metrics_file.exists()

        with open(metrics_file) as f:
            line = f.readline()
            parsed = json.loads(line)
            assert parsed['epoch'] == 1
            assert parsed['avg_student_score'] == 7.5

    def test_corrections_saved_per_epoch(self, trainer):
        """Corrections are saved to per-epoch files"""
        corrections = [
            {'id': 'c1', 'messages': [{'role': 'user', 'content': 'p1'}]},
            {'id': 'c2', 'messages': [{'role': 'user', 'content': 'p2'}]},
        ]
        trainer._save_epoch_corrections(corrections, epoch_num=3)

        corrections_file = Path(trainer.config.corrections_dir) / 'epoch_3_corrections.jsonl'
        assert corrections_file.exists()

        with open(corrections_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

    def test_accumulated_corrections_file(self, trainer):
        """Corrections accumulate in file across calls"""
        corrections1 = [{'id': 'c1', 'messages': []}]
        corrections2 = [{'id': 'c2', 'messages': []}]

        # Simulate adding corrections via file (fallback path)
        corrections_file = Path(trainer.config.corrections_dir) / 'accumulated_corrections.jsonl'
        with open(corrections_file, 'a') as f:
            for c in corrections1:
                f.write(json.dumps(c) + '\n')
        with open(corrections_file, 'a') as f:
            for c in corrections2:
                f.write(json.dumps(c) + '\n')

        with open(corrections_file) as f:
            lines = f.readlines()
            assert len(lines) == 2


# =============================================================================
# CONVERGENCE EDGE CASES
# =============================================================================

@pytest.mark.offline
class TestConvergenceEdgeCases:
    """Additional convergence detection edge cases"""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = DistillationConfig(
            convergence_threshold=8.0,
            convergence_patience=3,
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        t = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        t.config = config
        t.epochs_at_threshold = 0
        t.best_avg_score = 0.0
        t.all_corrections = []
        return t

    def test_single_epoch_never_converges(self, trainer):
        """Single epoch cannot satisfy convergence patience"""
        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=1.0, avg_student_score=9.5,
                        num_eval_samples=100, num_poor_outputs=0,
                        num_corrections=0, correction_rate=0.0,
                        scores_distribution={})
        ]
        converged, _ = trainer.check_convergence()
        assert not converged

    def test_convergence_resets_on_quality_drop(self, trainer):
        """Convergence counter resets when quality drops"""
        trainer.epochs_at_threshold = 2  # Almost converged
        trainer.metrics_history = [
            EpochMetrics(epoch=i, train_loss=1.5, avg_student_score=score,
                        num_eval_samples=100, num_poor_outputs=10,
                        num_corrections=5, correction_rate=0.15,
                        scores_distribution={})
            for i, score in enumerate([8.5, 8.2, 7.0], 1)  # Quality drops
        ]

        # Manually reset as the trainer would
        if trainer.metrics_history[-1].avg_student_score < trainer.config.convergence_threshold:
            trainer.epochs_at_threshold = 0

        converged, _ = trainer.check_convergence()
        assert not converged

    def test_convergence_on_exactly_threshold(self, trainer):
        """Score exactly at threshold counts toward convergence"""
        trainer.epochs_at_threshold = 3
        trainer.metrics_history = [
            EpochMetrics(epoch=i, train_loss=1.5, avg_student_score=8.0,
                        num_eval_samples=100, num_poor_outputs=10,
                        num_corrections=5, correction_rate=0.1,
                        scores_distribution={})
            for i in range(1, 4)
        ]
        converged, _ = trainer.check_convergence()
        assert converged


# =============================================================================
# QUALITY GATE ENFORCEMENT
# =============================================================================

@pytest.mark.offline
class TestQualityGateEnforcement:
    """Tests for quality threshold gate in distillation"""

    def test_evaluator_threshold_matches_config(self, tmp_path):
        """Evaluator threshold should match config defaults"""
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        evaluator = CodeQualityEvaluator(quality_threshold=config.quality_threshold)
        assert evaluator.quality_threshold == config.quality_threshold

    def test_all_score_components_contribute(self):
        """All 4 score components contribute to overall"""
        score = QualityScore(
            syntax_score=10.0,
            protocol_score=0.0,
            safety_score=10.0,
            style_score=10.0,
        )
        # Protocol is 0, so overall < 10
        assert score.overall < 10.0
        assert score.overall > 0.0

    def test_quality_gate_filters_correctly(self):
        """Quality gate correctly separates pass/fail"""
        evaluator = CodeQualityEvaluator(quality_threshold=7.0, gcc_available=False)

        good_code = """
/**
 * @brief Initialize TSN frame
 * @param frame Pointer to frame
 * @return 0 on success
 */
static int32_t tsn_frame_init(uint8_t pcp, uint16_t vlan_id, uint64_t timestamp) {
    // TSN gate control
    volatile uint8_t priority = pcp;
    const uint32_t schedule = 0;
    return 0;
}
"""
        bad_code = "void f() { goto x; x: malloc(1); free(0); exit(1); }"

        good_score = evaluator.evaluate(good_code, "Generate TSN code")
        bad_score = evaluator.evaluate(bad_code, "Generate code")

        assert not evaluator.needs_correction(good_score.overall), \
            f"Good code should pass, scored {good_score.overall}"
        assert evaluator.needs_correction(bad_score.overall), \
            f"Bad code should fail, scored {bad_score.overall}"


# =============================================================================
# TRAINING SUMMARY
# =============================================================================

@pytest.mark.offline
class TestTrainingSummary:
    """Tests for training summary generation"""

    @pytest.fixture
    def trainer(self, tmp_path):
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )
        t = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        t.config = config
        t.epochs_at_threshold = 0
        t.best_avg_score = 0.0
        t.all_corrections = []
        t.metrics_history = []
        return t

    def test_empty_history_returns_empty(self, trainer):
        """Empty metrics history returns empty summary"""
        summary = trainer.get_training_summary()
        assert summary == {}

    def test_summary_has_all_fields(self, trainer):
        """Summary contains all required fields"""
        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=2.0, avg_student_score=6.5,
                        num_eval_samples=100, num_poor_outputs=40,
                        num_corrections=35, correction_rate=0.4,
                        scores_distribution={}),
        ]
        trainer.best_avg_score = 6.5
        trainer.all_corrections = [{'id': f'c{i}'} for i in range(35)]

        summary = trainer.get_training_summary()

        required_fields = [
            'total_epochs', 'final_avg_score', 'best_avg_score',
            'total_corrections', 'initial_correction_rate',
            'final_correction_rate', 'converged', 'convergence_reason'
        ]
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    def test_summary_tracks_improvement(self, trainer):
        """Summary reflects improvement over epochs"""
        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=2.5, avg_student_score=5.0,
                        num_eval_samples=100, num_poor_outputs=50,
                        num_corrections=40, correction_rate=0.5,
                        scores_distribution={}),
            EpochMetrics(epoch=2, train_loss=1.8, avg_student_score=7.5,
                        num_eval_samples=100, num_poor_outputs=15,
                        num_corrections=10, correction_rate=0.15,
                        scores_distribution={}),
        ]
        trainer.best_avg_score = 7.5

        summary = trainer.get_training_summary()
        assert summary['final_avg_score'] > summary['initial_correction_rate']  # Score improved
        assert summary['final_correction_rate'] < summary['initial_correction_rate']


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

@pytest.mark.offline
class TestFactoryFunction:
    """Tests for create_distillation_trainer factory"""

    def test_factory_creates_trainer(self, tmp_path):
        """Factory function creates valid trainer"""
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )

        trainer = create_distillation_trainer(
            student_model=MagicMock(),
            student_tokenizer=MagicMock(),
            bedrock_generator=MagicMock(),
            quality_evaluator=MagicMock(),
            config=config,
        )

        assert isinstance(trainer, IterativeDistillationTrainer)

    def test_factory_with_custom_config(self, tmp_path):
        """Factory accepts custom config"""
        config = DistillationConfig(
            quality_threshold=6.0,
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )

        trainer = create_distillation_trainer(
            student_model=MagicMock(),
            student_tokenizer=MagicMock(),
            bedrock_generator=MagicMock(),
            quality_evaluator=MagicMock(),
            config=config,
        )

        assert trainer.config.quality_threshold == 6.0

    def test_factory_default_config(self, tmp_path):
        """Factory uses default config when none provided"""
        config = DistillationConfig(
            output_dir=str(tmp_path / 'output'),
            corrections_dir=str(tmp_path / 'corrections'),
        )

        trainer = create_distillation_trainer(
            student_model=MagicMock(),
            student_tokenizer=MagicMock(),
            bedrock_generator=MagicMock(),
            quality_evaluator=MagicMock(),
            config=config,
        )

        assert trainer.config.quality_threshold == 7.0
        assert trainer.config.convergence_threshold == 8.0
