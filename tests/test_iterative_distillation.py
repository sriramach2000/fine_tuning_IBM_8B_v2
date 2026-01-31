"""
Tests for Iterative Teacher-Student Distillation

Tests the core distillation logic:
1. Quality evaluation metrics
2. Teacher correction triggering
3. Convergence detection
4. Correction integration into training

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CATEGORY 1: QUALITY EVALUATION (8 tests)
# =============================================================================

class TestCodeQualityEvaluator:
    """Tests for code quality evaluation metrics"""

    def test_quality_score_calculation(self):
        """Verify overall score is weighted correctly"""
        from evaluation.code_quality_metrics import QualityScore

        score = QualityScore(
            syntax_score=10.0,
            protocol_score=10.0,
            safety_score=10.0,
            style_score=10.0
        )

        # All 10s should give 10.0 overall
        assert score.overall == 10.0

    def test_quality_score_weighting(self):
        """Verify weights are applied correctly"""
        from evaluation.code_quality_metrics import QualityScore

        # Syntax and protocol have higher weight (0.3 each)
        score = QualityScore(
            syntax_score=0.0,    # 30% weight
            protocol_score=10.0,  # 30% weight
            safety_score=10.0,    # 25% weight
            style_score=10.0      # 15% weight
        )

        # Expected: 0*0.3 + 10*0.3 + 10*0.25 + 10*0.15 = 7.0
        assert score.overall == pytest.approx(7.0, abs=0.01)

    def test_needs_correction_below_threshold(self):
        """Verify outputs below threshold are flagged for correction"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(quality_threshold=7.0)

        assert evaluator.needs_correction(6.5)  # Below threshold
        assert evaluator.needs_correction(3.0)  # Far below
        assert not evaluator.needs_correction(7.0)  # At threshold
        assert not evaluator.needs_correction(9.0)  # Above

    def test_heuristic_syntax_check_balanced_braces(self):
        """Verify heuristic syntax check catches unbalanced braces"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(gcc_available=False)

        good_code = """
        void foo() {
            if (x) {
                bar();
            }
        }
        """
        bad_code = """
        void foo() {
            if (x) {
                bar();
            // missing closing braces
        """

        good_score = evaluator._heuristic_syntax_check(good_code)
        bad_score = evaluator._heuristic_syntax_check(bad_code)

        assert good_score > bad_score

    def test_protocol_compliance_tsn_keywords(self):
        """Verify TSN protocol compliance checking"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator()

        good_code = """
        struct tsn_frame {
            uint8_t pcp;
            uint16_t vlan_id;
            uint64_t timestamp;
            uint8_t priority;
            gate_control_list_t gcl;
        };
        """

        bad_code = """
        void process() {
            int x = 0;
            return;
        }
        """

        tsn_prompt = "Generate TSN time-aware shaper code"

        good_score = evaluator._check_protocol_compliance(good_code, tsn_prompt)
        bad_score = evaluator._check_protocol_compliance(bad_code, tsn_prompt)

        assert good_score > bad_score

    def test_protocol_compliance_avb_keywords(self):
        """Verify AVB protocol compliance checking"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator()

        good_code = """
        struct avb_stream {
            uint32_t stream_id;
            uint32_t sample_rate;
            uint8_t channels;
            uint32_t bandwidth;
        };
        """

        avb_prompt = "Generate AVB audio stream code"

        score = evaluator._check_protocol_compliance(good_code, avb_prompt)
        assert score >= 7.0

    def test_misra_violation_detection(self):
        """Verify MISRA-C violations are detected"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator()

        # Code with MISRA violations
        bad_code = """
        void process() {
            char* ptr = malloc(100);  // MISRA 21.3 violation
            goto error;               // MISRA 15.1 violation
        error:
            free(ptr);
        }
        """

        # Code following MISRA guidelines
        good_code = """
        static int32_t process(uint8_t* buffer, uint32_t size) {
            if (buffer == NULL) {
                return -1;
            }
            return 0;
        }
        """

        bad_score = evaluator._check_misra_compliance(bad_code)
        good_score = evaluator._check_misra_compliance(good_code)

        assert good_score > bad_score

    def test_code_block_extraction(self):
        """Verify code is extracted from markdown blocks"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator()

        markdown = """
        Here is some C code:

        ```c
        void foo() {
            return;
        }
        ```

        This is the explanation.
        """

        extracted = evaluator._extract_code_block(markdown)
        assert "void foo()" in extracted
        assert "explanation" not in extracted


# =============================================================================
# CATEGORY 2: TEACHER CORRECTION FLOW (6 tests)
# =============================================================================

class TestTeacherCorrectionFlow:
    """Tests for teacher correction triggering and integration"""

    def test_filter_for_correction(self):
        """Verify poor outputs are filtered correctly"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(quality_threshold=7.0)

        prompts = [
            {'id': 'p1', 'prompt': 'Generate TSN code'},
            {'id': 'p2', 'prompt': 'Generate AVB code'},
            {'id': 'p3', 'prompt': 'Generate Ethernet code'},
        ]
        outputs = ['code1', 'code2', 'code3']
        scores = [5.0, 8.0, 6.5]  # p1 and p3 need correction

        poor = evaluator.filter_for_correction(outputs, prompts, scores)

        assert len(poor) == 2
        assert poor[0][0]['id'] == 'p1'  # First poor output
        assert poor[1][0]['id'] == 'p3'  # Second poor output

    def test_correction_prompt_format(self):
        """Verify correction prompt is properly formatted"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig

        # Create minimal trainer for testing
        config = DistillationConfig()
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config

        prompt = trainer._create_correction_prompt(
            original_prompt="Generate TSN code",
            student_output="void foo() {}",
            score=4.5
        )

        assert "Generate TSN code" in prompt
        assert "void foo() {}" in prompt
        assert "4.5/10" in prompt
        assert "MISRA-C" in prompt

    def test_correction_system_prompt_content(self):
        """Verify system prompt includes required guidance"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig

        config = DistillationConfig()
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config

        system_prompt = trainer._get_correction_system_prompt()

        assert "MISRA-C" in system_prompt
        assert "TSN/AVB" in system_prompt or "automotive" in system_prompt.lower()
        assert "error handling" in system_prompt.lower()

    def test_max_corrections_limit_respected(self):
        """Verify max corrections per epoch is respected"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig

        config = DistillationConfig(max_corrections_per_epoch=5)

        # Create mock trainer
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config

        # 10 poor outputs, but limit is 5
        poor_outputs = [
            ({'id': f'p{i}', 'prompt': 'test'}, f'output{i}', 5.0)
            for i in range(10)
        ]

        # Identify should respect limit
        limited = trainer._identify_poor_outputs(
            prompts=[{'id': f'p{i}', 'prompt': 'test'} for i in range(10)],
            outputs=[f'output{i}' for i in range(10)],
            scores=[5.0] * 10
        )

        assert len(limited) <= config.max_corrections_per_epoch

    def test_corrections_sorted_by_worst_first(self):
        """Verify poorest outputs get corrected first"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig

        config = DistillationConfig(max_corrections_per_epoch=3)
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config

        prompts = [{'id': f'p{i}', 'prompt': 'test'} for i in range(5)]
        outputs = [f'output{i}' for i in range(5)]
        scores = [6.0, 3.0, 5.0, 4.0, 2.0]  # Worst: 2.0, 3.0, 4.0

        limited = trainer._identify_poor_outputs(prompts, outputs, scores)

        # Should get worst first
        assert limited[0][2] == 2.0  # Worst score first
        assert limited[1][2] == 3.0
        assert limited[2][2] == 4.0

    def test_correction_message_format(self):
        """Verify corrections have correct message format for training"""
        # Correction should have messages array suitable for SFT training
        correction = {
            'id': 'p1_correction_e1',
            'messages': [
                {'role': 'user', 'content': 'Generate TSN code'},
                {'role': 'assistant', 'content': 'void tsn_process() {...}'}
            ]
        }

        assert len(correction['messages']) == 2
        assert correction['messages'][0]['role'] == 'user'
        assert correction['messages'][1]['role'] == 'assistant'


# =============================================================================
# CATEGORY 3: CONVERGENCE DETECTION (5 tests)
# =============================================================================

class TestConvergenceDetection:
    """Tests for training convergence detection"""

    def test_convergence_requires_minimum_epochs(self):
        """Verify convergence requires minimum epochs"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig, EpochMetrics

        config = DistillationConfig(convergence_patience=3)
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config
        trainer.epochs_at_threshold = 0

        # Only 2 epochs of history - not enough
        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=2.0, avg_student_score=8.5,
                        num_eval_samples=100, num_poor_outputs=20,
                        num_corrections=15, correction_rate=0.2,
                        scores_distribution={}),
            EpochMetrics(epoch=2, train_loss=1.5, avg_student_score=8.7,
                        num_eval_samples=100, num_poor_outputs=10,
                        num_corrections=8, correction_rate=0.1,
                        scores_distribution={}),
        ]

        converged, reason = trainer.check_convergence()
        assert not converged
        assert "Not enough epochs" in reason

    def test_convergence_on_high_quality(self):
        """Verify convergence when quality is consistently high"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig, EpochMetrics

        config = DistillationConfig(
            convergence_threshold=8.0,
            convergence_patience=3
        )
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config
        trainer.epochs_at_threshold = 3  # 3 epochs at threshold

        trainer.metrics_history = [
            EpochMetrics(epoch=i, train_loss=2.0-i*0.3, avg_student_score=8.0+i*0.2,
                        num_eval_samples=100, num_poor_outputs=20-i*5,
                        num_corrections=15-i*4, correction_rate=0.2-i*0.05,
                        scores_distribution={})
            for i in range(1, 4)
        ]

        converged, reason = trainer.check_convergence()
        assert converged
        assert "quality" in reason.lower()

    def test_convergence_on_low_correction_rate(self):
        """Verify convergence when correction rate is minimal"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig, EpochMetrics

        config = DistillationConfig(convergence_patience=3)
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config
        trainer.epochs_at_threshold = 0

        # Low correction rate for 3 epochs
        trainer.metrics_history = [
            EpochMetrics(epoch=i, train_loss=2.0, avg_student_score=7.5,
                        num_eval_samples=100, num_poor_outputs=5,
                        num_corrections=3, correction_rate=0.05,  # < 10%
                        scores_distribution={})
            for i in range(1, 4)
        ]

        converged, reason = trainer.check_convergence()
        assert converged
        assert "10%" in reason

    def test_no_convergence_on_fluctuating_quality(self):
        """Verify no convergence when quality fluctuates"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig, EpochMetrics

        config = DistillationConfig(
            convergence_threshold=8.0,
            convergence_patience=3
        )
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config
        trainer.epochs_at_threshold = 0

        # Fluctuating scores
        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=2.0, avg_student_score=8.5,
                        num_eval_samples=100, num_poor_outputs=10,
                        num_corrections=8, correction_rate=0.1,
                        scores_distribution={}),
            EpochMetrics(epoch=2, train_loss=1.8, avg_student_score=7.2,  # Drop
                        num_eval_samples=100, num_poor_outputs=30,
                        num_corrections=25, correction_rate=0.3,
                        scores_distribution={}),
            EpochMetrics(epoch=3, train_loss=1.5, avg_student_score=8.8,  # Up again
                        num_eval_samples=100, num_poor_outputs=8,
                        num_corrections=6, correction_rate=0.08,
                        scores_distribution={}),
        ]

        converged, reason = trainer.check_convergence()
        assert not converged

    def test_training_summary_accuracy(self):
        """Verify training summary captures correct metrics"""
        from training.iterative_distillation import IterativeDistillationTrainer, DistillationConfig, EpochMetrics

        config = DistillationConfig()
        trainer = IterativeDistillationTrainer.__new__(IterativeDistillationTrainer)
        trainer.config = config
        trainer.best_avg_score = 8.9
        trainer.all_corrections = [{'id': f'c{i}'} for i in range(25)]
        trainer.epochs_at_threshold = 0

        trainer.metrics_history = [
            EpochMetrics(epoch=1, train_loss=2.0, avg_student_score=6.5,
                        num_eval_samples=100, num_poor_outputs=40,
                        num_corrections=35, correction_rate=0.4,
                        scores_distribution={}),
            EpochMetrics(epoch=2, train_loss=1.5, avg_student_score=7.8,
                        num_eval_samples=100, num_poor_outputs=15,
                        num_corrections=12, correction_rate=0.15,
                        scores_distribution={}),
        ]

        summary = trainer.get_training_summary()

        assert summary['total_epochs'] == 2
        assert summary['final_avg_score'] == 7.8
        assert summary['best_avg_score'] == 8.9
        assert summary['total_corrections'] == 25
        assert summary['initial_correction_rate'] == 0.4
        assert summary['final_correction_rate'] == 0.15


# =============================================================================
# CATEGORY 4: EPOCH METRICS (4 tests)
# =============================================================================

class TestEpochMetrics:
    """Tests for epoch metrics tracking"""

    def test_epoch_metrics_to_dict(self):
        """Verify metrics serialize to dict correctly"""
        from training.iterative_distillation import EpochMetrics

        metrics = EpochMetrics(
            epoch=1,
            train_loss=2.5,
            avg_student_score=7.2,
            num_eval_samples=100,
            num_poor_outputs=30,
            num_corrections=25,
            correction_rate=0.3,
            scores_distribution={'below_5': 5, '5_to_7': 10, '7_to_9': 70, 'above_9': 15}
        )

        d = metrics.to_dict()

        assert d['epoch'] == 1
        assert d['train_loss'] == 2.5
        assert d['avg_student_score'] == 7.2
        assert d['correction_rate'] == 0.3
        assert 'timestamp' in d

    def test_score_distribution_tracking(self):
        """Verify score distribution is tracked"""
        from training.iterative_distillation import EpochMetrics

        metrics = EpochMetrics(
            epoch=1,
            train_loss=2.0,
            avg_student_score=7.5,
            num_eval_samples=100,
            num_poor_outputs=25,
            num_corrections=20,
            correction_rate=0.25,
            scores_distribution={
                'below_5': 10,
                '5_to_7': 15,
                '7_to_9': 55,
                'above_9': 20
            }
        )

        total = sum(metrics.scores_distribution.values())
        assert total == 100  # Should sum to eval samples

    def test_correction_rate_calculation(self):
        """Verify correction rate is calculated correctly"""
        num_eval = 200
        num_poor = 50
        expected_rate = 50 / 200  # 0.25

        from training.iterative_distillation import EpochMetrics

        metrics = EpochMetrics(
            epoch=1,
            train_loss=2.0,
            avg_student_score=7.0,
            num_eval_samples=num_eval,
            num_poor_outputs=num_poor,
            num_corrections=45,
            correction_rate=expected_rate,
            scores_distribution={}
        )

        assert metrics.correction_rate == 0.25

    def test_metrics_have_timestamp(self):
        """Verify metrics include timestamp"""
        from training.iterative_distillation import EpochMetrics
        from datetime import datetime

        metrics = EpochMetrics(
            epoch=1,
            train_loss=2.0,
            avg_student_score=7.0,
            num_eval_samples=100,
            num_poor_outputs=20,
            num_corrections=15,
            correction_rate=0.2,
            scores_distribution={}
        )

        assert metrics.timestamp is not None
        # Should be ISO format
        datetime.fromisoformat(metrics.timestamp)


# =============================================================================
# CATEGORY 5: DISTILLATION CONFIG (4 tests)
# =============================================================================

class TestDistillationConfig:
    """Tests for distillation configuration"""

    def test_default_config_values(self):
        """Verify default config values are sensible"""
        from training.iterative_distillation import DistillationConfig

        config = DistillationConfig()

        assert config.quality_threshold == 7.0
        assert config.convergence_threshold == 8.0
        assert config.max_corrections_per_epoch == 500
        assert config.max_parallel_teacher_calls == 10

    def test_config_uses_sagemaker_env_vars(self):
        """Verify config picks up SageMaker environment variables"""
        import os
        from training.iterative_distillation import DistillationConfig

        # Mock SageMaker environment
        os.environ['SM_CHANNEL_TRAIN'] = '/opt/ml/input/data/train'
        os.environ['SM_OUTPUT_DATA_DIR'] = '/opt/ml/output'

        config = DistillationConfig()

        assert config.train_data_dir == '/opt/ml/input/data/train'
        assert config.output_dir == '/opt/ml/output'

        # Clean up
        del os.environ['SM_CHANNEL_TRAIN']
        del os.environ['SM_OUTPUT_DATA_DIR']

    def test_custom_config_values(self):
        """Verify custom config values are applied"""
        from training.iterative_distillation import DistillationConfig

        config = DistillationConfig(
            quality_threshold=6.0,
            max_corrections_per_epoch=100,
            teacher_model='custom-model-id'
        )

        assert config.quality_threshold == 6.0
        assert config.max_corrections_per_epoch == 100
        assert config.teacher_model == 'custom-model-id'

    def test_config_creates_directories(self):
        """Verify config creates required directories"""
        import tempfile
        from training.iterative_distillation import DistillationConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DistillationConfig(
                output_dir=f"{tmpdir}/output",
                corrections_dir=f"{tmpdir}/output/corrections"
            )

            from pathlib import Path
            assert Path(config.corrections_dir).exists()
