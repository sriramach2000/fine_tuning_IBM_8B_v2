"""
Tests for Training Callbacks (Early Stopping and NaN Detection)

Tests:
1. test_early_stopping_patience - Stop after 3 non-improvements
2. test_early_stopping_threshold - Threshold sensitivity
3. test_nan_detection_stops - Training stops on NaN loss
4. test_inf_detection_stops - Training stops on Inf loss
5. test_best_checkpoint_saved - Best model saved correctly
"""

import os
import sys
import math
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEarlyStopping:
    """Test suite for CustomEarlyStoppingCallback"""

    @pytest.fixture
    def callback_class(self):
        """Import the callback class"""
        from training.train_granite_qlora import CustomEarlyStoppingCallback
        return CustomEarlyStoppingCallback

    @pytest.fixture
    def create_trainer_args(self):
        """Create mock trainer args"""
        args = MagicMock()
        args.output_dir = '/tmp/test_output'
        return args

    # =========================================================================
    # Test 1: Early Stopping Patience
    # =========================================================================
    def test_early_stopping_patience(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Stop after 3 evaluations without improvement"""
        callback = callback_class(patience=3, threshold=0.0)

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Simulate 4 evaluations with no improvement
        eval_losses = [2.0, 2.0, 2.0, 2.0]

        for i, loss in enumerate(eval_losses):
            state.global_step = (i + 1) * 50
            metrics = {'eval_loss': loss}
            callback.on_evaluate(args, state, control, metrics=metrics)

            if i < 3:  # First 3 evaluations
                assert control.should_training_stop is False, f"Should not stop at eval {i+1}"
            else:  # 4th evaluation (patience=3 exceeded)
                assert control.should_training_stop is True, "Should stop after patience exceeded"

    # =========================================================================
    # Test 2: Early Stopping Threshold
    # =========================================================================
    def test_early_stopping_threshold(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Test threshold sensitivity - small improvements below threshold don't count"""
        callback = callback_class(patience=3, threshold=0.1)  # Require 0.1 improvement

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Small improvements that don't meet threshold
        eval_losses = [2.0, 1.95, 1.90, 1.85]  # Only 0.05 improvement each

        for i, loss in enumerate(eval_losses):
            state.global_step = (i + 1) * 50
            metrics = {'eval_loss': loss}
            callback.on_evaluate(args, state, control, metrics=metrics)

        # Should have stopped because improvements < threshold
        assert control.should_training_stop is True

        # Now test with significant improvements
        callback2 = callback_class(patience=3, threshold=0.1)
        control2 = MagicMock()
        control2.should_training_stop = False

        # Large improvements that meet threshold
        large_improvements = [2.0, 1.8, 1.6, 1.4]  # 0.2 improvement each

        for i, loss in enumerate(large_improvements):
            state.global_step = (i + 1) * 50
            metrics = {'eval_loss': loss}
            callback2.on_evaluate(args, state, control2, metrics=metrics)

        # Should NOT have stopped because improvements >= threshold
        assert control2.should_training_stop is False

    # =========================================================================
    # Test 3: Best Checkpoint Saved
    # =========================================================================
    def test_best_checkpoint_saved(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Verify best model checkpoint is tracked correctly"""
        callback = callback_class(patience=3, threshold=0.0, save_best=True)

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Simulate improving then worsening loss
        eval_sequence = [2.0, 1.5, 1.8, 1.9, 2.0]

        for i, loss in enumerate(eval_sequence):
            state.global_step = (i + 1) * 50
            metrics = {'eval_loss': loss}
            callback.on_evaluate(args, state, control, metrics=metrics)

        # Best loss should be 1.5 (at step 100)
        assert callback.best_loss == 1.5
        assert callback.best_step == 100


class TestNaNInfDetection:
    """Test suite for NaNInfDetectionCallback"""

    @pytest.fixture
    def callback_class(self):
        """Import the callback class"""
        from training.train_granite_qlora import NaNInfDetectionCallback
        return NaNInfDetectionCallback

    @pytest.fixture
    def create_trainer_args(self):
        """Create mock trainer args"""
        return MagicMock()

    # =========================================================================
    # Test 4: NaN Detection Stops Training
    # =========================================================================
    def test_nan_detection_stops(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Training stops immediately on NaN loss"""
        callback = callback_class()

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Normal loss - should continue
        logs = {'loss': 2.5}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is False

        # NaN loss - should stop
        logs = {'loss': float('nan')}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is True

    # =========================================================================
    # Test 5: Inf Detection Stops Training
    # =========================================================================
    def test_inf_detection_stops(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Training stops immediately on Inf loss"""
        callback = callback_class()

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Normal loss - should continue
        logs = {'loss': 2.5}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is False

        # Positive Inf - should stop
        logs = {'loss': float('inf')}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is True

    def test_negative_inf_detection(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Training stops on negative Inf loss too"""
        callback = callback_class()

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # Negative Inf - should also stop
        logs = {'loss': float('-inf')}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is True

    def test_no_loss_in_logs(self, callback_class, mock_trainer_state, mock_trainer_control, create_trainer_args):
        """Handle missing loss gracefully"""
        callback = callback_class()

        args = create_trainer_args
        state = mock_trainer_state
        control = mock_trainer_control

        # No loss key - should continue
        logs = {'learning_rate': 1e-4}
        callback.on_log(args, state, control, logs=logs)
        assert control.should_training_stop is False

        # None logs - should continue
        callback.on_log(args, state, control, logs=None)
        assert control.should_training_stop is False
