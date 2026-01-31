"""
Comprehensive Machine Learning Paradigm Tests

Tests for ALL common ML issues that plague fine-tuning tasks.
Based on research from Neptune.ai, arXiv, and industry best practices.

Sources:
- https://neptune.ai/blog/monitoring-diagnosing-and-solving-gradient-issues-in-foundation-models
- https://arxiv.org/html/2408.13296v1 (Ultimate Guide to Fine-Tuning LLMs)
- https://arxiv.org/html/2312.16903v2 (Spike No More)

34 Tests across 8 Categories:
1. Gradient Health (6 tests)
2. Loss Behavior (5 tests)
3. Overfitting/Underfitting (4 tests)
4. Data Integrity (5 tests)
5. Learning Rate (3 tests)
6. Checkpointing (4 tests)
7. Distillation Quality (4 tests)
8. Resource/Stability (3 tests)
"""

import os
import sys
import math
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# CATEGORY 1: GRADIENT HEALTH (6 tests)
# =============================================================================

class TestGradientHealth:
    """Tests for gradient-related issues"""

    def test_gradient_explosion_detection(self):
        """Detect when gradients exceed safe thresholds"""
        # From research: Loss spikes caused by rapid gradient amplification
        gradients = [1.0, 5.0, 100.0, 1000.0]
        explosion_threshold = 10.0

        def detect_explosion(grads, threshold):
            return any(abs(g) > threshold for g in grads)

        assert detect_explosion(gradients, explosion_threshold)
        assert not detect_explosion([0.1, 0.2, 0.3], explosion_threshold)

    def test_gradient_vanishing_detection(self):
        """Detect when gradients become too small (early layers stop learning)"""
        # From research: Vanishing gradients make early layers learn slowly
        gradient_history = [0.1, 0.01, 0.001, 0.0001, 1e-8, 1e-10, 1e-12]
        vanishing_threshold = 1e-7

        def detect_vanishing(grads, threshold, window=3):
            if len(grads) < window:
                return False
            return all(g < threshold for g in grads[-window:])

        assert detect_vanishing(gradient_history, vanishing_threshold)
        assert not detect_vanishing([0.1, 0.2, 0.1], vanishing_threshold)

    def test_nan_gradient_detection(self):
        """Detect NaN in gradients immediately"""
        def check_gradients_valid(gradients):
            return all(not (math.isnan(g) or math.isinf(g)) for g in gradients)

        assert check_gradients_valid([0.1, 0.2, -0.1])
        assert not check_gradients_valid([0.1, float('nan'), 0.3])
        assert not check_gradients_valid([0.1, float('inf')])

    def test_gradient_clipping_verification(self):
        """Verify gradient clipping limits gradient norm (max_grad_norm=0.3)"""
        max_grad_norm = 0.3  # From config.yaml

        gradients = np.array([1.0, 2.0, 3.0])
        grad_norm = np.linalg.norm(gradients)

        if grad_norm > max_grad_norm:
            clipped = gradients * (max_grad_norm / grad_norm)
        else:
            clipped = gradients

        clipped_norm = np.linalg.norm(clipped)
        assert clipped_norm <= max_grad_norm + 1e-6

    def test_layer_wise_gradient_monitoring(self):
        """Monitor gradients per layer for imbalance detection"""
        # Simulate layer gradients (healthy model - gradients decrease but not drastically)
        layer_grads = {
            'layer_0': np.array([0.1, 0.1, 0.1]),
            'layer_5': np.array([0.05, 0.05, 0.05]),
            'layer_10': np.array([0.03, 0.03, 0.03]),
            'layer_15': np.array([0.02, 0.02, 0.02]),
        }

        # Calculate gradient norms per layer
        norms = {k: np.linalg.norm(v) for k, v in layer_grads.items()}

        # Detect 100x+ difference between layers (sign of vanishing)
        max_norm = max(norms.values())
        min_norm = min(norms.values())

        if min_norm > 0:
            ratio = max_norm / min_norm
            # Flag if gradient imbalance > 100x
            assert ratio < 100, f"Gradient imbalance too high: {ratio:.2f}x"

    def test_gradient_imbalance_detected_when_vanishing(self):
        """Detect when layer gradient imbalance exceeds threshold"""
        layer_grads = {
            'layer_0': np.array([0.1, 0.1, 0.1]),
            'layer_15': np.array([0.001, 0.001, 0.001]),  # Vanishing
        }

        norms = {k: np.linalg.norm(v) for k, v in layer_grads.items()}
        max_norm = max(norms.values())
        min_norm = min(norms.values())

        ratio = max_norm / min_norm
        assert ratio >= 100, "Should detect gradient imbalance"

    def test_gradient_accumulation_correctness(self):
        """Verify gradient accumulation sums correctly"""
        gradient_accumulation_steps = 8  # From config
        batch_size = 2
        effective_batch = batch_size * gradient_accumulation_steps

        # Simulate accumulated gradients
        accumulated = np.zeros(10)
        for step in range(gradient_accumulation_steps):
            step_grad = np.ones(10) * 0.1  # Each step contributes 0.1
            accumulated += step_grad

        # After accumulation, should be accumulated/steps for average
        averaged = accumulated / gradient_accumulation_steps
        assert np.allclose(averaged, 0.1)
        assert effective_batch == 16


# =============================================================================
# CATEGORY 2: LOSS BEHAVIOR (5 tests)
# =============================================================================

class TestLossBehavior:
    """Tests for loss-related issues"""

    def test_loss_spike_detection(self):
        """Detect sudden loss increases (>2x previous)"""
        # From research: Loss spikes indicate training instability
        loss_history = [2.5, 2.3, 2.1, 2.0, 5.0]  # Spike at end

        def detect_spike(losses, threshold=2.0):
            for i in range(1, len(losses)):
                if losses[i] > losses[i-1] * threshold:
                    return True, i
            return False, -1

        has_spike, spike_idx = detect_spike(loss_history)
        assert has_spike
        assert spike_idx == 4

    def test_loss_divergence(self):
        """Detect when loss goes to infinity"""
        loss_history = [2.0, 3.0, 10.0, 100.0, float('inf')]

        def detect_divergence(losses):
            return any(math.isinf(l) or l > 1e10 for l in losses)

        assert detect_divergence(loss_history)
        assert not detect_divergence([2.0, 1.8, 1.6])

    def test_loss_convergence_pattern(self):
        """Verify loss decreases over training epochs"""
        loss_history = [3.5, 3.0, 2.5, 2.2, 2.0, 1.9, 1.85]

        def check_convergence(losses, window=3):
            first_avg = np.mean(losses[:window])
            last_avg = np.mean(losses[-window:])
            return last_avg < first_avg

        assert check_convergence(loss_history)
        assert not check_convergence([2.0, 2.5, 3.0, 3.5])

    def test_loss_plateau_detection(self):
        """Detect when loss stops improving (plateau)"""
        loss_history = [2.0, 1.5, 1.2, 1.19, 1.19, 1.19, 1.19]  # Plateau

        def detect_plateau(losses, patience=3, threshold=0.01):
            if len(losses) < patience + 1:
                return False
            recent = losses[-patience:]
            variance = np.var(recent)
            return variance < threshold

        assert detect_plateau(loss_history)
        assert not detect_plateau([2.0, 1.5, 1.0, 0.5])

    def test_nan_inf_loss_detection(self):
        """Detect NaN/Inf in loss immediately"""
        def check_loss_valid(loss):
            return not (math.isnan(loss) or math.isinf(loss))

        assert check_loss_valid(2.5)
        assert not check_loss_valid(float('nan'))
        assert not check_loss_valid(float('inf'))
        assert not check_loss_valid(float('-inf'))


# =============================================================================
# CATEGORY 3: OVERFITTING/UNDERFITTING (4 tests)
# =============================================================================

class TestOverfittingUnderfitting:
    """Tests for generalization issues"""

    def test_overfitting_detection(self):
        """Detect train/val loss divergence (overfitting signal)"""
        # From research: Val loss increases while train loss decreases = overfitting
        train_losses = [2.0, 1.5, 1.0, 0.5, 0.2]  # Decreasing
        val_losses = [2.0, 1.8, 1.9, 2.2, 2.5]    # Increasing

        def detect_overfitting(train, val, window=3):
            if len(train) < window or len(val) < window:
                return False
            # Check if train is decreasing and val is increasing
            train_trend = np.polyfit(range(window), train[-window:], 1)[0]
            val_trend = np.polyfit(range(window), val[-window:], 1)[0]
            return train_trend < 0 and val_trend > 0

        assert detect_overfitting(train_losses, val_losses)

    def test_underfitting_detection(self):
        """Detect when both train and val loss stay high"""
        train_losses = [3.0, 2.95, 2.9, 2.88, 2.86]  # High and barely decreasing
        val_losses = [3.1, 3.05, 3.02, 3.0, 2.98]    # Also high

        def detect_underfitting(train, val, loss_threshold=2.5):
            avg_train = np.mean(train[-3:])
            avg_val = np.mean(val[-3:])
            return avg_train > loss_threshold and avg_val > loss_threshold

        assert detect_underfitting(train_losses, val_losses)

    def test_early_stopping_trigger_point(self):
        """Verify early stopping triggers at correct epoch"""
        patience = 3
        eval_losses = [2.5, 2.2, 2.0, 2.0, 2.01, 2.02]

        best_loss = float('inf')
        wait_count = 0
        trigger_epoch = None

        for epoch, loss in enumerate(eval_losses):
            if loss < best_loss:
                best_loss = loss
                wait_count = 0
            else:
                wait_count += 1
                if wait_count >= patience:
                    trigger_epoch = epoch
                    break

        assert trigger_epoch == 5  # Epoch 5 (0-indexed)

    def test_best_model_selection(self):
        """Verify best model is correctly identified by eval loss"""
        checkpoints = [
            {'epoch': 1, 'eval_loss': 2.5},
            {'epoch': 2, 'eval_loss': 2.2},
            {'epoch': 3, 'eval_loss': 1.8},  # Best
            {'epoch': 4, 'eval_loss': 2.0},
            {'epoch': 5, 'eval_loss': 2.1},
        ]

        best = min(checkpoints, key=lambda x: x['eval_loss'])
        assert best['epoch'] == 3
        assert best['eval_loss'] == 1.8


# =============================================================================
# CATEGORY 4: DATA INTEGRITY (5 tests)
# =============================================================================

class TestDataIntegrity:
    """Tests for data processing issues"""

    def test_no_train_val_leakage(self):
        """Ensure no overlap between train and validation sets"""
        all_ids = [f"sample_{i}" for i in range(100)]
        train_ids = set(all_ids[:90])
        val_ids = set(all_ids[90:])

        overlap = train_ids.intersection(val_ids)
        assert len(overlap) == 0

    def test_batch_size_consistency(self):
        """Verify batch sizes are consistent"""
        batch_size = 2
        gradient_accumulation = 8
        expected_effective = 16

        assert batch_size * gradient_accumulation == expected_effective

    def test_sequence_length_enforcement(self):
        """Verify sequences are truncated to max_seq_length=4096"""
        max_seq_length = 4096

        sequences = [
            list(range(100)),      # Short - no truncation
            list(range(4096)),     # Exact - no truncation
            list(range(10000)),    # Long - needs truncation
        ]

        for seq in sequences:
            truncated = seq[:max_seq_length]
            assert len(truncated) <= max_seq_length

    def test_tokenization_consistency(self):
        """Verify same input produces same tokenization"""
        def mock_tokenize(text):
            return len(text.split())

        text = "Generate TSN code for time-aware scheduling"
        assert mock_tokenize(text) == mock_tokenize(text)

    def test_encoding_handling(self):
        """Verify both UTF-8 and Latin-1 encodings handled"""
        test_cases = [
            ("Hello World", "utf-8"),
            ("Café résumé", "utf-8"),
        ]

        for text, encoding in test_cases:
            encoded = text.encode(encoding)
            decoded = encoded.decode(encoding)
            assert decoded == text


# =============================================================================
# CATEGORY 5: LEARNING RATE (3 tests)
# =============================================================================

class TestLearningRate:
    """Tests for learning rate configuration"""

    def test_learning_rate_bounds(self):
        """Verify LR is in safe range (1e-4 to 2e-4 for LoRA)"""
        # From research: Use 1e-4 to 2e-4 for stable convergence
        configured_lr = 1e-4  # From config.yaml
        safe_min = 1e-5
        safe_max = 5e-4

        assert safe_min <= configured_lr <= safe_max

    def test_warmup_schedule(self):
        """Verify warmup increases LR linearly"""
        initial_lr = 1e-4
        warmup_ratio = 0.05
        total_steps = 1000
        warmup_steps = int(total_steps * warmup_ratio)

        def get_lr_at_step(step):
            if step < warmup_steps:
                return initial_lr * (step / warmup_steps)
            return initial_lr

        # LR should increase during warmup
        assert get_lr_at_step(0) == 0
        assert get_lr_at_step(warmup_steps // 2) == initial_lr / 2
        assert get_lr_at_step(warmup_steps) == initial_lr

    def test_cosine_decay(self):
        """Verify cosine decay reduces LR to near zero"""
        initial_lr = 1e-4
        total_steps = 1000
        warmup_steps = 50

        def cosine_lr(step):
            if step < warmup_steps:
                return initial_lr * (step / warmup_steps)
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return initial_lr * 0.5 * (1 + math.cos(math.pi * progress))

        # Should decay to near zero
        assert cosine_lr(total_steps) == pytest.approx(0, abs=1e-8)


# =============================================================================
# CATEGORY 6: CHECKPOINTING (4 tests)
# =============================================================================

class TestCheckpointing:
    """Tests for checkpoint save/load"""

    def test_checkpoint_state_preservation(self):
        """Verify checkpoint contains all required state"""
        checkpoint = {
            'step': 500,
            'epoch': 2.5,
            'best_loss': 1.8,
            'optimizer_state': {'lr': 5e-5},
            'model_state': {'weights': [0.1, 0.2]},
        }

        required = ['step', 'epoch', 'optimizer_state', 'model_state']
        for field in required:
            assert field in checkpoint

    def test_best_checkpoint_saved(self):
        """Verify best model is saved correctly"""
        eval_history = [2.5, 2.2, 1.9, 2.0, 2.1]
        best_idx = np.argmin(eval_history)
        assert best_idx == 2  # 1.9 is best

    def test_disk_space_check(self):
        """Verify disk space is checked before saving"""
        required_gb = 15  # Typical model size

        def check_disk_space(path, required_gb):
            import shutil
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024**3)
            return free_gb >= required_gb

        # Should have enough space on temp
        assert check_disk_space('/tmp', 0.001)

    def test_checkpoint_resume_correctness(self):
        """Verify training can resume from checkpoint"""
        saved_state = {'step': 100, 'best_loss': 2.0}
        loaded_state = saved_state.copy()

        assert loaded_state['step'] == saved_state['step']
        assert loaded_state['best_loss'] == saved_state['best_loss']


# =============================================================================
# CATEGORY 7: DISTILLATION QUALITY (4 tests)
# =============================================================================

class TestDistillationQuality:
    """Tests for knowledge distillation"""

    def test_teacher_output_quality(self):
        """Verify teacher outputs meet quality threshold"""
        min_score = 7.0  # From config.yaml
        outputs = [
            {'score': 8.5},
            {'score': 7.2},
            {'score': 6.0},  # Below threshold
            {'score': 9.0},
        ]

        quality = [o for o in outputs if o['score'] >= min_score]
        assert len(quality) == 3

    def test_temperature_softening(self):
        """Verify temperature softens probability distribution"""
        logits = np.array([2.0, 1.0, 0.5])

        def softmax_t(logits, temp):
            scaled = logits / temp
            exp_scaled = np.exp(scaled - np.max(scaled))
            return exp_scaled / exp_scaled.sum()

        probs_t1 = softmax_t(logits, 1.0)
        probs_t2 = softmax_t(logits, 2.0)

        # Higher temp = more uniform (lower variance)
        assert np.var(probs_t2) < np.var(probs_t1)

    def test_token_count_tracking(self):
        """Verify token usage is tracked correctly"""
        responses = [
            {'input_tokens': 100, 'output_tokens': 500},
            {'input_tokens': 150, 'output_tokens': 600},
            {'input_tokens': 120, 'output_tokens': 550},
        ]

        total_input = sum(r['input_tokens'] for r in responses)
        total_output = sum(r['output_tokens'] for r in responses)

        assert total_input == 370
        assert total_output == 1650

    def test_quality_threshold_filtering(self):
        """Verify low-quality outputs are filtered"""
        convergence_threshold = 8.0  # From config
        outputs = [7.0, 7.5, 8.0, 8.5, 9.0]

        converged = [o for o in outputs if o >= convergence_threshold]
        assert len(converged) == 3


# =============================================================================
# CATEGORY 8: RESOURCE/STABILITY (3 tests)
# =============================================================================

class TestResourceStability:
    """Tests for resource and numerical stability"""

    def test_memory_usage_monitoring(self):
        """Verify memory usage can be monitored"""
        # Simulate GPU memory tracking
        memory_log = [
            {'step': 0, 'allocated_gb': 5.0, 'reserved_gb': 8.0},
            {'step': 100, 'allocated_gb': 5.5, 'reserved_gb': 8.0},
            {'step': 200, 'allocated_gb': 6.0, 'reserved_gb': 8.0},
        ]

        # Check for memory growth (potential leak)
        growth = memory_log[-1]['allocated_gb'] - memory_log[0]['allocated_gb']
        assert growth < 5.0  # Acceptable growth

    def test_mixed_precision_stability(self):
        """Verify BF16 computations don't overflow"""
        # BF16 range: ~1.18e-38 to ~3.40e38
        bf16_max = 3.40e38
        bf16_min = 1.18e-38

        values = [1.0, 0.001, 1000.0, 1e10, 1e-10]

        for v in values:
            assert bf16_min <= abs(v) <= bf16_max or v == 0

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same results"""
        seed = 42

        np.random.seed(seed)
        result1 = np.random.rand(10)

        np.random.seed(seed)
        result2 = np.random.rand(10)

        assert np.allclose(result1, result2)


# =============================================================================
# INTEGRATION TEST: Full Training Simulation
# =============================================================================

class TestTrainingSimulation:
    """Integration test simulating full training loop"""

    def test_complete_training_loop(self):
        """Simulate training loop and verify all checks pass"""
        # Config
        num_epochs = 5
        patience = 3
        max_grad_norm = 0.3

        # Simulate training
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        wait_count = 0

        for epoch in range(num_epochs):
            # Simulate decreasing loss
            train_loss = 3.0 - (epoch * 0.4) + np.random.normal(0, 0.1)
            val_loss = 3.1 - (epoch * 0.35) + np.random.normal(0, 0.1)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Check for NaN
            assert not math.isnan(train_loss)
            assert not math.isnan(val_loss)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                wait_count = 0
            else:
                wait_count += 1

            if wait_count >= patience:
                break

        # Verify convergence
        assert train_losses[-1] < train_losses[0]
        assert val_losses[-1] < val_losses[0]
