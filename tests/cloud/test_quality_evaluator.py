"""
Quality Evaluator Test Suite

Comprehensive tests for CodeQualityEvaluator including:
- Scoring validation
- Edge cases (empty, malformed, unicode)
- Score range enforcement
- MISRA compliance checking
- Recursion detection (including the bug fix validation)
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# EVALUATOR INITIALIZATION
# =============================================================================

@pytest.mark.offline
class TestEvaluatorInitialization:
    """Tests for evaluator initialization"""

    @pytest.fixture
    def evaluator_class(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator

    def test_default_initialization(self, evaluator_class):
        """Test default evaluator initialization"""
        evaluator = evaluator_class()

        assert evaluator.quality_threshold == 7.0
        assert evaluator.strict_mode is True

    def test_custom_threshold(self, evaluator_class):
        """Test custom quality threshold"""
        evaluator = evaluator_class(quality_threshold=8.5)
        assert evaluator.quality_threshold == 8.5

    def test_gcc_disabled(self, evaluator_class):
        """Test with GCC explicitly disabled"""
        evaluator = evaluator_class(gcc_available=False)
        assert evaluator.gcc_available is False


# =============================================================================
# SCORE RANGE VALIDATION
# =============================================================================

@pytest.mark.offline
class TestScoreRangeValidation:
    """Tests for score range enforcement"""

    @pytest.fixture
    def evaluator(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator(gcc_available=False)

    def test_syntax_score_range(self, evaluator, sample_automotive_code):
        """Test syntax score is within 0-10"""
        for code_type, code in sample_automotive_code.items():
            score = evaluator.evaluate(code, "Generate code")
            assert 0.0 <= score.syntax_score <= 10.0, \
                f"Syntax score out of range for {code_type}"

    def test_protocol_score_range(self, evaluator, sample_automotive_code):
        """Test protocol score is within 0-10"""
        for code_type, code in sample_automotive_code.items():
            score = evaluator.evaluate(code, "Generate TSN code")
            assert 0.0 <= score.protocol_score <= 10.0, \
                f"Protocol score out of range for {code_type}"

    def test_safety_score_range(self, evaluator, sample_automotive_code):
        """Test safety score is within 0-10"""
        for code_type, code in sample_automotive_code.items():
            score = evaluator.evaluate(code, "Generate code")
            assert 0.0 <= score.safety_score <= 10.0, \
                f"Safety score out of range for {code_type}"

    def test_style_score_range(self, evaluator, sample_automotive_code):
        """Test style score is within 0-10"""
        for code_type, code in sample_automotive_code.items():
            score = evaluator.evaluate(code, "Generate code")
            assert 0.0 <= score.style_score <= 10.0, \
                f"Style score out of range for {code_type}"

    def test_overall_score_range(self, evaluator, sample_automotive_code):
        """Test overall score is within 0-10"""
        for code_type, code in sample_automotive_code.items():
            score = evaluator.evaluate(code, "Generate code")
            assert 0.0 <= score.overall <= 10.0, \
                f"Overall score out of range for {code_type}"


# =============================================================================
# EDGE CASES
# =============================================================================

@pytest.mark.offline
class TestEvaluatorEdgeCases:
    """Edge case tests for evaluator"""

    @pytest.fixture
    def evaluator(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator(gcc_available=False)

    def test_empty_code(self, evaluator):
        """Test handling of empty code"""
        score = evaluator.evaluate("", "Generate code")
        assert score.syntax_score == 0.0  # Empty code should fail
        assert score.overall >= 0.0  # Overall should still be valid

    def test_whitespace_only_code(self, evaluator):
        """Test handling of whitespace-only code"""
        score = evaluator.evaluate("   \n\t\n   ", "Generate code")
        assert score.syntax_score == 0.0

    def test_unicode_code(self, evaluator, sample_automotive_code):
        """Test handling of unicode in code"""
        score = evaluator.evaluate(
            sample_automotive_code['unicode_code'],
            "Generate code"
        )
        # Should not crash and return valid scores
        assert score.overall >= 0.0

    def test_very_long_code(self, evaluator, sample_automotive_code):
        """Test handling of very long code"""
        score = evaluator.evaluate(
            sample_automotive_code['very_long_code'],
            "Generate code"
        )
        # Should complete without timeout and return valid scores
        assert score.overall >= 0.0

    def test_binary_data_in_code(self, evaluator):
        """Test handling of binary data"""
        binary_code = b'\x00\x01\x02\x03'.decode('latin-1')
        score = evaluator.evaluate(binary_code, "Generate code")
        # Should not crash
        assert score.overall >= 0.0

    def test_code_with_markdown_blocks(self, evaluator):
        """Test extraction of code from markdown blocks"""
        markdown_code = '''
Here is the code:

```c
int main() {
    return 0;
}
```

That's all.
'''
        score = evaluator.evaluate(markdown_code, "Generate code")
        # Should extract and evaluate the C code
        assert score.overall >= 0.0


# =============================================================================
# RECURSION DETECTION (Bug Fix Validation)
# =============================================================================

@pytest.mark.offline
class TestRecursionDetection:
    """Tests for recursion detection - validates bug fix"""

    @pytest.fixture
    def evaluator(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator(gcc_available=False)

    def test_recursive_function_detection(self, evaluator, sample_automotive_code):
        """Test that recursive functions are detected"""
        score = evaluator.evaluate(
            sample_automotive_code['code_with_recursion'],
            "Generate code"
        )
        # Recursion should reduce safety score
        # This test validates the bug fix works
        assert score.safety_score < 10.0

    def test_non_recursive_function(self, evaluator, sample_automotive_code):
        """Test that non-recursive functions are not penalized"""
        score = evaluator.evaluate(
            sample_automotive_code['good_tsn_code'],
            "Generate TSN code"
        )
        # Good code should not be penalized for recursion
        assert score.safety_score >= 7.0

    def test_recursion_detection_no_crash(self, evaluator):
        """Test that recursion detection does not crash (validates bug fix)"""
        # This specifically tests the TypeError bug fix
        test_codes = [
            "int foo() { return foo(); }",  # Direct recursion
            "void bar() { }",  # No recursion
            "",  # Empty
            "int main() { return 0; }",  # No recursion
        ]

        for code in test_codes:
            # Should not raise TypeError
            score = evaluator.evaluate(code, "Generate code")
            assert score is not None


# =============================================================================
# MISRA COMPLIANCE
# =============================================================================

@pytest.mark.offline
class TestMISRACompliance:
    """Tests for MISRA-C compliance checking"""

    @pytest.fixture
    def evaluator(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator(gcc_available=False)

    def test_goto_penalized(self, evaluator, sample_automotive_code):
        """Test that goto statements are penalized"""
        score = evaluator.evaluate(
            sample_automotive_code['bad_code_with_goto'],
            "Generate code"
        )
        assert score.safety_score < 10.0

    def test_malloc_penalized(self, evaluator):
        """Test that malloc is penalized"""
        code = "void* ptr = malloc(100);"
        score = evaluator.evaluate(code, "Generate code")
        assert score.safety_score < 10.0

    def test_fixed_width_integers_rewarded(self, evaluator):
        """Test that fixed-width integers are rewarded"""
        code = """
uint8_t byte_val;
uint16_t short_val;
uint32_t int_val;
"""
        score = evaluator.evaluate(code, "Generate code")
        # Should get bonus for fixed-width types
        assert score.safety_score >= 7.0

    @pytest.mark.parametrize("violation,expected_penalty", [
        ("goto error;", True),
        ("malloc(100)", True),
        ("free(ptr)", True),
        ("exit(1)", True),
        ("abort()", True),
    ])
    def test_misra_violations(self, evaluator, violation, expected_penalty):
        """Parameterized test for MISRA violations"""
        code = f"void func() {{ {violation} }}"
        score = evaluator.evaluate(code, "Generate code")

        if expected_penalty:
            assert score.safety_score < 10.0


# =============================================================================
# SCORING CONSISTENCY
# =============================================================================

@pytest.mark.offline
class TestScoringConsistency:
    """Tests for scoring consistency and determinism"""

    @pytest.fixture
    def evaluator(self):
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        return CodeQualityEvaluator(gcc_available=False)

    def test_same_code_same_score(self, evaluator, sample_automotive_code):
        """Test that same code produces same score"""
        code = sample_automotive_code['good_tsn_code']

        score1 = evaluator.evaluate(code, "Generate TSN code")
        score2 = evaluator.evaluate(code, "Generate TSN code")

        assert score1.overall == score2.overall
        assert score1.syntax_score == score2.syntax_score
        assert score1.safety_score == score2.safety_score

    def test_good_code_beats_bad_code(self, evaluator, sample_automotive_code):
        """Test that good code scores higher than bad code"""
        good_score = evaluator.evaluate(
            sample_automotive_code['good_tsn_code'],
            "Generate TSN code"
        )
        bad_score = evaluator.evaluate(
            sample_automotive_code['bad_code_with_goto'],
            "Generate TSN code"
        )

        assert good_score.overall > bad_score.overall
