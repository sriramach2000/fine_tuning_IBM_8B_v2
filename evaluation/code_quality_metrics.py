#!/usr/bin/env python3
"""
Code Quality Metrics for Student Output Evaluation

Evaluates generated code on multiple dimensions:
1. Syntax correctness (compilation)
2. Protocol compliance (TSN/AVB)
3. Safety guidelines (MISRA-C)
4. Code style and documentation

Used in iterative distillation to identify outputs
that need teacher correction.

Author: Sriram Acharya
Organization: Excelfore
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class QualityScore:
    """
    Composite quality score for generated code.

    Attributes:
        syntax_score: 0-10, does code compile?
        protocol_score: 0-10, TSN/AVB compliance
        safety_score: 0-10, MISRA-C adherence
        style_score: 0-10, code quality and documentation
    """
    syntax_score: float
    protocol_score: float
    safety_score: float
    style_score: float

    @property
    def overall(self) -> float:
        """
        Calculate weighted average of all scores.

        Weights:
        - Syntax: 30% (critical - must compile)
        - Protocol: 30% (critical - must be correct)
        - Safety: 25% (important for automotive)
        - Style: 15% (nice to have)
        """
        weights = {
            'syntax': 0.30,
            'protocol': 0.30,
            'safety': 0.25,
            'style': 0.15,
        }
        return (
            self.syntax_score * weights['syntax'] +
            self.protocol_score * weights['protocol'] +
            self.safety_score * weights['safety'] +
            self.style_score * weights['style']
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'syntax_score': self.syntax_score,
            'protocol_score': self.protocol_score,
            'safety_score': self.safety_score,
            'style_score': self.style_score,
            'overall': self.overall,
        }

    def __str__(self) -> str:
        return (
            f"QualityScore(syntax={self.syntax_score:.1f}, "
            f"protocol={self.protocol_score:.1f}, "
            f"safety={self.safety_score:.1f}, "
            f"style={self.style_score:.1f}, "
            f"overall={self.overall:.2f})"
        )


class CodeQualityEvaluator:
    """
    Evaluate quality of generated automotive code.

    Used in iterative distillation to identify outputs
    that need teacher correction.

    Quality threshold: Outputs scoring below threshold
    are sent to the teacher model for correction.
    """

    # TSN/AVB protocol keywords
    TSN_KEYWORDS = [
        'pcp', 'vlan', 'timestamp', 'gate', 'priority',
        'qbv', 'qav', 'shaper', 'gcl', 'schedule'
    ]
    AVB_KEYWORDS = [
        'stream', 'sample', 'channel', 'bandwidth', 'srp',
        'talker', 'listener', 'reservation', 'class'
    ]

    # MISRA-C violation patterns
    MISRA_VIOLATIONS = [
        ('goto ', -3.0, "MISRA 15.1: goto prohibited"),
        ('malloc(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
        ('free(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
        ('realloc(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
        ('calloc(', -2.0, "MISRA 21.3: dynamic memory prohibited"),
        ('setjmp', -3.0, "MISRA 21.4: setjmp/longjmp prohibited"),
        ('longjmp', -3.0, "MISRA 21.4: setjmp/longjmp prohibited"),
        ('abort(', -1.5, "MISRA 21.8: abort prohibited"),
        ('exit(', -1.5, "MISRA 21.8: exit prohibited"),
    ]

    # Good practices for automotive code
    GOOD_PRACTICES = [
        ('uint8_t', 0.5, "Fixed-width integer type"),
        ('uint16_t', 0.5, "Fixed-width integer type"),
        ('uint32_t', 0.5, "Fixed-width integer type"),
        ('uint64_t', 0.5, "Fixed-width integer type"),
        ('int8_t', 0.5, "Fixed-width integer type"),
        ('int16_t', 0.5, "Fixed-width integer type"),
        ('int32_t', 0.5, "Fixed-width integer type"),
        ('static ', 0.3, "Static linkage"),
        ('const ', 0.3, "Const correctness"),
        ('volatile ', 0.2, "Hardware interface awareness"),
    ]

    def __init__(
        self,
        strict_mode: bool = True,
        quality_threshold: float = 7.0,
        gcc_available: bool = True
    ):
        """
        Initialize the evaluator.

        Args:
            strict_mode: If True, apply stricter checks
            quality_threshold: Minimum score to pass without correction
            gcc_available: If False, skip compilation check
        """
        self.strict_mode = strict_mode
        self.quality_threshold = quality_threshold
        self.gcc_available = gcc_available and self._check_gcc()

    def _check_gcc(self) -> bool:
        """Check if gcc is available for syntax checking"""
        try:
            result = subprocess.run(
                ['gcc', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def evaluate(self, code: str, prompt: str) -> QualityScore:
        """
        Evaluate a single code sample.

        Args:
            code: Generated code string
            prompt: Original prompt (for context)

        Returns:
            QualityScore with breakdown of all metrics
        """
        # Extract code from markdown if present
        extracted_code = self._extract_code_block(code)

        return QualityScore(
            syntax_score=self._check_syntax(extracted_code),
            protocol_score=self._check_protocol_compliance(extracted_code, prompt),
            safety_score=self._check_misra_compliance(extracted_code),
            style_score=self._check_style(extracted_code)
        )

    def evaluate_batch(
        self,
        outputs: List[str],
        prompts: List[Dict]
    ) -> List[Tuple[float, QualityScore]]:
        """
        Evaluate multiple outputs.

        Args:
            outputs: List of generated code strings
            prompts: List of prompt dictionaries with 'prompt' key

        Returns:
            List of tuples (overall_score, full_quality_score)
        """
        results = []
        for output, prompt_data in zip(outputs, prompts):
            prompt_text = prompt_data.get('prompt', '') if isinstance(prompt_data, dict) else str(prompt_data)
            quality = self.evaluate(output, prompt_text)
            results.append((quality.overall, quality))
        return results

    def needs_correction(self, score: float) -> bool:
        """
        Check if a score is below the threshold and needs teacher correction.

        Args:
            score: Overall quality score

        Returns:
            True if score is below threshold
        """
        return score < self.quality_threshold

    def filter_for_correction(
        self,
        outputs: List[str],
        prompts: List[Dict],
        scores: List[float]
    ) -> List[Tuple[Dict, str, float]]:
        """
        Filter outputs that need teacher correction.

        Returns list of (prompt, output, score) tuples for outputs
        that scored below the quality threshold.
        """
        poor_outputs = []
        for prompt, output, score in zip(prompts, outputs, scores):
            if self.needs_correction(score):
                poor_outputs.append((prompt, output, score))
        return poor_outputs

    def _check_syntax(self, code: str) -> float:
        """
        Check if code compiles.

        Returns:
            10.0 if compiles successfully
            5.0-9.0 for warnings
            0.0-5.0 for errors
        """
        if not code.strip():
            return 0.0

        if not self.gcc_available:
            # Fallback: basic syntax heuristics
            return self._heuristic_syntax_check(code)

        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.c',
                delete=False
            ) as f:
                # Add minimal includes for common types
                wrapped_code = self._wrap_code_for_compilation(code)
                f.write(wrapped_code)
                f.flush()
                temp_path = f.name

            try:
                result = subprocess.run(
                    [
                        'gcc', '-fsyntax-only', '-std=c99',
                        '-Wall', '-Wextra',
                        '-Wno-unused-variable',
                        '-Wno-unused-parameter',
                        temp_path
                    ],
                    capture_output=True,
                    timeout=10
                )

                stderr = result.stderr.decode('utf-8', errors='ignore')

                if result.returncode == 0:
                    # Check for warnings
                    warnings = stderr.count('warning:')
                    if warnings == 0:
                        return 10.0
                    else:
                        return max(7.0, 9.0 - warnings * 0.5)
                else:
                    # Count errors
                    errors = stderr.count('error:')
                    if errors == 0:
                        # Only warnings, but non-zero return
                        return 6.0
                    else:
                        return max(0.0, 5.0 - errors * 1.0)

            finally:
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            return 5.0  # Unknown - neutral score
        except Exception:
            return 5.0

    def _heuristic_syntax_check(self, code: str) -> float:
        """
        Basic syntax check using heuristics when gcc is unavailable.
        """
        score = 7.0

        # Check for balanced braces
        if code.count('{') != code.count('}'):
            score -= 2.0

        # Check for balanced parentheses
        if code.count('(') != code.count(')'):
            score -= 2.0

        # Check for semicolons at end of statements
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('{', '}', '//', '/*', '*/', ',')):
                if not line.endswith(';') and not line.startswith(('#', 'if', 'else', 'for', 'while', 'switch')):
                    score -= 0.3

        return max(0.0, min(10.0, score))

    def _wrap_code_for_compilation(self, code: str) -> str:
        """Wrap code with common includes for compilation test"""
        includes = """
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

// Placeholder for automotive types
#ifndef ETH_ALEN
#define ETH_ALEN 6
#endif

"""
        # Check if code already has includes
        if '#include' in code[:500]:
            return code
        return includes + code

    def _check_protocol_compliance(self, code: str, prompt: str) -> float:
        """
        Check TSN/AVB protocol compliance.

        Looks for required protocol elements based on prompt context.
        """
        score = 10.0
        code_lower = code.lower()
        prompt_lower = prompt.lower()

        # Determine which protocols are expected
        expects_tsn = any(kw in prompt_lower for kw in ['tsn', '802.1q', 'qbv', 'qav', '802.1as', 'time-aware', 'shaper'])
        expects_avb = any(kw in prompt_lower for kw in ['avb', 'audio', 'video', 'stream reservation', 'srp', 'listener', 'talker'])

        if expects_tsn:
            # Check for TSN-specific elements
            found_keywords = sum(1 for kw in self.TSN_KEYWORDS if kw in code_lower)
            expected_min = 3  # Should have at least 3 TSN keywords
            if found_keywords < expected_min:
                score -= (expected_min - found_keywords) * 1.5

        if expects_avb:
            # Check for AVB-specific elements
            found_keywords = sum(1 for kw in self.AVB_KEYWORDS if kw in code_lower)
            expected_min = 3
            if found_keywords < expected_min:
                score -= (expected_min - found_keywords) * 1.5

        # Check for common automotive patterns
        if 'ethernet' in prompt_lower or 'eth' in prompt_lower:
            if 'eth_hdr' not in code_lower and 'ethhdr' not in code_lower and 'ethernet_header' not in code_lower:
                score -= 1.0

        if 'vlan' in prompt_lower:
            if 'vlan_id' not in code_lower and 'vid' not in code_lower:
                score -= 1.5

        return max(0.0, score)

    def _check_misra_compliance(self, code: str) -> float:
        """
        Check MISRA-C guideline adherence.

        Penalizes violations and rewards good practices.
        """
        score = 10.0

        # Check for violations
        for pattern, penalty, _ in self.MISRA_VIOLATIONS:
            if pattern in code:
                score += penalty  # penalty is negative

        # Bonus for good practices (capped contribution)
        bonus = 0.0
        for pattern, value, _ in self.GOOD_PRACTICES:
            if pattern in code:
                bonus += value

        score = min(10.0, score + min(bonus, 2.0))

        # Check for recursion (function calls itself within its body)
        functions = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', code)
        for func in functions:
            func_pattern = func + '('
            func_start = code.find(func_pattern)
            if func_start != -1:
                # Look for the function calling itself after its definition
                func_body = code[func_start + len(func_pattern):]
                if func_body.count(func_pattern) > 0:
                    score -= 1.5  # Possible recursion

        # Check for unbounded loops
        if 'while(1)' in code.replace(' ', '') or 'while (1)' in code:
            if 'break' not in code:
                score -= 2.0  # Infinite loop without break

        return max(0.0, score)

    def _check_style(self, code: str) -> float:
        """
        Check code style and documentation quality.
        """
        score = 6.0  # Start at average

        # Check for comments
        single_comments = code.count('//')
        block_comments = code.count('/*')
        total_comments = single_comments + block_comments

        if total_comments >= 5:
            score += 2.0
        elif total_comments >= 2:
            score += 1.0
        elif total_comments == 0:
            score -= 1.0

        # Check for function documentation (doxygen style)
        if '/**' in code:
            score += 1.0
        if '@param' in code or '@return' in code:
            score += 0.5
        if '@brief' in code:
            score += 0.5

        # Check for meaningful variable names (penalize single letters)
        single_letter_vars = len(re.findall(r'\b[a-z]\s*[=;]', code))
        if single_letter_vars > 5:
            score -= 1.0

        # Check for consistent indentation
        lines = code.split('\n')
        indent_types = set()
        for line in lines:
            if line and not line.lstrip() == line:
                leading = len(line) - len(line.lstrip())
                if line.startswith('\t'):
                    indent_types.add('tab')
                elif leading > 0:
                    indent_types.add('space')

        if len(indent_types) > 1:
            score -= 0.5  # Mixed indentation

        return max(0.0, min(10.0, score))

    def _extract_code_block(self, text: str) -> str:
        """
        Extract code from markdown code blocks.

        Handles ```c, ```cpp, ``` blocks.
        """
        # Look for fenced code blocks
        patterns = [
            r'```(?:c|cpp|C|C\+\+)?\n(.*?)```',
            r'~~~(?:c|cpp)?\n(.*?)~~~',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return '\n'.join(matches)

        # No code blocks found, return as-is
        return text


def create_evaluator(
    quality_threshold: float = 7.0,
    strict: bool = True
) -> CodeQualityEvaluator:
    """
    Factory function to create a configured evaluator.

    Args:
        quality_threshold: Minimum score to pass without correction
        strict: Enable strict MISRA checking

    Returns:
        Configured CodeQualityEvaluator instance
    """
    return CodeQualityEvaluator(
        strict_mode=strict,
        quality_threshold=quality_threshold
    )
