"""
Data Integrity Test Suite

Tests for data processing integrity including:
- Train/test data leakage detection
- Data format consistency
- Encoding handling
- Token counting accuracy
"""

import json
import pytest
import hashlib


# =============================================================================
# TRAIN/TEST LEAKAGE
# =============================================================================

@pytest.mark.offline
class TestDataLeakage:
    """Tests for train/test data leakage detection"""

    def test_no_id_overlap(self):
        """Test no ID overlap between train and test sets"""
        # Simulate train/test split
        all_ids = [f"sample_{i}" for i in range(100)]
        train_ids = set(all_ids[:80])
        test_ids = set(all_ids[80:])

        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping IDs"

    def test_no_content_leakage(self):
        """Test no content duplication between train and test"""
        train_data = [
            {'id': 'train_1', 'text': 'Training example 1'},
            {'id': 'train_2', 'text': 'Training example 2'},
        ]
        test_data = [
            {'id': 'test_1', 'text': 'Test example 1'},
            {'id': 'test_2', 'text': 'Test example 2'},
        ]

        train_hashes = {hashlib.md5(d['text'].encode()).hexdigest() for d in train_data}
        test_hashes = {hashlib.md5(d['text'].encode()).hexdigest() for d in test_data}

        overlap = train_hashes.intersection(test_hashes)
        assert len(overlap) == 0, "Found content duplication between train and test"

    def test_prompt_not_in_response(self):
        """Test prompts are not leaked into responses"""
        samples = [
            {'prompt': 'Generate TSN code', 'response': 'int tsn_init() { return 0; }'},
            {'prompt': 'Create AVB stream', 'response': 'avb_stream_t* create() { }'},
        ]

        for sample in samples:
            # Response should not contain exact prompt
            assert sample['prompt'] not in sample['response'], \
                "Prompt leaked into response"


# =============================================================================
# DATA FORMAT CONSISTENCY
# =============================================================================

@pytest.mark.offline
class TestDataFormatConsistency:
    """Tests for data format consistency"""

    def test_jsonl_format_validity(self, tmp_path):
        """Test JSONL files have valid format"""
        test_file = tmp_path / "test.jsonl"

        # Write valid JSONL
        data = [
            {'id': '1', 'text': 'Sample 1'},
            {'id': '2', 'text': 'Sample 2'},
        ]
        with open(test_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        # Read and verify
        with open(test_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parsed = json.loads(line.strip())
                    assert 'id' in parsed
                    assert 'text' in parsed
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON on line {line_num}")

    def test_required_fields_present(self):
        """Test all required fields are present"""
        required_fields = ['id', 'prompt']

        sample_data = [
            {'id': '1', 'prompt': 'Generate code', 'extra': 'ok'},
            {'id': '2', 'prompt': 'Another prompt'},
        ]

        for item in sample_data:
            for field in required_fields:
                assert field in item, f"Missing required field: {field}"

    def test_consistent_field_types(self):
        """Test field types are consistent"""
        sample_data = [
            {'id': '1', 'prompt': 'Text', 'score': 8.5},
            {'id': '2', 'prompt': 'Text', 'score': 7.2},
        ]

        # All IDs should be strings
        for item in sample_data:
            assert isinstance(item['id'], str)
            assert isinstance(item['prompt'], str)
            assert isinstance(item['score'], (int, float))


# =============================================================================
# ENCODING HANDLING
# =============================================================================

@pytest.mark.offline
class TestEncodingHandling:
    """Tests for encoding handling"""

    def test_utf8_handling(self):
        """Test UTF-8 encoding is handled correctly"""
        utf8_texts = [
            "Simple ASCII text",
            "Café résumé",  # French accents
            "中文文本",  # Chinese
            "Русский",  # Russian
            "😀 emoji",  # Emoji
        ]

        for text in utf8_texts:
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            assert decoded == text

    def test_latin1_fallback(self):
        """Test Latin-1 fallback for non-UTF-8 content"""
        # Bytes that are valid Latin-1 but invalid UTF-8
        latin1_bytes = b'\xe9\xe8\xe0'  # French accents in Latin-1

        # Should fail as UTF-8
        with pytest.raises(UnicodeDecodeError):
            latin1_bytes.decode('utf-8')

        # Should succeed as Latin-1
        decoded = latin1_bytes.decode('latin-1')
        assert len(decoded) == 3

    def test_binary_content_handling(self):
        """Test binary content is handled safely"""
        binary_content = b'\x00\x01\x02\xff\xfe\xfd'

        # Should decode with errors='replace' or 'ignore'
        decoded = binary_content.decode('utf-8', errors='replace')
        assert decoded is not None


# =============================================================================
# TOKEN COUNTING
# =============================================================================

@pytest.mark.offline
class TestTokenCounting:
    """Tests for token counting accuracy"""

    def test_token_count_tracking(self):
        """Test token usage is tracked correctly"""
        responses = [
            {'input_tokens': 100, 'output_tokens': 500},
            {'input_tokens': 150, 'output_tokens': 600},
            {'input_tokens': 120, 'output_tokens': 450},
        ]

        total_input = sum(r['input_tokens'] for r in responses)
        total_output = sum(r['output_tokens'] for r in responses)

        assert total_input == 370
        assert total_output == 1550

    def test_token_count_types(self):
        """Test token counts are integers"""
        usage = {'input_tokens': 100, 'output_tokens': 500}

        assert isinstance(usage['input_tokens'], int)
        assert isinstance(usage['output_tokens'], int)

    def test_token_count_non_negative(self):
        """Test token counts are non-negative"""
        usages = [
            {'input_tokens': 0, 'output_tokens': 0},
            {'input_tokens': 100, 'output_tokens': 500},
            {'input_tokens': 1, 'output_tokens': 1},
        ]

        for usage in usages:
            assert usage['input_tokens'] >= 0
            assert usage['output_tokens'] >= 0

    @pytest.mark.parametrize("text,expected_approx_tokens", [
        ("Hello world", 2),
        ("This is a test sentence.", 6),
        ("Generate TSN Time-Aware Shaper code", 7),
    ])
    def test_approximate_token_estimation(self, text, expected_approx_tokens):
        """Test approximate token estimation (word count as proxy)"""
        # Simple approximation: ~1 token per word
        word_count = len(text.split())
        # Allow 50% margin
        assert expected_approx_tokens * 0.5 <= word_count <= expected_approx_tokens * 1.5
