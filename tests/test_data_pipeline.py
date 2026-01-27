"""
Tests for Data Pipeline (S3 and Data Processing)

Tests:
1. test_empty_s3_bucket - Handle no files found
2. test_malformed_code_file - Handle encoding errors
3. test_empty_code_extraction - Handle files with no functions
4. test_prompt_generation - Verify prompt structure
5. test_jsonl_output_format - Verify train/val split format
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPipeline:
    """Test suite for automotive data pipeline"""

    @pytest.fixture
    def pipeline_class(self):
        """Import the pipeline class"""
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        return AutomotiveDataPipeline

    # =========================================================================
    # Test 1: Empty S3 Bucket
    # =========================================================================
    def test_empty_s3_bucket(self, pipeline_class, mock_env_vars):
        """Handle no files found in S3 gracefully"""
        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            # Return empty bucket
            mock_client.list_objects_v2.return_value = {
                'Contents': [],
                'KeyCount': 0
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Should return empty list, not crash
                files = pipeline.list_s3_objects('tsn_data/')
                assert files == []

    # =========================================================================
    # Test 2: Malformed Code File (Encoding Errors)
    # =========================================================================
    def test_malformed_code_file(self, pipeline_class, mock_env_vars):
        """Handle encoding errors with fallback"""
        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Create file with mixed encodings
                test_file = Path(tmpdir) / 'mixed_encoding.c'

                # Write bytes that are valid latin-1 but invalid utf-8
                with open(test_file, 'wb') as f:
                    f.write(b'// Comment with \xe9\xe8\xe0\n')  # latin-1 accents
                    f.write(b'int main() { return 0; }\n')

                # Should read successfully with fallback encoding
                content = pipeline.read_code_file(str(test_file))
                assert content is not None
                assert 'int main()' in content

    # =========================================================================
    # Test 3: Empty Code Extraction
    # =========================================================================
    def test_empty_code_extraction(self, pipeline_class, mock_env_vars):
        """Handle files with no extractable functions"""
        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Create file with only comments and includes
                test_file = Path(tmpdir) / 'no_functions.c'
                test_file.write_text('''
// This is a header file with no functions
#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 100

// End of file
''')

                # Read file
                content = pipeline.read_code_file(str(test_file))
                assert content is not None

                # Extract functions (should find none or minimal)
                functions = pipeline.extract_functions(content)
                # Empty files should not crash, may return empty list
                assert isinstance(functions, list)

    # =========================================================================
    # Test 4: Prompt Generation
    # =========================================================================
    def test_prompt_generation(self, pipeline_class, mock_env_vars):
        """Verify prompt structure matches expected format"""
        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Sample code for prompt generation
                sample_code = '''
int tsn_frame_init(tsn_frame_t *frame) {
    if (!frame) return -1;
    frame->priority = 0;
    frame->vlan_id = 0;
    return 0;
}
'''

                prompts = pipeline.generate_prompts_from_code(sample_code, 'tsn_frame.c')

                # Verify prompt structure
                assert len(prompts) > 0
                for prompt in prompts:
                    assert 'id' in prompt
                    assert 'prompt' in prompt
                    assert isinstance(prompt['id'], str)
                    assert isinstance(prompt['prompt'], str)
                    assert len(prompt['prompt']) > 10  # Non-empty prompt

    # =========================================================================
    # Test 5: JSONL Output Format
    # =========================================================================
    def test_jsonl_output_format(self, pipeline_class, mock_env_vars):
        """Verify train/val split format is valid JSONL"""
        with patch('scripts.prepare_automotive_data.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline = pipeline_class(
                    bucket_name='test-bucket',
                    output_dir=tmpdir
                )

                # Create sample prompts
                prompts = [
                    {'id': f'test_{i}', 'prompt': f'Generate code for test {i}'}
                    for i in range(10)
                ]

                # Save to JSONL
                output_file = Path(tmpdir) / 'test_prompts.jsonl'
                pipeline.save_to_jsonl(prompts, str(output_file))

                # Verify file exists and is valid JSONL
                assert output_file.exists()

                # Read and verify each line is valid JSON
                with open(output_file, 'r') as f:
                    lines = f.readlines()

                assert len(lines) == 10

                for line in lines:
                    data = json.loads(line.strip())
                    assert 'id' in data
                    assert 'prompt' in data


class TestDataValidation:
    """Test data validation functions"""

    @pytest.fixture
    def validation_functions(self):
        """Import validation functions"""
        from training.train_granite_qlora import validate_datasets
        return validate_datasets

    def test_validate_empty_dataset(self, validation_functions):
        """Validation should fail on empty dataset"""
        from datasets import Dataset

        empty_train = Dataset.from_dict({'text': []})
        empty_val = Dataset.from_dict({'text': []})

        # Should raise or return False for empty datasets
        with pytest.raises(Exception):
            validation_functions(empty_train, empty_val)

    def test_validate_missing_fields(self, validation_functions):
        """Validation should fail on missing required fields"""
        from datasets import Dataset

        # Dataset missing 'text' field
        bad_train = Dataset.from_dict({'wrong_field': ['data']})
        bad_val = Dataset.from_dict({'wrong_field': ['data']})

        with pytest.raises(Exception):
            validation_functions(bad_train, bad_val)

    def test_validate_good_dataset(self, validation_functions):
        """Validation should pass on good dataset"""
        from datasets import Dataset

        good_train = Dataset.from_dict({
            'text': ['Sample training text 1', 'Sample training text 2'] * 10
        })
        good_val = Dataset.from_dict({
            'text': ['Sample validation text'] * 5
        })

        # Should not raise
        try:
            validation_functions(good_train, good_val)
        except Exception as e:
            pytest.fail(f"Validation should pass but raised: {e}")
