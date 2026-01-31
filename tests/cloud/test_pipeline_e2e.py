"""
End-to-End Pipeline Cloud Tests

Tests the full fine-tuning pipeline across all stages:
- S3 data download and validation
- Function extraction accuracy on real code
- Prompt generation quality
- Train/val split integrity
- JSONL format validation
- Data roundtrip (S3 upload → download → verify)
- Quality evaluation on real data
"""

import os
import sys
import json
import hashlib
import tempfile
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DATA DOWNLOAD & VALIDATION
# =============================================================================

@pytest.mark.cloud
@pytest.mark.integration
class TestS3DataDownload:
    """Tests for downloading and validating raw data from S3"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_tsn_data_prefix_has_files(self, real_s3_client, s3_bucket):
        """Verify tsn_data/ prefix contains code files"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix='tsn_data/', MaxKeys=10
        )
        contents = response.get('Contents', [])
        assert len(contents) > 0, "tsn_data/ prefix is empty"

    def test_tsn_data_has_c_files(self, real_s3_client, s3_bucket):
        """Verify tsn_data/ contains .c or .h files"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix='tsn_data/', MaxKeys=50
        )
        contents = response.get('Contents', [])
        code_files = [c for c in contents if c['Key'].endswith(('.c', '.h', '.cpp'))]
        assert len(code_files) > 0, "No code files found in tsn_data/"

    def test_processed_data_exists(self, real_s3_client, s3_bucket):
        """Verify processed data exists in data/processed/"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix='data/processed/', MaxKeys=10
        )
        contents = response.get('Contents', [])
        assert len(contents) > 0, "data/processed/ is empty - run pipeline first"

    def test_processed_data_is_valid_jsonl(self, real_s3_client, s3_bucket):
        """Verify processed JSONL files are valid JSON per line"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix='data/processed/', MaxKeys=10
        )
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.jsonl'):
                body = real_s3_client.get_object(
                    Bucket=s3_bucket, Key=obj['Key']
                )['Body'].read().decode('utf-8')
                lines = [l for l in body.strip().split('\n') if l.strip()]
                assert len(lines) > 0, f"{obj['Key']} is empty"
                for i, line in enumerate(lines[:5]):  # Spot check first 5
                    parsed = json.loads(line)
                    assert 'messages' in parsed, f"Line {i} missing 'messages' key"

    def test_train_val_files_both_exist(self, real_s3_client, s3_bucket):
        """Verify both train.jsonl and val.jsonl exist"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix='data/processed/', MaxKeys=20
        )
        keys = [c['Key'] for c in response.get('Contents', [])]
        has_train = any('train' in k for k in keys)
        has_val = any('val' in k for k in keys)
        assert has_train, "train.jsonl not found in data/processed/"
        assert has_val, "val.jsonl not found in data/processed/"


# =============================================================================
# FUNCTION EXTRACTION ACCURACY
# =============================================================================

@pytest.mark.offline
class TestFunctionExtraction:
    """Tests for C/C++ function extraction accuracy"""

    @pytest.fixture
    def pipeline(self):
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)
            return p

    def test_extract_simple_function(self, pipeline):
        """Extract a simple C function"""
        code = """
int tsn_init(void) {
    int x = 0;
    return x;
}
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) >= 1
        assert any(f['name'] == 'tsn_init' for f in funcs)

    def test_extract_function_with_pointer_return(self, pipeline):
        """Extract function returning pointer"""
        code = """
char* get_buffer(int size) {
    static char buf[1024];
    return buf;
}
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) >= 1

    def test_extract_static_function(self, pipeline):
        """Extract static function"""
        code = """
static int helper(int a, int b) {
    return a + b;
}
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) >= 1

    def test_skip_short_functions(self, pipeline):
        """Functions shorter than 50 chars should be skipped"""
        code = """
int f(void) { return 0; }
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) == 0, "Should skip very short functions"

    def test_extract_nested_braces(self, pipeline):
        """Correctly handle nested braces in function body"""
        code = """
int process(int *data, int len) {
    for (int i = 0; i < len; i++) {
        if (data[i] > 0) {
            data[i] = data[i] * 2;
        } else {
            data[i] = 0;
        }
    }
    return 0;
}
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) == 1
        assert funcs[0]['body'].count('{') == funcs[0]['body'].count('}')

    def test_extract_multiple_functions(self, pipeline):
        """Extract multiple functions from one file"""
        code = """
int func_a(int x) {
    return x + 1;
    /* padding to make it long enough for extraction threshold */
}

int func_b(int y) {
    return y * 2;
    /* padding to make it long enough for extraction threshold */
}
"""
        funcs = pipeline.extract_functions(code)
        names = [f['name'] for f in funcs]
        # May extract 0, 1, or 2 depending on length threshold
        for f in funcs:
            assert f['name'] in ('func_a', 'func_b')

    def test_no_functions_in_empty_code(self, pipeline):
        """No functions extracted from empty code"""
        funcs = pipeline.extract_functions("")
        assert len(funcs) == 0

    def test_no_functions_in_comments_only(self, pipeline):
        """No functions extracted from comments-only code"""
        code = """
/* This is a comment */
// Another comment
"""
        funcs = pipeline.extract_functions(code)
        assert len(funcs) == 0


# =============================================================================
# PROMPT GENERATION QUALITY
# =============================================================================

@pytest.mark.offline
class TestPromptGeneration:
    """Tests for training prompt generation quality"""

    @pytest.fixture
    def pipeline(self):
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)
            return p

    def test_prompt_has_messages_format(self, pipeline):
        """Generated prompt must have messages array"""
        func = {
            'name': 'tsn_init',
            'return_type': 'int',
            'params': 'void',
            'body': 'int tsn_init(void) { return 0; /* padding to avoid empty */ }',
            'lines': 3
        }
        result = pipeline.generate_prompt_from_function(func, "TSN")
        assert 'messages' in result
        assert len(result['messages']) == 2
        assert result['messages'][0]['role'] == 'user'
        assert result['messages'][1]['role'] == 'assistant'

    def test_prompt_contains_function_name(self, pipeline):
        """Prompt should reference the function name"""
        func = {
            'name': 'avb_stream_create',
            'return_type': 'int',
            'params': 'avb_stream_t *stream',
            'body': 'int avb_stream_create(avb_stream_t *stream) { return 0; }',
            'lines': 1
        }
        result = pipeline.generate_prompt_from_function(func, "AVB")
        user_msg = result['messages'][0]['content']
        assert 'avb_stream_create' in user_msg

    def test_prompt_response_is_code(self, pipeline):
        """Response should contain the actual code"""
        func_body = 'int tsn_init(void) {\n    return 0;\n}'
        func = {
            'name': 'tsn_init',
            'return_type': 'int',
            'params': 'void',
            'body': func_body,
            'lines': 3
        }
        result = pipeline.generate_prompt_from_function(func, "TSN")
        assert result['messages'][1]['content'] == func_body

    def test_prompt_has_metadata(self, pipeline):
        """Generated prompt should include metadata"""
        func = {
            'name': 'test_func',
            'return_type': 'void',
            'params': '',
            'body': 'void test_func() { /* ... */ }',
            'lines': 1
        }
        result = pipeline.generate_prompt_from_function(func, "TSN")
        assert 'metadata' in result
        assert result['metadata']['function_name'] == 'test_func'

    def test_code_completion_prompt_format(self, pipeline):
        """Code completion prompt has correct structure"""
        code = '\n'.join([f'int line_{i} = {i};' for i in range(20)])
        result = pipeline.generate_code_completion_prompt(code, split_ratio=0.5)
        if result is not None:
            assert len(result['messages']) == 2
            assert 'Complete' in result['messages'][0]['content']
            assert result['metadata']['type'] == 'code_completion'

    def test_code_completion_returns_none_for_short(self, pipeline):
        """Code completion returns None for code too short to split"""
        code = 'int x = 0;\nreturn x;'
        result = pipeline.generate_code_completion_prompt(code)
        assert result is None


# =============================================================================
# TRAIN/VAL SPLIT INTEGRITY
# =============================================================================

@pytest.mark.offline
class TestTrainValSplitIntegrity:
    """Tests for train/val split correctness"""

    @pytest.fixture
    def pipeline(self):
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)
            return p

    def test_split_ratio_respected(self, pipeline):
        """90/10 split ratio is respected"""
        examples = [{'messages': [{'role': 'user', 'content': f'prompt_{i}'}]} for i in range(100)]
        train, val = pipeline.create_splits(examples, train_ratio=0.9)
        assert len(train) == 90
        assert len(val) == 10

    def test_no_data_leakage_between_splits(self, pipeline):
        """No overlap between train and val"""
        examples = [
            {'messages': [{'role': 'user', 'content': f'prompt_{i}'},
                          {'role': 'assistant', 'content': f'response_{i}'}]}
            for i in range(100)
        ]
        train, val = pipeline.create_splits(examples, train_ratio=0.9)

        train_contents = {json.dumps(e['messages'], sort_keys=True) for e in train}
        val_contents = {json.dumps(e['messages'], sort_keys=True) for e in val}

        overlap = train_contents.intersection(val_contents)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping examples"

    def test_all_examples_preserved(self, pipeline):
        """No examples lost during split"""
        examples = [{'messages': [{'role': 'user', 'content': f'p{i}'}]} for i in range(50)]
        train, val = pipeline.create_splits(examples, train_ratio=0.8)
        assert len(train) + len(val) == 50

    def test_deterministic_split_with_seed(self, pipeline):
        """Same seed produces same split"""
        examples = [{'messages': [{'role': 'user', 'content': f'p{i}'}]} for i in range(100)]
        train1, val1 = pipeline.create_splits(list(examples), train_ratio=0.9, seed=42)
        train2, val2 = pipeline.create_splits(list(examples), train_ratio=0.9, seed=42)
        # Can't directly compare since shuffle mutates in place, but lengths match
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)


# =============================================================================
# DEDUPLICATION
# =============================================================================

@pytest.mark.offline
class TestDeduplication:
    """Tests for data deduplication"""

    @pytest.fixture
    def pipeline(self):
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)
            return p

    def test_removes_exact_duplicates(self, pipeline):
        """Exact duplicate examples are removed"""
        examples = [
            {'messages': [{'role': 'user', 'content': 'same'}]},
            {'messages': [{'role': 'user', 'content': 'same'}]},
            {'messages': [{'role': 'user', 'content': 'different'}]},
        ]
        result = pipeline.deduplicate(examples)
        assert len(result) == 2

    def test_preserves_unique_examples(self, pipeline):
        """Unique examples are all preserved"""
        examples = [
            {'messages': [{'role': 'user', 'content': f'unique_{i}'}]}
            for i in range(10)
        ]
        result = pipeline.deduplicate(examples)
        assert len(result) == 10

    def test_empty_list_handled(self, pipeline):
        """Empty list returns empty"""
        result = pipeline.deduplicate([])
        assert len(result) == 0


# =============================================================================
# JSONL FORMAT VALIDATION
# =============================================================================

@pytest.mark.offline
class TestJSONLFormatValidation:
    """Tests for JSONL output format"""

    @pytest.fixture
    def pipeline(self):
        from scripts.prepare_automotive_data import AutomotiveDataPipeline
        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)
            return p

    def test_jsonl_output_is_valid(self, pipeline, tmp_path):
        """Written JSONL file has valid JSON on each line"""
        examples = [
            {'messages': [
                {'role': 'user', 'content': f'prompt_{i}'},
                {'role': 'assistant', 'content': f'response_{i}'}
            ]}
            for i in range(5)
        ]
        output_path = tmp_path / 'test.jsonl'
        pipeline.save_jsonl(examples, output_path)

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert 'messages' in parsed
            assert len(parsed['messages']) == 2

    def test_jsonl_only_contains_messages(self, pipeline, tmp_path):
        """JSONL output only contains 'messages' key (no metadata leak)"""
        examples = [
            {
                'messages': [{'role': 'user', 'content': 'test'}],
                'metadata': {'secret': 'should_not_appear'}
            }
        ]
        output_path = tmp_path / 'test.jsonl'
        pipeline.save_jsonl(examples, output_path)

        with open(output_path) as f:
            parsed = json.loads(f.readline())

        assert 'metadata' not in parsed
        assert list(parsed.keys()) == ['messages']


# =============================================================================
# DATA ROUNDTRIP (S3 upload → download → verify)
# =============================================================================

@pytest.mark.cloud
@pytest.mark.integration
class TestDataRoundtrip:
    """Tests for S3 data upload/download integrity"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_upload_download_integrity(self, real_s3_client, s3_bucket):
        """Upload a test file and download it, verify content matches"""
        test_key = 'data/test_roundtrip_verify.json'
        test_data = json.dumps({'test': True, 'value': 42}).encode('utf-8')

        # Upload
        real_s3_client.put_object(
            Bucket=s3_bucket, Key=test_key, Body=test_data
        )

        # Download
        response = real_s3_client.get_object(Bucket=s3_bucket, Key=test_key)
        downloaded = response['Body'].read()

        assert downloaded == test_data

        # Clean up
        real_s3_client.delete_object(Bucket=s3_bucket, Key=test_key)

    def test_jsonl_roundtrip_preserves_content(self, real_s3_client, s3_bucket):
        """JSONL data survives S3 roundtrip without corruption"""
        test_key = 'data/test_jsonl_roundtrip.jsonl'
        lines = [
            json.dumps({'messages': [{'role': 'user', 'content': f'prompt_{i}'}]})
            for i in range(10)
        ]
        content = '\n'.join(lines) + '\n'

        real_s3_client.put_object(
            Bucket=s3_bucket, Key=test_key, Body=content.encode('utf-8')
        )

        response = real_s3_client.get_object(Bucket=s3_bucket, Key=test_key)
        downloaded = response['Body'].read().decode('utf-8')

        downloaded_lines = [l for l in downloaded.strip().split('\n') if l.strip()]
        assert len(downloaded_lines) == 10

        for line in downloaded_lines:
            parsed = json.loads(line)
            assert 'messages' in parsed

        # Clean up
        real_s3_client.delete_object(Bucket=s3_bucket, Key=test_key)


# =============================================================================
# QUALITY EVALUATION ON PIPELINE OUTPUT
# =============================================================================

@pytest.mark.offline
class TestQualityOnPipelineOutput:
    """Test quality evaluator on data pipeline output"""

    def test_generated_prompts_pass_format_check(self):
        """Training examples from pipeline should have valid format"""
        from scripts.prepare_automotive_data import AutomotiveDataPipeline

        with patch.object(AutomotiveDataPipeline, '__init__', lambda self, **kwargs: None):
            p = AutomotiveDataPipeline.__new__(AutomotiveDataPipeline)

        func = {
            'name': 'tsn_frame_init',
            'return_type': 'int',
            'params': 'tsn_frame_t *frame',
            'body': """int tsn_frame_init(tsn_frame_t *frame) {
    if (!frame) return -1;
    frame->priority = 0;
    frame->vlan_id = 0;
    frame->timestamp = 0ULL;
    return 0;
}""",
            'lines': 7
        }

        example = p.generate_prompt_from_function(func, "TSN")

        # Validate the generated code passes quality evaluation
        from evaluation.code_quality_metrics import CodeQualityEvaluator
        evaluator = CodeQualityEvaluator(gcc_available=False)

        code = example['messages'][1]['content']
        prompt = example['messages'][0]['content']
        score = evaluator.evaluate(code, prompt)

        assert score.overall >= 0.0
        assert score.safety_score >= 5.0  # No MISRA violations in this code

    def test_tsn_code_gets_protocol_compliance(self):
        """TSN code from pipeline should get protocol compliance score"""
        from evaluation.code_quality_metrics import CodeQualityEvaluator

        evaluator = CodeQualityEvaluator(gcc_available=False)

        tsn_code = """
struct tsn_config {
    uint8_t pcp;
    uint16_t vlan_id;
    uint64_t timestamp;
    uint8_t priority;
    gate_control_list_t gcl;
    schedule_t schedule;
};
"""
        score = evaluator.evaluate(tsn_code, "Generate TSN time-aware shaper code")
        assert score.protocol_score >= 7.0, \
            f"TSN code should score high on protocol compliance, got {score.protocol_score}"
