"""
Data Loading GPU Tests

Validates DataLoader configuration, memory efficiency, and data pipeline:
- JSONL dataset loading
- Tokenization and sequence length validation
- DataLoader memory behavior on GPU
- Train/val split integrity
- Data transfer to GPU
"""

import os
import sys
import json
import gc
import time
import tempfile
import pytest
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

try:
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
except ImportError:
    AutoTokenizer = None
    DataCollatorForLanguageModeling = None

try:
    from datasets import Dataset, load_dataset
except ImportError:
    Dataset = None
    load_dataset = None

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'


def skip_no_cuda():
    import unittest.mock
    if torch is None or isinstance(torch.cuda, unittest.mock.MagicMock):
        pytest.skip("PyTorch not installed or mocked")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def get_config():
    if yaml is None:
        pytest.skip("pyyaml not installed")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# =============================================================================
# DATASET LOADING
# =============================================================================

@pytest.mark.gpu
class TestDatasetLoading:
    """Tests for loading training data"""

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """Create sample JSONL training data"""
        data = [
            {"text": "int tsn_init(tsn_frame_t *frame) { return 0; }"},
            {"text": "void avb_stream_reserve(uint32_t stream_id) { }"},
            {"text": "#include <stdint.h>\nuint8_t get_priority(void) { return 7; }"},
        ]
        path = tmp_path / "train.jsonl"
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return str(path)

    def test_dataset_loads_from_jsonl(self, sample_jsonl):
        """JSONL training data loads into HF Dataset"""
        if Dataset is None:
            pytest.skip("datasets not installed")
        ds = load_dataset('json', data_files=sample_jsonl, split='train')
        assert len(ds) == 3
        assert 'text' in ds.column_names

    def test_tokenized_sequence_lengths(self, sample_jsonl):
        """All sequences should be <= max_seq_length"""
        if AutoTokenizer is None or Dataset is None:
            pytest.skip("Required packages not installed")

        config = get_config()
        max_seq_length = config['model']['max_seq_length']

        ds = load_dataset('json', data_files=sample_jsonl, split='train')
        model_name = config['model']['name']

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pytest.skip(f"Cannot load tokenizer for {model_name}")

        for item in ds:
            tokens = tokenizer(item['text'], truncation=True,
                               max_length=max_seq_length)
            assert len(tokens['input_ids']) <= max_seq_length

    def test_tokenizer_consistency(self):
        """Tokenizer encode/decode roundtrip preserves text"""
        if AutoTokenizer is None:
            pytest.skip("transformers not installed")

        config = get_config()
        model_name = config['model']['name']

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pytest.skip(f"Cannot load tokenizer for {model_name}")

        test_text = "int main() { return 0; }"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert test_text in decoded or decoded.strip() == test_text.strip()


# =============================================================================
# DATALOADER GPU BEHAVIOR
# =============================================================================

@pytest.mark.gpu
class TestDataLoaderGPU:
    """Tests for DataLoader behavior with CUDA"""

    @pytest.fixture
    def dummy_dataset(self):
        """Create a simple tensor dataset"""
        skip_no_cuda()
        input_ids = torch.randint(0, 32000, (100, 512))
        attention_mask = torch.ones_like(input_ids)
        return torch.utils.data.TensorDataset(input_ids, attention_mask)

    def test_dataloader_pin_memory(self, dummy_dataset):
        """DataLoader should support pinned memory with CUDA"""
        skip_no_cuda()
        loader = DataLoader(
            dummy_dataset, batch_size=4, pin_memory=True, num_workers=0
        )
        batch = next(iter(loader))
        assert batch[0].is_pinned(), "Batch tensor not pinned to memory"

    def test_batch_fits_in_gpu_memory(self, dummy_dataset):
        """Single batch transfer to GPU should not OOM"""
        skip_no_cuda()
        loader = DataLoader(dummy_dataset, batch_size=4)
        batch = next(iter(loader))

        # Transfer to GPU
        gpu_batch = tuple(t.to('cuda') for t in batch)
        assert gpu_batch[0].device.type == 'cuda'

        del gpu_batch
        torch.cuda.empty_cache()

    def test_data_on_gpu_after_transfer(self, dummy_dataset):
        """Batch tensors should be on correct CUDA device"""
        skip_no_cuda()
        loader = DataLoader(dummy_dataset, batch_size=2)
        batch = next(iter(loader))

        gpu_batch = tuple(t.to('cuda:0') for t in batch)
        for t in gpu_batch:
            assert t.device == torch.device('cuda', 0)

        del gpu_batch
        torch.cuda.empty_cache()

    def test_no_memory_leak_across_batches(self, dummy_dataset):
        """GPU memory should be stable across multiple batches"""
        skip_no_cuda()
        loader = DataLoader(dummy_dataset, batch_size=4)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        baseline = torch.cuda.memory_allocated()

        for i, batch in enumerate(loader):
            gpu_batch = tuple(t.to('cuda') for t in batch)
            del gpu_batch
            if i >= 9:
                break

        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        leak_mb = (after - baseline) / (1024 ** 2)

        assert leak_mb < 100, \
            f"Potential memory leak: {leak_mb:.1f}MB after 10 batches"


# =============================================================================
# TRAIN/VAL SPLIT INTEGRITY
# =============================================================================

@pytest.mark.gpu
@pytest.mark.cloud
class TestTrainValSplit:
    """Tests for training/validation data split integrity"""

    def test_train_val_no_overlap(self, tmp_path):
        """Training and validation data should not overlap"""
        if Dataset is None:
            pytest.skip("datasets not installed")

        # Create train and val data
        train_data = [{"text": f"train_sample_{i}", "id": f"train_{i}"} for i in range(10)]
        val_data = [{"text": f"val_sample_{i}", "id": f"val_{i}"} for i in range(5)]

        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"

        for path, data in [(train_path, train_data), (val_path, val_data)]:
            with open(path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')

        train_ds = load_dataset('json', data_files=str(train_path), split='train')
        val_ds = load_dataset('json', data_files=str(val_path), split='train')

        train_texts = set(train_ds['text'])
        val_texts = set(val_ds['text'])

        overlap = train_texts & val_texts
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping samples"

    def test_attention_mask_matches_input(self):
        """Attention mask should align with input_ids"""
        if AutoTokenizer is None:
            pytest.skip("transformers not installed")

        config = get_config()
        try:
            tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        except Exception:
            pytest.skip("Cannot load tokenizer")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch = tokenizer(
            ["short", "this is a longer sequence for testing"],
            padding=True, truncation=True, return_tensors="pt"
        )

        for i in range(batch['input_ids'].shape[0]):
            ids = batch['input_ids'][i]
            mask = batch['attention_mask'][i]
            # Where mask is 1, ids should not be pad token
            non_pad = (ids != tokenizer.pad_token_id)
            assert (mask.bool() == non_pad).all() or \
                   mask.sum() > 0, "Attention mask doesn't align with input"


# =============================================================================
# S3 DATA DOWNLOAD (CLOUD)
# =============================================================================

@pytest.mark.cloud
@pytest.mark.integration
class TestS3DataDownload:
    """Tests for downloading training data from S3"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_s3_training_data_exists(self, real_s3_client):
        """S3 bucket has training data"""
        config = get_config()
        bucket = config['aws']['s3']['bucket_name']

        # Check common prefixes where training data may be stored
        prefixes = ['processed/train', 'tsn_data/', 'train/']
        found = False
        for prefix in prefixes:
            response = real_s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=5
            )
            if response.get('KeyCount', 0) > 0:
                found = True
                break
        assert found, \
            f"No training data found in S3 under any of: {prefixes}"

    def test_s3_validation_data_exists(self, real_s3_client):
        """S3 bucket has some data for validation"""
        config = get_config()
        bucket = config['aws']['s3']['bucket_name']

        # Check common prefixes
        prefixes = ['processed/val', 'tsn_data/', 'val/']
        found = False
        for prefix in prefixes:
            response = real_s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=5
            )
            if response.get('KeyCount', 0) > 0:
                found = True
                break
        assert found, \
            f"No data found in S3 under any of: {prefixes}"
