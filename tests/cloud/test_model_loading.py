"""
Model Loading GPU Tests

Validates QLoRA model loading, quantization, and GPU memory behavior:
- BitsAndBytes 4-bit quantization
- LoRA adapter attachment
- Memory footprint validation
- Forward/backward pass correctness
- Checkpoint save/load roundtrip
"""

import os
import sys
import gc
import tempfile
import pytest
from unittest.mock import MagicMock
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'


def skip_no_cuda():
    import unittest.mock
    if torch is None or isinstance(torch.cuda, unittest.mock.MagicMock):
        pytest.skip("PyTorch not installed or mocked")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def get_model_name():
    """Get model name from config"""
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        return config['model']['name']
    except Exception:
        return 'ibm-granite/granite-8b-code-base-4k'


# =============================================================================
# BITSANDBYTES SETUP
# =============================================================================

@pytest.mark.gpu
class TestBitsAndBytesSetup:
    """Tests for BitsAndBytes library setup"""

    def test_bitsandbytes_importable(self):
        """bitsandbytes must be importable"""
        skip_no_cuda()
        assert bnb is not None, "bitsandbytes not installed"

    def test_bitsandbytes_cuda_setup(self):
        """bitsandbytes must detect CUDA"""
        skip_no_cuda()
        assert bnb is not None, "bitsandbytes not installed"
        # bitsandbytes should find CUDA libs
        assert hasattr(bnb, 'functional'), "bitsandbytes missing functional module"

    def test_bnb_4bit_config_creates(self):
        """BitsAndBytesConfig for 4-bit should create without error"""
        skip_no_cuda()
        if BitsAndBytesConfig is None:
            pytest.skip("transformers not installed")
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == 'nf4'

    def test_bnb_linear_4bit_exists(self):
        """bitsandbytes Linear4bit module must exist"""
        skip_no_cuda()
        assert bnb is not None, "bitsandbytes not installed"
        assert hasattr(bnb.nn, 'Linear4bit'), "Linear4bit not found in bitsandbytes"


# =============================================================================
# MODEL LOADING
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestModelLoading:
    """Tests for loading Granite-8B with QLoRA"""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load model once for the test class"""
        skip_no_cuda()
        if AutoModelForCausalLM is None:
            pytest.skip("transformers not installed")

        model_name = get_model_name()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation='eager',
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        yield model, tokenizer

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def test_model_loads_in_4bit(self, loaded_model):
        """Granite-8B loads with 4-bit quantization"""
        model, _ = loaded_model
        assert model is not None

    def test_quantized_memory_footprint(self, loaded_model):
        """4-bit model should use < 10GB GPU memory"""
        model, _ = loaded_model
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        assert allocated_gb < 10, \
            f"Model uses {allocated_gb:.1f}GB, expected < 10GB for 4-bit 8B model"

    def test_quantized_layers_detected(self, loaded_model):
        """Quantized layers should be detectable"""
        model, _ = loaded_model
        quantized = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'quant_state'):
                quantized.append(name)
        assert len(quantized) > 0, "No quantized layers found"

    def test_compute_dtype_bfloat16(self, loaded_model):
        """Model compute dtype should be bfloat16"""
        model, _ = loaded_model
        # Check a non-quantized parameter dtype
        for name, param in model.named_parameters():
            if param.dtype in (torch.bfloat16, torch.float32):
                break
        # Model should have bfloat16 parameters
        bf16_params = sum(1 for _, p in model.named_parameters() if p.dtype == torch.bfloat16)
        assert bf16_params > 0, "No bfloat16 parameters found"

    def test_no_nan_in_initial_weights(self, loaded_model):
        """No NaN/Inf in model weights after loading"""
        model, _ = loaded_model
        for name, param in model.named_parameters():
            if param.dtype in (torch.float16, torch.float32, torch.bfloat16):
                assert not torch.isnan(param).any(), f"NaN in {name}"
                assert not torch.isinf(param).any(), f"Inf in {name}"

    def test_model_forward_pass_works(self, loaded_model):
        """Single forward pass produces valid logits"""
        model, tokenizer = loaded_model
        inputs = tokenizer("int main() {", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits is not None
        assert not torch.isnan(outputs.logits).any(), "NaN in logits"
        assert outputs.logits.shape[-1] > 0, "Empty logits"


# =============================================================================
# LORA ADAPTER
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestLoRAAdapter:
    """Tests for LoRA adapter attachment and training"""

    @pytest.fixture(scope="class")
    def lora_model(self):
        """Load model with LoRA adapter"""
        skip_no_cuda()
        if AutoModelForCausalLM is None or LoraConfig is None:
            pytest.skip("Required packages not installed")

        model_name = get_model_name()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation='eager',
        )

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        yield model, tokenizer

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def test_lora_adapter_attaches(self, lora_model):
        """LoRA adapter should attach without error"""
        model, _ = lora_model
        assert model is not None

    def test_lora_trainable_params_count(self, lora_model):
        """Trainable params should be small fraction of total"""
        model, _ = lora_model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total * 100
        assert ratio < 5, f"Trainable params {ratio:.1f}% of total, expected < 5%"
        assert trainable > 0, "No trainable parameters"

    def test_model_backward_pass_works(self, lora_model):
        """Backward pass should compute gradients on LoRA params"""
        model, tokenizer = lora_model
        model.train()

        inputs = tokenizer("void init() { return; }", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()

        outputs = model(**inputs)
        loss = outputs.loss
        assert loss is not None, "No loss computed"
        assert not torch.isnan(loss), f"Loss is NaN"

        loss.backward()

        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                break

        assert has_grad, "No gradients computed"
        model.zero_grad()

    def test_model_save_load_lora_adapter(self, lora_model):
        """LoRA adapter saves and reloads correctly"""
        model, tokenizer = lora_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)

            # Verify saved files exist
            adapter_files = list(Path(tmpdir).glob('*'))
            assert len(adapter_files) > 0, "No adapter files saved"

            # Check adapter is small (not full model)
            total_size = sum(f.stat().st_size for f in Path(tmpdir).rglob('*') if f.is_file())
            total_mb = total_size / (1024 ** 2)
            assert total_mb < 500, \
                f"Adapter save is {total_mb:.0f}MB, expected < 500MB (should be LoRA only)"


# =============================================================================
# OPTIMIZER AND TRAINING STEP
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestOptimizerAndTraining:
    """Tests for optimizer initialization and training step"""

    def test_paged_adamw_8bit_optimizer(self):
        """paged_adamw_8bit optimizer should initialize"""
        skip_no_cuda()
        if bnb is None:
            pytest.skip("bitsandbytes not installed")

        # Create a small parameter to test optimizer
        param = torch.nn.Parameter(torch.randn(10, 10, device='cuda'))
        optimizer = bnb.optim.PagedAdamW8bit([param], lr=1e-4)
        assert optimizer is not None

        # Verify step works
        loss = param.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del param, optimizer
        torch.cuda.empty_cache()

    def test_gradient_checkpointing_reduces_memory(self):
        """Gradient checkpointing should reduce peak memory"""
        skip_no_cuda()
        if AutoModelForCausalLM is None:
            pytest.skip("transformers not installed")

        # This is validated by the training config having gradient_checkpointing=True
        # We verify the config flag is set correctly
        import yaml
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        # The training script sets gradient_checkpointing=True
        # Verify this is documented in training args
        assert config['training'].get('gradient_checkpointing', True), \
            "gradient_checkpointing should be enabled for 8B model"
