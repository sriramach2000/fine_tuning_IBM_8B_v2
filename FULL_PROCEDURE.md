# Granite-8B Fine-Tuning Pipeline: Complete Procedure

**Project**: Fine-tuning IBM Granite-8B-Code-Instruct-128K for Embedded Automotive Code Generation
**Domains**: AVB (Audio Video Bridging) & TSN (Time-Sensitive Networking)
**Author**: Sriram Acharya, Excelfore Corporation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [AWS Infrastructure Deployment](#3-aws-infrastructure-deployment)
4. [Phase 1: Pipeline Validation (Dry Run)](#4-phase-1-pipeline-validation-dry-run)
5. [Phase 2: Data Preparation](#5-phase-2-data-preparation)
6. [Phase 3: Teacher Output Generation](#6-phase-3-teacher-output-generation)
7. [Phase 4: Training Job Launch](#7-phase-4-training-job-launch)
8. [Phase 5: Iterative Distillation](#8-phase-5-iterative-distillation)
9. [Quality Evaluation System](#9-quality-evaluation-system)
10. [Configuration Reference](#10-configuration-reference)
11. [Script CLI Reference](#11-script-cli-reference)
12. [Test Suite](#12-test-suite)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Cost Estimates](#14-cost-estimates)
15. [Troubleshooting](#15-troubleshooting)
16. [File Inventory](#16-file-inventory)

---

## 1. Architecture Overview

### High-Level Pipeline

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Validation   │───▶│  Data Preparation │───▶│ Teacher Gen     │───▶│ Training Launch   │───▶│ Distillation │
│  (dry_run)    │    │  (prepare_data)   │    │ (Bedrock Claude) │    │ (SageMaker)       │    │ (iterative)  │
└──────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘    └──────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Student Model** | IBM Granite-8B-Code-Instruct-128K | The model being fine-tuned |
| **Teacher Model** | Claude Sonnet 4.5 via Amazon Bedrock | Generates high-quality corrections |
| **Training Method** | QLoRA (4-bit NF4 + LoRA rank=32) | Memory-efficient fine-tuning |
| **Compute** | AWS SageMaker (ml.p4d.24xlarge) | 8x A100 GPUs for training |
| **Data Store** | S3 (`granite-8b-unified-automotive-data`) | Raw data, splits, model artifacts |
| **Evaluation** | Custom CodeQualityEvaluator | Syntax, protocol, safety, style scoring |

### Data Flow

```
S3 (24 automotive sources)
    │
    ▼ aws s3 sync
Local/SageMaker (C/C++ files)
    │
    ▼ extract_functions() + generate_prompts()
JSONL (train.jsonl, val.jsonl)
    │
    ├──▶ Bedrock Claude → Teacher outputs (knowledge distillation seeds)
    │
    ▼ SageMaker Training Job
Granite-8B + QLoRA
    │
    ├──▶ Student generates → Evaluate quality → Poor outputs → Teacher corrects
    │                                                              │
    │◀────────── Corrections added to training set ◀───────────────┘
    │
    ▼ Converged
Fine-tuned LoRA adapters → S3 (model.tar.gz)
```

---

## 2. Prerequisites & Environment Setup

### 2.1 System Requirements

- **Python**: 3.10+
- **CUDA**: 12.1+ (for local GPU testing)
- **Disk**: 50GB+ free space (for raw data download)
- **RAM**: 16GB+ (for data processing)
- **AWS CLI**: v2 installed and configured

### 2.2 Install Dependencies

```bash
git clone git@github.com:YOUR_USERNAME/fine_tuning_IBM_8B_v2.git
cd fine_tuning_IBM_8B_v2
pip install -r requirements.txt
```

**Key packages**:
- `torch>=2.1.0` — PyTorch
- `transformers>=4.45.2` — Hugging Face model loading
- `peft>=0.12.0` — LoRA/QLoRA adapters
- `trl>=0.9.6` — SFTTrainer for supervised fine-tuning
- `bitsandbytes>=0.43.0` — 4-bit quantization
- `boto3>=1.34.0` — AWS SDK
- `sagemaker>=2.199.0` — SageMaker Python SDK
- `neo4j>=5.14.0` — Knowledge graph context database
- `sentence-transformers>=2.2.2` — Embedding model for context DB

### 2.3 Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Required variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | IAM user access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key | Yes |
| `AWS_REGION` | Default: `us-east-1` | Yes |
| `HF_TOKEN` | Hugging Face API token (starts with `hf_`) | Yes |
| `AMAZON_BEDROCK_MODEL_API_KEY` | Bedrock API key for additional auth | Optional |
| `SAGEMAKER_ROLE` | SageMaker execution role ARN | For cloud jobs |

### 2.4 Verify AWS Access

```bash
aws sts get-caller-identity
aws s3 ls s3://granite-8b-unified-automotive-data/ --max-items 5
```

---

## 3. AWS Infrastructure Deployment

### 3.1 CloudFormation Stack

Deploy the infrastructure stack that creates all required AWS resources:

```bash
aws cloudformation deploy \
    --template-file aws/cloudformation/sagemaker_stack.yaml \
    --stack-name granite-8b-avb-tsn-finetuning \
    --capabilities CAPABILITY_NAMED_IAM
```

**Resources created**:

| Resource | Type | Purpose |
|----------|------|---------|
| `SageMakerExecutionRole` | IAM Role | SageMaker + S3 + Bedrock + CloudWatch access |
| `SageMakerLogGroup` | CloudWatch Log Group | Training logs (30-day retention) |
| `AlertTopic` | SNS Topic | Training failure notifications |
| `TrainingFailureAlarm` | CloudWatch Alarm | Auto-alerts on job failure |
| `ECRRepository` | ECR Repository | Custom training container images (keeps last 10) |

**Stack parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ProjectName` | `granite-8b-avb-tsn-finetuning` | Names all resources |
| `S3BucketName` | `granite-8b-unified-automotive-data` | Training data bucket |
| `Environment` | `development` | `development` / `staging` / `production` |

**Stack outputs** (used by other scripts):
- `SageMakerRoleArn` — Role ARN for training jobs
- `ECRRepositoryUri` — Container image URI
- `LogGroupName` — CloudWatch log group
- `AlertTopicArn` — SNS topic for alerts

### 3.2 Docker Image (Optional)

If using a custom container instead of the HuggingFace DLC:

```bash
# Build
docker build -t granite-8b-finetuning -f docker/Dockerfile .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com
docker tag granite-8b-finetuning:latest <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/granite-8b-avb-tsn-finetuning:latest
docker push <ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/granite-8b-avb-tsn-finetuning:latest
```

**Dockerfile details**:
- Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
- Includes: transformers, peft, trl, bitsandbytes, accelerate, boto3
- Entrypoint: `python train_granite_qlora.py`
- CUDA optimizations: `CUDA_LAUNCH_BLOCKING=1`, expandable segments

---

## 4. Phase 1: Pipeline Validation (Dry Run)

**Script**: `scripts/dry_run_pipeline.py`
**Purpose**: Validate the entire pipeline before spending money on AWS resources.

### Run

```bash
python scripts/dry_run_pipeline.py

# Skip Bedrock test (saves API cost)
python scripts/dry_run_pipeline.py --skip-bedrock
```

### What It Tests

| Test | Checks | Pass Criteria |
|------|--------|---------------|
| **AWS Credentials** | `sts:GetCallerIdentity` | Valid IAM credentials |
| **S3 Bucket Access** | `s3:ListObjects`, checks `tsn_data/` and `advanced_academic/` prefixes | Objects found |
| **Bedrock Access** | Sends test prompt to Claude Sonnet 4.5 | Response received |
| **IAM Role** | `iam:GetRole` for SageMaker execution role | Role exists |
| **HuggingFace Token** | Checks `HF_TOKEN` env var | Starts with `hf_` |
| **Configuration** | Validates all required sections in `config.yaml` | `model`, `qlora`, `training`, `aws`, `distillation` present |
| **Local Files** | Checks all 7 required files exist | All found |
| **Script Syntax** | `compile()` on `train_granite_qlora.py` | Valid Python |
| **Data Pipeline Import** | `import prepare_automotive_data` | No import errors |

### Output

```
Tests Passed: 12/12
Tests Failed: 0/12
All tests passed! Pipeline is ready for training.
```

**If tests fail**: Fix each `[FAIL]` item before proceeding. Common issues:
- Missing `.env` file → copy and fill in credentials
- IAM role not found → deploy CloudFormation stack
- Bedrock access denied → enable Claude model in Bedrock console

---

## 5. Phase 2: Data Preparation

**Script**: `scripts/prepare_automotive_data.py`
**Alternative (cloud)**: `scripts/launch_processing_job.py` → runs `scripts/sagemaker_processing.py` on SageMaker

### 5.1 Data Sources

The pipeline processes **24 S3 prefixes** across 8 automotive domains:

| Domain | S3 Prefixes | Content |
|--------|-------------|---------|
| **TSN** | `tsn_data/` | IEEE 802.1Qbv/Qav/AS protocol implementations |
| **AVB** | `avb_data/` | Audio Video Bridging, SRP, AVTP |
| **CARLA** | `advanced_academic/carla_autonomous_driving_simulator/` | Autonomous driving C++ code |
| **Embedded** | `advanced_embedded/`, `phase_2_embedded/`, `phase_3_embedded/` | Bare-metal firmware, HAL, MCU drivers |
| **RTOS** | `advanced_rtos/`, `phase_2_rtos/`, `phase_3_rtos/`, `nxp_automotive_freertos/`, `nxp_s32k_freertos_bsp/`, `car_freertos_example/` | FreeRTOS, NXP BSP |
| **Middleware** | `advanced_middleware/`, `phase_2_middleware/`, `covesa_commonapi_core_tools/`, `genivi_candevstudio/` | SOME/IP, CommonAPI, CAN |
| **Safety** | `advanced_safety/`, `phase_3_safety/`, `functional_safety_examples/` | ISO 26262, MISRA-C, ASIL |
| **Other** | `autosar_learning_project/`, `phase_2_academic/`, `phase_3_academic/`, `awesome_vehicle_security/` | AUTOSAR, research, cybersecurity |

**File extensions processed**: `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.txt`

### 5.2 Processing Pipeline

1. **Download**: `aws s3 sync` with extension filters (fast bulk download)
2. **Read**: UTF-8 with Latin-1 fallback encoding
3. **Extract functions**: Regex-based C/C++ function extraction (brace-matching)
4. **Generate prompts**: 4 template variants per function + code completion examples
5. **Deduplicate**: MD5 hash of `messages` content
6. **Split**: 90/10 train/val (seed=42, shuffled)
7. **Save**: JSONL format (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`)

### 5.3 Run Locally

```bash
# Full pipeline (download + process + save)
python scripts/prepare_automotive_data.py

# Test with 10 files per source
python scripts/prepare_automotive_data.py --sample-only

# Skip download (already have local data)
python scripts/prepare_automotive_data.py --no-download

# Full pipeline + upload splits to S3 for Colab
python scripts/prepare_automotive_data.py --upload-splits

# Custom S3 output location
python scripts/prepare_automotive_data.py --upload-splits --output-bucket granite-8b-training-outputs --output-prefix runs/data/splits
```

### 5.4 Run on SageMaker (Cloud)

For large datasets, use a SageMaker Processing Job to avoid local processing:

```bash
# Launch processing job on ml.c5.xlarge ($0.05/hr spot)
python scripts/launch_processing_job.py

# Larger instance
python scripts/launch_processing_job.py --instance-type ml.c5.2xlarge --workers 8
```

The `sagemaker_processing.py` script is a memory-optimized version that:
- Streams examples to disk (not RAM) to stay under 1GB memory
- Uses `multiprocessing.Pool` with `imap_unordered` for parallel processing
- Deduplicates via streaming (only hashes in memory)

### 5.5 Output

```
data/splits/
├── train.jsonl    # ~90% of examples
└── val.jsonl      # ~10% of examples
```

Each line:
```json
{"messages": [{"role": "user", "content": "Generate a C function named 'tsn_init_shaper'..."}, {"role": "assistant", "content": "void tsn_init_shaper(tsn_config_t *config) {\n..."}]}
```

---

## 6. Phase 3: Teacher Output Generation

**Script**: `scripts/generate_teacher_outputs.py`
**Purpose**: Generate high-quality code examples using Claude Sonnet 4.5 via Amazon Bedrock for knowledge distillation.

### 6.1 How It Works

1. Load prompts from JSONL (or use built-in test prompts)
2. Send each prompt to Claude Sonnet 4.5 via Bedrock with automotive system prompt
3. Process in parallel (ThreadPoolExecutor, default 10 workers)
4. Retry with exponential backoff on throttling (up to 5 retries)
5. Checkpoint every 10 results
6. Save results as JSONL

### 6.2 System Prompt

The teacher receives a system prompt establishing it as an expert in:
- TSN protocols (IEEE 802.1Qbv, 802.1Qav, 802.1AS)
- AVB for automotive infotainment
- Real-time embedded C/C++
- MISRA-C safety guidelines
- Automotive Ethernet

### 6.3 Run

```bash
# Test mode (5 built-in automotive prompts)
python scripts/generate_teacher_outputs.py --test-mode --prompts 5

# Full generation from processed prompts
python scripts/generate_teacher_outputs.py \
    --input-file data/processed/prompts.jsonl \
    --output-file data/teacher_outputs/bedrock_outputs.jsonl

# Custom settings
python scripts/generate_teacher_outputs.py \
    --input-file data/processed/prompts.jsonl \
    --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --max-tokens 2048 \
    --temperature 0.7 \
    --workers 20
```

### 6.4 Authentication

The script supports dual authentication:
1. **Primary**: IAM credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`)
2. **Secondary**: Bedrock API key (`AMAZON_BEDROCK_MODEL_API_KEY`) added as `x-amz-bedrock-api-key` header

### 6.5 Rate Limiting & Error Handling

| Error | Action | Delay |
|-------|--------|-------|
| `ThrottlingException` | Exponential backoff retry | `1s × 2^attempt` |
| `ModelStreamErrorException` | Exponential backoff retry | `1s × 2^attempt` |
| Max retries exceeded | Skip prompt, record failure | — |
| Other errors | Record failure, continue | — |

### 6.6 Output

```
data/teacher_outputs/bedrock_outputs.jsonl
```

Each line:
```json
{"id": "tsn_tas_001", "prompt": "Generate C code for...", "response": "```c\nvoid tsn_tas_init...", "success": true, "usage": {"input_tokens": 150, "output_tokens": 800}, "timestamp": "2026-02-15T10:30:00"}
```

---

## 7. Phase 4: Training Job Launch

**Scripts**:
- `scripts/run_training_job.py` — Standard QLoRA training on SageMaker
- `scripts/launch_cloud_training.py` — Iterative distillation on SageMaker

### 7.1 Standard QLoRA Training

Uses the HuggingFace Estimator to launch `training/train_granite_qlora.py` on SageMaker:

```bash
# Dry run (print config, don't launch)
python scripts/run_training_job.py --dry-run

# Launch with spot instances (default)
python scripts/run_training_job.py --upload-data

# Override instance type
python scripts/run_training_job.py --instance-type ml.g5.12xlarge

# Disable spot (on-demand)
python scripts/run_training_job.py --no-spot
```

**What it does**:
1. Loads `config.yaml`
2. Gets AWS account ID and IAM role ARN
3. Optionally uploads local `data/splits/` to S3
4. Verifies `train.jsonl` and `val.jsonl` exist in S3
5. Creates HuggingFace Estimator with all hyperparameters
6. Launches training job asynchronously
7. Streams CloudWatch logs

### 7.2 Iterative Distillation Launch

For the full iterative teacher-student loop:

```bash
# Dry run
python scripts/launch_cloud_training.py --dry-run

# Launch with spot instances
python scripts/launch_cloud_training.py --use-spot

# Custom configuration
python scripts/launch_cloud_training.py \
    --instance-type ml.g5.12xlarge \
    --max-epochs 5 \
    --quality-threshold 7.0 \
    --convergence-threshold 8.0 \
    --max-corrections-per-epoch 500 \
    --eval-samples 200
```

### 7.3 SageMaker Job Details

| Setting | Value |
|---------|-------|
| **Container** | HuggingFace DLC: PyTorch 2.5.1, Transformers 4.49.0, CUDA 12.4, Python 3.11 |
| **Default Instance** | `ml.p4d.24xlarge` (8× A100 80GB) |
| **Budget Instance** | `ml.g5.12xlarge` (4× A10G 24GB) |
| **Volume** | 200 GB EBS |
| **Max Runtime** | 8 hours (standard) / 24 hours (distillation) |
| **Spot Instances** | Enabled by default (max wait: 10 hours) |
| **Input Channels** | `train`: `s3://.../train.jsonl`, `val`: `s3://.../val.jsonl` |

---

## 8. Phase 5: Iterative Distillation

**Script**: `training/iterative_distillation.py`
**Purpose**: Closed-loop training where the teacher corrects the student's worst outputs each epoch.

### 8.1 Epoch Loop

Each epoch executes 4 steps:

```
Step 1: Train student on current dataset (original + accumulated corrections)
    │
Step 2: Generate student outputs on eval prompts (200 samples)
    │ temperature=0.7, top_p=0.95, max_new_tokens=1024
    │
Step 3: Evaluate quality (syntax + protocol + safety + style)
    │
Step 4: Identify poor outputs (score < 7.0)
    │
    ├── Sort by score (worst first)
    ├── Cap at 500 corrections per epoch
    ├── Send to teacher (parallel, 10 workers)
    └── Add corrections to training dataset
```

### 8.2 Convergence Criteria

Training stops when ANY of these conditions is met:

| Criterion | Condition |
|-----------|-----------|
| **Quality Threshold** | Average student score ≥ 8.0 for **3 consecutive epochs** |
| **Low Correction Rate** | Correction rate < 10% for **3 consecutive epochs** |
| **Max Epochs** | Reached `max_iterations` (default: 5) |

### 8.3 Configuration

```python
@dataclass
class DistillationConfig:
    quality_threshold: float = 7.0      # Min score to pass without correction
    convergence_threshold: float = 8.0  # Target avg score for convergence
    convergence_patience: int = 3       # Epochs at threshold before stopping
    max_corrections_per_epoch: int = 500
    max_parallel_teacher_calls: int = 10
    eval_samples_per_epoch: int = 200
```

### 8.4 Correction Prompt

When a student output scores below 7.0, the teacher receives:

```
A student model was asked to generate automotive embedded code for the following prompt:
<prompt>{original_prompt}</prompt>

The student produced this output (scored {score}/10):
<student_output>{student_output}</student_output>

Please provide a corrected, production-quality version that:
1. Fixes any errors or bugs
2. Follows MISRA-C guidelines
3. Includes proper error handling
4. Uses correct data types (uint8_t, uint32_t, etc.)
5. Is suitable for embedded automotive systems (TSN/AVB)
```

### 8.5 Metrics Tracked Per Epoch

| Metric | Description |
|--------|-------------|
| `train_loss` | SFTTrainer training loss |
| `avg_student_score` | Mean quality score across eval samples |
| `num_poor_outputs` | Outputs scoring < 7.0 |
| `num_corrections` | Corrections received from teacher |
| `correction_rate` | `num_poor_outputs / eval_samples` |
| `scores_distribution` | Counts in brackets: <5, 5-7, 7-9, 9+ |

---

## 9. Quality Evaluation System

**Module**: `evaluation/code_quality_metrics.py`
**Class**: `CodeQualityEvaluator`

### 9.1 Scoring Dimensions

| Dimension | Weight | Range | Method |
|-----------|--------|-------|--------|
| **Syntax** | 30% | 0-10 | GCC `-fsyntax-only` or heuristic fallback |
| **Protocol Compliance** | 30% | 0-10 | TSN/AVB keyword matching |
| **Safety (MISRA-C)** | 25% | 0-10 | Violation pattern detection |
| **Style** | 15% | 0-10 | Comment count, Doxygen, naming, indentation |

**Overall = 0.30×syntax + 0.30×protocol + 0.25×safety + 0.15×style**

### 9.2 Syntax Scoring

| Result | Score |
|--------|-------|
| Clean compile (0 warnings) | 10.0 |
| Warnings only | 9.0 − 0.5 per warning (min 7.0) |
| Errors present | 5.0 − 1.0 per error (min 0.0) |
| GCC unavailable | Heuristic: base 7.0 ± brace/paren/semicolon penalties |

### 9.3 Protocol Keywords

**TSN keywords** (need ≥3): `pcp`, `vlan`, `timestamp`, `gate`, `priority`, `qbv`, `qav`, `shaper`, `gcl`, `schedule`

**AVB keywords** (need ≥3): `stream`, `sample`, `channel`, `bandwidth`, `srp`, `talker`, `listener`, `reservation`, `class`

**Penalty**: −1.5 per missing keyword below the minimum of 3.

### 9.4 MISRA-C Violations

| Pattern | Penalty | MISRA Rule |
|---------|---------|------------|
| `goto` | −3.0 | 15.1 |
| `malloc`/`free`/`realloc`/`calloc` | −2.0 each | 21.3 |
| `setjmp`/`longjmp` | −3.0 each | 21.4 |
| `abort`/`exit` | −1.5 each | 21.8 |
| Recursion detected | −1.5 | — |
| `while(1)` without `break` | −2.0 | — |

### 9.5 Good Practice Bonuses

| Pattern | Bonus |
|---------|-------|
| Fixed-width types (`uint8_t`, etc.) | +0.5 each |
| `static` linkage | +0.3 |
| `const` correctness | +0.3 |
| `volatile` keyword | +0.2 |
| Doxygen `/**` blocks | +1.0 |
| `@param`/`@return`/`@brief` | +0.5 each |

---

## 10. Configuration Reference

**File**: `config.yaml`

### 10.1 Model Configuration

```yaml
model:
  name: "ibm-granite/granite-8b-code-instruct-128k"
  context_length: 131072    # 128K native context
  max_seq_length: 4096      # Training sequence length (limited for memory)
  trust_remote_code: false
```

### 10.2 QLoRA Configuration

```yaml
qlora:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true    # Nested quantization (saves ~0.4 bits/param)
  bnb_4bit_quant_type: "nf4"         # NormalFloat4 quantization
  bnb_4bit_compute_dtype: "bfloat16" # BF16 recommended for Granite
  lora_r: 32                          # LoRA rank (higher for 8B model)
  lora_alpha: 64                      # Scaling factor (2× rank)
  lora_dropout: 0.05
  lora_target_modules:                # All attention + MLP projections
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### 10.3 Training Hyperparameters

```yaml
training:
  num_epochs: 5
  per_device_train_batch_size: 2       # Per GPU
  gradient_accumulation_steps: 8       # Effective batch = 2 × 8 = 16
  learning_rate: 1.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 0.3
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true         # Saves VRAM at ~30% speed cost
  bf16: true
  logging_steps: 10
  eval_steps: 50
  save_steps: 100
  early_stopping_patience: 3
```

### 10.4 Distillation Configuration

```yaml
distillation:
  teacher_model: "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
  teacher_provider: "bedrock"
  temperature: 2.0           # Softening factor
  alpha: 0.7                 # Weight: 70% distillation, 30% ground truth
  max_teacher_tokens: 2048
  batch_size: 50
  min_score_threshold: 7
  convergence_threshold: 8.0
  max_iterations: 5
```

### 10.5 AWS Configuration

```yaml
aws:
  region: "us-east-1"
  account_id: "122634724608"
  s3:
    bucket_name: "granite-8b-unified-automotive-data"
  iam:
    role_name: "granite-8b-avb-tsn-finetuning-sagemaker-role"
  training_job:
    instance_type: "ml.p4d.24xlarge"    # 8× A100
    volume_size_gb: 200
    max_runtime_seconds: 28800          # 8 hours
    use_spot_instances: true
    max_wait_seconds: 36000             # 10 hours
  bedrock:
    model_id: "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_tokens: 2048
    temperature: 0.7
```

---

## 11. Script CLI Reference

### 11.1 `scripts/dry_run_pipeline.py`

```
usage: dry_run_pipeline.py [--config CONFIG] [--skip-bedrock]

Options:
  --config CONFIG        Path to config file (default: config.yaml)
  --skip-bedrock         Skip Bedrock test to save API cost
```

### 11.2 `scripts/prepare_automotive_data.py`

```
usage: prepare_automotive_data.py [OPTIONS]

Options:
  --s3-bucket BUCKET     Source S3 bucket (default: granite-8b-unified-automotive-data)
  --region REGION        AWS region (default: us-east-1)
  --max-files N          Max files per data type, 0=unlimited (default: 0)
  --train-ratio RATIO    Train/val split ratio (default: 0.9)
  --sample-only          Process only 10 files per source
  --no-download          Skip S3 download
  --upload-splits        Upload train.jsonl/val.jsonl to S3
  --output-bucket BUCKET Output S3 bucket (default: granite-8b-training-outputs)
  --output-prefix PREFIX S3 prefix for uploads (default: runs/data/splits)
```

### 11.3 `scripts/generate_teacher_outputs.py`

```
usage: generate_teacher_outputs.py [OPTIONS]

Options:
  --input-file FILE      JSONL file with prompts
  --output-file FILE     Output JSONL (default: data/teacher_outputs/bedrock_outputs.jsonl)
  --model-id ID          Bedrock model ID (default: Claude Sonnet 4.5)
  --region REGION        AWS region (default: us-east-1)
  --max-tokens N         Max tokens per response (default: 2048)
  --temperature T        Generation temperature (default: 0.7)
  --test-mode            Use built-in sample prompts
  --prompts N            Number of test prompts (default: 5)
  --workers N            Parallel workers (default: 10, max recommended: 50)
```

### 11.4 `scripts/run_training_job.py`

```
usage: run_training_job.py [OPTIONS]

Options:
  --config CONFIG        Path to config file (default: config.yaml)
  --instance-type TYPE   Override instance type (e.g., ml.g5.12xlarge)
  --no-spot              Disable spot instances
  --upload-data          Upload local data/splits/ to S3
  --local-data-dir DIR   Local data directory (default: ./data/splits)
  --max-steps N          Override max training steps
  --dry-run              Show config without launching
```

### 11.5 `scripts/launch_cloud_training.py`

```
usage: launch_cloud_training.py [OPTIONS]

Options:
  --job-name-prefix PREFIX  Job name prefix (default: granite-8b-distillation)
  --instance-type TYPE      Instance type (default: ml.g5.12xlarge)
  --instance-count N        Number of instances (default: 1)
  --volume-size GB          EBS volume size (default: 100)
  --max-run-seconds N       Max runtime (default: 86400 = 24h)
  --use-spot                Enable spot instances
  --max-wait-seconds N      Max spot wait (default: 172800 = 48h)
  --s3-bucket BUCKET        S3 bucket (default: granite-8b-unified-automotive-data)
  --train-prefix PREFIX     S3 training data prefix (default: data/processed)
  --eval-prefix PREFIX      S3 eval data prefix (default: data/eval)
  --output-prefix PREFIX    S3 output prefix (default: output/distillation)
  --max-epochs N            Max distillation epochs (default: 5)
  --quality-threshold T     Min quality score (default: 7.0)
  --convergence-threshold T Target avg score (default: 8.0)
  --max-corrections-per-epoch N  Max teacher calls per epoch (default: 500)
  --eval-samples N          Eval samples per epoch (default: 200)
  --image-uri URI           Custom container image
  --role ARN                SageMaker execution role
  --dry-run                 Print config without launching
```

### 11.6 `scripts/launch_processing_job.py`

```
usage: launch_processing_job.py [OPTIONS]

Options:
  --region REGION           AWS region (default: us-east-1)
  --source-bucket BUCKET    Source S3 bucket (default: granite-8b-unified-automotive-data)
  --output-bucket BUCKET    Output S3 bucket (default: granite-8b-training-outputs)
  --output-prefix PREFIX    S3 prefix for output (default: runs/data/splits)
  --instance-type TYPE      SageMaker instance (default: ml.c5.xlarge)
  --workers N               Parallel workers (default: 4)
  --no-spot                 Use on-demand pricing
  --max-runtime-hours N     Max runtime (default: 4)
```

---

## 12. Test Suite

### 12.1 Local Tests

```bash
# Run all local tests
pytest tests/ -v

# Specific test files
pytest tests/test_data_pipeline.py -v          # Data processing logic
pytest tests/test_bedrock_generator.py -v      # Teacher generation mocks
pytest tests/test_early_stopping.py -v         # Early stopping callback
pytest tests/test_failure_scenarios.py -v      # Error handling
pytest tests/test_iterative_distillation.py -v # Distillation loop
pytest tests/test_ml_paradigm.py -v            # ML training paradigms
pytest tests/test_context_db.py -v             # Knowledge graph context DB
```

### 12.2 Cloud Tests (Run on SageMaker)

```bash
# Launch cloud test suite as SageMaker job
python tests/cloud/launch_cloud_tests.py

# Or run individual test modules
python tests/cloud/run_cloud_tests.py
```

**Cloud test modules**:

| Test File | What It Validates |
|-----------|-------------------|
| `test_s3_operations.py` | S3 read/write/list operations |
| `test_bedrock_operations.py` | Bedrock model invocation |
| `test_quality_evaluator.py` | Code quality scoring on real outputs |
| `test_concurrency.py` | Parallel teacher calls under load |
| `test_data_integrity.py` | JSONL format, encoding, dedup |
| `test_pipeline_e2e.py` | End-to-end pipeline on small dataset |
| `test_teacher_generation.py` | Bedrock teacher prompt/response cycle |
| `test_config_validation.py` | Config.yaml schema validation |
| `test_distillation_e2e.py` | Full distillation loop (small scale) |
| `test_model_loading.py` | Granite-8B loading + quantization |
| `test_resource_management.py` | Memory, disk, GPU resource checks |
| `test_dependency_matrix.py` | Package version compatibility |
| `test_data_loading.py` | Dataset loading via HuggingFace datasets |
| `test_gpu_environment.py` | CUDA, GPU detection, memory |
| `test_sagemaker_environment.py` | SageMaker env vars, paths |
| `test_pipeline_state_machine.py` | State transitions match spec |

### 12.3 GPU Tests

```bash
# Launch GPU-specific tests on a GPU instance
python tests/cloud/launch_gpu_tests.py
```

---

## 13. Monitoring & Observability

### 13.1 CloudWatch Metrics

Namespace: `Granite8BFineTuning`

| Metric | Threshold | Alert |
|--------|-----------|-------|
| `TrainingLoss` | < 2.0 | — |
| `EvalLoss` | < 2.5 | — |
| `TrainingJobStatus` | < 1 | SNS alert |

### 13.2 Custom Metrics (from Distillation)

Parsed from CloudWatch logs via regex:

| Pattern | Regex |
|---------|-------|
| `train_loss` | `Train loss: ([0-9\.]+)` |
| `avg_student_score` | `Avg student score: ([0-9\.]+)` |
| `correction_rate` | `Correction rate: ([0-9\.]+)%` |
| `num_corrections` | `Corrections: ([0-9]+)` |

### 13.3 Monitoring Commands

```bash
# Check training job status
aws sagemaker describe-training-job --training-job-name <JOB_NAME>

# Tail CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix <JOB_NAME>

# View in console
# https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/<JOB_NAME>
```

### 13.4 Training Callbacks

The training script includes two safety callbacks:

**NaNInfDetectionCallback**: Halts training immediately if loss becomes NaN or Inf. Logs CUDA memory state for debugging.

**CustomEarlyStoppingCallback**: Stops training after `patience` evaluations without improvement. Tracks best eval_loss and best checkpoint.

---

## 14. Cost Estimates

### 14.1 Per-Run Costs

| Phase | Resource | Duration | Cost |
|-------|----------|----------|------|
| Data Processing | ml.c5.xlarge (spot) | ~2-4 hours | ~$0.10-0.20 |
| Teacher Generation | Bedrock Claude Sonnet 4.5 | ~1,000 prompts | ~$50-100 |
| Training (Standard) | ml.p4d.24xlarge (spot) | ~4-8 hours | ~$400-600 |
| Training (Budget) | ml.g5.12xlarge (spot) | ~8-16 hours | ~$50-110 |
| Distillation Corrections | Bedrock Claude Sonnet 4.5 | ~500/epoch × 5 epochs | ~$100-200 |
| S3 Storage | Data + checkpoints | Monthly | ~$5-10 |

### 14.2 Instance Pricing

| Instance | GPUs | Hourly (On-Demand) | Hourly (Spot ~70% off) |
|----------|------|--------------------|-----------------------|
| ml.g5.xlarge | 1× A10G | $1.41 | ~$0.42 |
| ml.g5.2xlarge | 1× A10G | $2.82 | ~$0.85 |
| ml.g5.12xlarge | 4× A10G | $7.09 | ~$2.13 |
| ml.p4d.24xlarge | 8× A100 | $40.77 | ~$12.23 |

### 14.3 Total Budget

| Configuration | Estimated Total |
|--------------|----------------|
| **Budget** (g5.12xlarge spot) | ~$200-400 |
| **Standard** (p4d.24xlarge spot) | ~$500-750 |
| **Full** (p4d + extensive distillation) | ~$700-1,000 |

---

## 15. Troubleshooting

### 15.1 AWS Issues

| Problem | Solution |
|---------|----------|
| `NoCredentialsError` | Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` |
| `AccessDenied` on S3 | Check IAM policy includes `s3:GetObject`, `s3:ListBucket` on the bucket |
| `AccessDeniedException` on Bedrock | Enable Claude model access in Bedrock console → Model access |
| `NoSuchEntity` for IAM role | Deploy CloudFormation stack: `aws cloudformation deploy ...` |
| Spot interruption | Job auto-resumes from checkpoint. Increase `max_wait_seconds` |

### 15.2 Training Issues

| Problem | Solution |
|---------|----------|
| NaN/Inf in loss | Auto-detected by callback. Reduce learning rate or check data |
| CUDA Out of Memory | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` |
| Disk space low | Increase `volume_size_gb` or reduce `save_total_limit` |
| Slow training | Enable `gradient_checkpointing`, use `bf16` |
| Early stopping too aggressive | Increase `early_stopping_patience` |

### 15.3 Data Issues

| Problem | Solution |
|---------|----------|
| `UnicodeDecodeError` | Automatic fallback to Latin-1. If still fails, file is skipped |
| Empty dataset | Verify S3 prefixes exist: `aws s3 ls s3://bucket/tsn_data/ --max-items 5` |
| Invalid JSONL | Check `messages` field has `role` and `content` keys |
| Too few examples | Increase `--max-files` or ensure S3 sync completed |

### 15.4 Bedrock Issues

| Problem | Solution |
|---------|----------|
| `ThrottlingException` | Automatic retry with backoff. Reduce `--workers` if persistent |
| `ModelStreamErrorException` | Transient. Automatic retry |
| High latency | Reduce `--max-tokens`, use fewer workers |
| Empty responses | Check prompt format, increase `--max-tokens` |

---

## 16. File Inventory

### 16.1 Scripts

| File | Purpose | Entry Point |
|------|---------|-------------|
| `scripts/dry_run_pipeline.py` | Validate entire pipeline | `PipelineValidator.run_all_tests()` |
| `scripts/prepare_automotive_data.py` | Download & process S3 data locally | `AutomotiveDataPipeline.run_pipeline()` |
| `scripts/generate_teacher_outputs.py` | Generate Claude teacher responses | `BedrockTeacherGenerator.generate_batch()` |
| `scripts/run_training_job.py` | Launch SageMaker QLoRA training | `estimator.fit()` |
| `scripts/launch_cloud_training.py` | Launch SageMaker distillation job | `estimator.fit()` |
| `scripts/launch_processing_job.py` | Launch SageMaker data processing | `processor.run()` |
| `scripts/sagemaker_processing.py` | Data processing (runs inside SageMaker) | `run_pipeline()` |

### 16.2 Training

| File | Purpose |
|------|---------|
| `training/train_granite_qlora.py` | Core QLoRA training script (SageMaker entrypoint) |
| `training/iterative_distillation.py` | Iterative teacher-student distillation loop |

### 16.3 Evaluation

| File | Purpose |
|------|---------|
| `evaluation/__init__.py` | Exports `CodeQualityEvaluator`, `QualityScore` |
| `evaluation/code_quality_metrics.py` | Syntax, protocol, safety, style scoring |

### 16.4 Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Central configuration (model, QLoRA, training, AWS, distillation) |
| `.env` | Environment variables (credentials) |
| `requirements.txt` | Python dependencies |

### 16.5 Infrastructure

| File | Purpose |
|------|---------|
| `aws/cloudformation/sagemaker_stack.yaml` | IAM role, ECR, CloudWatch, SNS |
| `docker/Dockerfile` | Custom training container |

### 16.6 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Quick start guide |
| `PIPELINE_STATE_MACHINE.md` | Mermaid state machine diagrams (69 states, 27 decision points, 35 error states) |
| `FULL_PROCEDURE.md` | This document — complete procedure reference |

### 16.7 Notebooks

| File | Purpose |
|------|---------|
| `notebooks/training.ipynb` | Interactive training notebook (VS Code / Colab compatible) |

### 16.8 Data Directories

| Path | Contents |
|------|----------|
| `data/raw/` | Downloaded source code files from S3 |
| `data/processed/` | Processed training data |
| `data/splits/` | `train.jsonl`, `val.jsonl` |
| `data/teacher_outputs/` | Bedrock Claude responses |
| `models/` | Fine-tuned model output |

---

## End-to-End Execution Checklist

```
[ ] 1. Clone repo, install dependencies, configure .env
[ ] 2. Deploy CloudFormation stack (aws/cloudformation/sagemaker_stack.yaml)
[ ] 3. Run dry run:  python scripts/dry_run_pipeline.py
[ ] 4. Fix any [FAIL] items from dry run
[ ] 5. Prepare data: python scripts/prepare_automotive_data.py --upload-splits
[ ] 6. Verify splits: aws s3 ls s3://granite-8b-training-outputs/runs/data/splits/
[ ] 7. Test teacher: python scripts/generate_teacher_outputs.py --test-mode
[ ] 8. Generate full teacher outputs (if using teacher-seeded data)
[ ] 9. Launch training: python scripts/run_training_job.py --upload-data --dry-run
[ ] 10. Review dry-run config, then launch for real (remove --dry-run)
[ ] 11. Monitor: aws logs tail /aws/sagemaker/TrainingJobs --follow
[ ] 12. Download model: aws s3 cp s3://bucket/output/model.tar.gz ./models/
```
