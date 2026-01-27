# Granite-8B Fine-Tuning for Embedded Automotive Code Generation

Fine-tuning IBM Granite-8B-Code-Instruct-128K for embedded automotive code generation, specifically targeting **AVB (Audio Video Bridging)** and **TSN (Time-Sensitive Networking)** protocols.

## Overview

This project implements a knowledge distillation pipeline using:
- **Teacher Model**: Claude Sonnet via Amazon Bedrock
- **Student Model**: IBM Granite-8B-Code-Instruct-128K
- **Training Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Infrastructure**: AWS SageMaker

## Project Structure

```
fine_tuning_IBM_8B_v2/
├── config.yaml                 # Central configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (credentials)
├── aws/
│   └── cloudformation/
│       └── sagemaker_stack.yaml  # AWS infrastructure
├── docker/
│   └── Dockerfile              # Training container
├── training/
│   └── train_granite_qlora.py  # Core training script
├── scripts/
│   ├── prepare_automotive_data.py    # S3 data processing
│   ├── generate_teacher_outputs.py   # Bedrock Claude teacher
│   ├── run_training_job.py           # SageMaker launcher
│   └── dry_run_pipeline.py           # Pipeline validation
├── data/
│   ├── raw/                    # Downloaded source data
│   ├── processed/              # Processed training data
│   ├── teacher_outputs/        # Claude-generated examples
│   └── splits/                 # train.jsonl, val.jsonl
└── models/                     # Fine-tuned model outputs
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone git@github.com:YOUR_USERNAME/fine_tuning_IBM_8B_v2.git
cd fine_tuning_IBM_8B_v2

# Install dependencies
pip install -r requirements.txt

# Configure environment (edit .env with your credentials)
cp .env.example .env
```

### 2. Run Dry Run Test

**Always run this first to verify the pipeline:**

```bash
python scripts/dry_run_pipeline.py
```

This tests:
- AWS credentials and permissions
- S3 bucket access
- Bedrock model access
- IAM role configuration
- Local file integrity

### 3. Prepare Training Data

```bash
# Download and process automotive data from S3
python scripts/prepare_automotive_data.py --max-files 100

# Or run a small sample for testing
python scripts/prepare_automotive_data.py --sample-only
```

### 4. Generate Teacher Outputs (Knowledge Distillation)

```bash
# Test with sample prompts
python scripts/generate_teacher_outputs.py --test-mode --prompts 5

# Generate full teacher outputs
python scripts/generate_teacher_outputs.py \
    --input-file data/processed/prompts.jsonl \
    --output-file data/teacher_outputs/bedrock_outputs.jsonl
```

### 5. Launch Training Job

```bash
# Dry run (show configuration only)
python scripts/run_training_job.py --dry-run

# Launch training with spot instances
python scripts/run_training_job.py --upload-data

# Launch with specific instance type
python scripts/run_training_job.py --instance-type ml.g5.12xlarge
```

## Configuration

Key settings in `config.yaml`:

```yaml
model:
  name: "ibm-granite/granite-8b-code-instruct-128k"
  max_seq_length: 4096

qlora:
  lora_r: 32
  lora_alpha: 64
  bnb_4bit_compute_dtype: "bfloat16"

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  num_epochs: 5
```

## AWS Infrastructure

Deploy the CloudFormation stack:

```bash
aws cloudformation deploy \
    --template-file aws/cloudformation/sagemaker_stack.yaml \
    --stack-name granite-8b-avb-tsn-finetuning \
    --capabilities CAPABILITY_NAMED_IAM
```

This creates:
- SageMaker execution role with Bedrock access
- ECR repository for training container
- CloudWatch log group
- SNS alert topic

## Data Sources

Training data from S3 bucket: `granite-8b-unified-automotive-data`

- **TSN Data**: Time-Sensitive Networking protocol implementations
- **AVB Data**: Audio Video Bridging code
- **CARLA**: Autonomous driving simulator C++ code

## Cost Estimate

| Resource | Instance/Service | Est. Cost |
|----------|------------------|-----------|
| Teacher Generation | Bedrock Claude Sonnet | ~$50-100 |
| Training (Spot) | ml.p4d.24xlarge | ~$400-600 |
| Processing | ml.m5.xlarge | ~$5 |
| **Total** | | **~$500-750** |

## Troubleshooting

### Bedrock Access Denied
Enable model access in the AWS Bedrock console for Claude models.

### IAM Role Not Found
Deploy the CloudFormation stack first.

### Out of Memory
Reduce batch size or use gradient checkpointing (already enabled).

## License

Proprietary - Excelfore Corporation
