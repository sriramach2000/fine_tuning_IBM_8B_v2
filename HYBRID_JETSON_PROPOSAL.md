# Hybrid Jetson AGX Orin + Cloud: Cost Basis Report & Migration Proposal

**Project**: Granite-8B Fine-Tuning Pipeline Migration
**Author**: Sriram Acharya, Excelfore Corporation
**Date**: February 15, 2026
**Hardware**: NVIDIA Jetson AGX Orin 64GB Developer Kit ($1,999.95/unit)

---

## Executive Summary

This report evaluates migrating the Granite-8B fine-tuning pipeline from a fully cloud-native architecture (AWS SageMaker + Bedrock) to a **hybrid architecture** using NVIDIA Jetson AGX Orin 64GB boards for local data preprocessing and training, while retaining Amazon Bedrock for the teacher model (Claude Sonnet 4.5) in the iterative distillation loop.

**Key Finding**: The Bedrock API costs for the teacher model ($150-$300/run) **dominate total pipeline costs** at 40-60% of each run. Since these API calls cannot be moved off-cloud, the potential savings from local hardware are limited to the data processing and GPU training portions (~$17-$110/run on spot instances). The Jetson AGX Orin is **not cost-optimal for training** compared to either cloud spot instances or desktop GPUs (RTX 4090), but offers unique advantages in power efficiency, form factor, and unified 64GB memory.

**Recommendation**: A **desktop RTX 4090 build** ($3,000-$3,700) provides 15x faster training than a Jetson at the same price point. If edge deployment/inference is a requirement, a hybrid RTX 4090 (training) + single Jetson Orin (inference deployment) is optimal.

---

## Table of Contents

1. [Pipeline Step-by-Step Cost Analysis](#1-pipeline-step-by-step-cost-analysis)
2. [Cloud-Only Cost Baseline](#2-cloud-only-cost-baseline)
3. [Hybrid Architecture: What Moves Local](#3-hybrid-architecture-what-moves-local)
4. [Jetson AGX Orin Hardware Analysis](#4-jetson-agx-orin-hardware-analysis)
5. [Distributed Jetson Cluster Feasibility](#5-distributed-jetson-cluster-feasibility)
6. [Per-Step Cost Comparison: Cloud vs Hybrid](#6-per-step-cost-comparison-cloud-vs-hybrid)
7. [ROI & Break-Even Analysis](#7-roi--break-even-analysis)
8. [Desktop GPU Alternative Comparison](#8-desktop-gpu-alternative-comparison)
9. [Recommendation & Proposed Architecture](#9-recommendation--proposed-architecture)

---

## 1. Pipeline Step-by-Step Cost Analysis

### Current Pipeline Steps (from FULL_PROCEDURE.md)

```
Step 1: Validation (Dry Run)          --> Free (local checks)
Step 2: Data Preparation              --> Cloud or Local (S3 download + processing)
Step 3: Teacher Output Generation     --> CLOUD ONLY (Bedrock Claude Sonnet 4.5)
Step 4: Training Job Launch           --> Cloud or Local (GPU compute)
Step 5: Iterative Distillation        --> HYBRID (GPU local + Bedrock API calls)
```

### Cost Breakdown Per Step (Single Full Pipeline Run)

| Step | What Happens | Cloud Resource | Cloud Cost (Spot) | Cloud Cost (On-Demand) | Can Move Local? |
|------|-------------|---------------|-------------------|----------------------|-----------------|
| **1. Dry Run** | Validate AWS access, config, files | None | $0 | $0 | Already local |
| **2. Data Prep** | Download S3 data, extract functions, generate JSONL | ml.c5.xlarge (2-4hr) | **$0.10-$0.20** | $0.41-$0.82 | **YES** |
| **3. Teacher Gen** | Claude Sonnet generates ~1,000 code examples | Bedrock API | **$50-$100** | $50-$100 | **NO** (API required) |
| **4. QLoRA Training** | Fine-tune Granite-8B on prepared data | ml.g5.12xlarge (8-16hr) | **$17-$34** | $57-$113 | **YES** |
| **5a. Student Inference** | Granite-8B generates eval outputs | Same GPU instance | Included in Step 4 | Included | **YES** |
| **5b. Quality Eval** | Score student outputs (syntax, protocol, MISRA) | CPU on same instance | Included in Step 4 | Included | **YES** |
| **5c. Teacher Corrections** | Claude corrects poor outputs (500/epoch x 5) | Bedrock API | **$34-$83** | $34-$83 | **NO** (API required) |
| **S3 Storage** | Data + checkpoints (~200GB) | S3 Standard | **$4.60/month** | $4.60/month | **YES** (local NVMe) |
| **Monitoring** | CloudWatch logs, metrics, alarms | CloudWatch | **<$1** | <$1 | Partial (local logging) |

### API Cost Deep-Dive: Bedrock Claude Sonnet 4.5

**Verified Pricing (as of Feb 2026):**

| Metric | Price | Source |
|--------|-------|--------|
| Input tokens | **$3.00 / 1M tokens** | [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing), [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) |
| Output tokens | **$15.00 / 1M tokens** | Same sources |
| Batch pricing (50% off) | $1.50 / $7.50 per 1M | Bedrock Batch Inference |
| Prompt caching (90% off reads) | $0.30 / 1M cached input | Bedrock prompt caching |

#### Step 3: Teacher Output Generation (1,000 prompts)

| Component | Tokens/Call | Total Tokens | Cost |
|-----------|------------|-------------|------|
| Input (prompt + system) | ~500 | 500K (0.5M) | $1.50 |
| Output (generated code) | ~1,500 | 1.5M | $22.50 |
| **Subtotal** | | 2.0M | **$24.00** |
| With 10 parallel workers, retries (+20%) | | 2.4M | **~$29** |
| **Realistic range** | | | **$25-$50** |

**Note**: Your FULL_PROCEDURE.md estimates $50-$100 for this step, which accounts for a larger prompt set. The per-call cost is ~$0.024-$0.05.

#### Step 5c: Iterative Distillation Corrections (2,500 calls across 5 epochs)

| Component | Tokens/Call | Total Tokens | Cost |
|-----------|------------|-------------|------|
| Input (prompt + student output + system) | ~650 avg | 1.625M | $4.88 |
| Output (corrected code) | ~1,400 avg | 3.5M | $52.50 |
| **Subtotal** | | 5.125M | **$57.38** |

**Epoch-by-epoch (corrections decrease as student improves):**

| Epoch | Est. Corrections | Input Cost | Output Cost | Epoch Total |
|-------|-----------------|------------|-------------|-------------|
| 1 (worst student) | 500 (100%) | $0.98 | $10.50 | **$11.48** |
| 2 | 400 (80%) | $0.78 | $8.40 | **$9.18** |
| 3 | 300 (60%) | $0.59 | $6.30 | **$6.89** |
| 4 | 200 (40%) | $0.39 | $4.20 | **$4.59** |
| 5 (near convergence) | 100 (20%) | $0.20 | $2.10 | **$2.30** |
| **Total** | **1,500** | **$2.93** | **$31.50** | **$34.43** |

**Range with max 500/epoch**: $34 - $83

---

## 2. Cloud-Only Cost Baseline

### Scenario A: Budget (ml.g5.12xlarge Spot)

| Component | Cost |
|-----------|------|
| Data Processing (ml.c5.xlarge spot, 3hr) | $0.15 |
| Teacher Generation (Bedrock, 1K prompts) | $29.00 |
| Training (ml.g5.12xlarge spot, 12hr) | $25.56 |
| Distillation Corrections (Bedrock, 5 epochs) | $34.43 |
| S3 Storage (1 month) | $4.60 |
| CloudWatch | $0.50 |
| **Total per run** | **$94.24** |

### Scenario B: Standard (ml.p4d.24xlarge Spot)

| Component | Cost |
|-----------|------|
| Data Processing | $0.15 |
| Teacher Generation (Bedrock) | $29.00 |
| Training (ml.p4d.24xlarge spot, 6hr) | $73.38 |
| Distillation Corrections (Bedrock) | $34.43 |
| S3 Storage | $4.60 |
| CloudWatch | $0.50 |
| **Total per run** | **$142.06** |

### Scenario C: Maximum (ml.p4d.24xlarge On-Demand + Full Corrections)

| Component | Cost |
|-----------|------|
| Data Processing | $0.82 |
| Teacher Generation (Bedrock, 2K prompts) | $58.00 |
| Training (ml.p4d.24xlarge on-demand, 8hr) | $326.16 |
| Distillation Corrections (Bedrock, max) | $82.80 |
| S3 Storage | $4.60 |
| **Total per run** | **$472.38** |

### Cost Composition (Budget Scenario)

```
Bedrock API (Teacher + Corrections):  $63.43  (67.3%)  <-- CANNOT MOVE LOCAL
GPU Training Compute:                 $25.56  (27.1%)  <-- CAN MOVE LOCAL
Data Processing:                       $0.15  ( 0.2%)  <-- CAN MOVE LOCAL
Storage + Monitoring:                  $5.10  ( 5.4%)  <-- CAN MOVE LOCAL
                                     -------
Total:                                $94.24  (100%)

Movable to local:                     $30.81  (32.7%)
Stuck on cloud (Bedrock API):         $63.43  (67.3%)
```

**Critical Insight: Only 32.7% of costs can potentially be eliminated by moving to local hardware. The Bedrock API is 67.3% of the total cost.**

---

## 3. Hybrid Architecture: What Moves Local

### Proposed Hybrid Flow

```
LOCAL (Jetson Orin Cluster)               CLOUD (AWS)
┌──────────────────────────┐              ┌────────────────────────┐
│                          │              │                        │
│  1. Data Preparation     │              │                        │
│     - S3 download (once) │◄─── sync ───│  S3 Bucket (source)    │
│     - Function extraction│              │                        │
│     - JSONL generation   │              │                        │
│     - Train/val split    │              │                        │
│                          │              │                        │
│  2. QLoRA Training       │              │                        │
│     - Granite-8B 4-bit   │              │                        │
│     - 5 epochs           │              │                        │
│     - Student inference  │── API call ─►│  3. Bedrock Claude     │
│     - Quality evaluation │              │     Sonnet 4.5         │
│     - Integrate corrects │◄─ response ──│     (Teacher Model)    │
│                          │              │                        │
│  4. Model Export         │              │                        │
│     - LoRA adapters      │── upload ──► │  S3 (model artifacts)  │
│     - Checkpoints        │              │                        │
└──────────────────────────┘              └────────────────────────┘
```

### What Moves Local vs What Stays Cloud

| Component | Where | Reason |
|-----------|-------|--------|
| Data download from S3 | One-time transfer | ~5-50GB, done once |
| Data preprocessing (extract, generate, split) | **LOCAL** | Pure CPU, no cloud needed |
| Teacher output generation | **CLOUD (Bedrock)** | API-only access to Claude |
| QLoRA training (forward/backward pass) | **LOCAL (Jetson GPU)** | GPU compute |
| Student inference (eval generation) | **LOCAL (Jetson GPU)** | Same GPU |
| Quality evaluation (syntax, MISRA, protocol) | **LOCAL (CPU)** | Rule-based checks |
| Teacher corrections (poor outputs) | **CLOUD (Bedrock)** | API-only access to Claude |
| Model checkpoint storage | **LOCAL (NVMe)** | No S3 costs |
| Monitoring | **LOCAL** | Custom logging replaces CloudWatch |

---

## 4. Jetson AGX Orin Hardware Analysis

### Specifications Relevant to Training

| Spec | Jetson AGX Orin 64GB | Impact on Training |
|------|---------------------|-------------------|
| GPU | 2048 CUDA + 64 Tensor Cores (Ampere) | ~15x slower than A100 for training |
| FP16 TFLOPS | **5.3 TFLOPS** (Tensor), 10.6 (CUDA) | Training bottleneck |
| FP32 TFLOPS | **2.6 TFLOPS** | Very slow for FP32 ops |
| INT8 TOPS | 275 (sparse), 138 (dense) | Irrelevant for training |
| Memory | **64GB LPDDR5 (unified CPU+GPU)** | Sufficient for 8B QLoRA (needs ~10-17GB) |
| Memory Bandwidth | **204.8 GB/s (shared)** | 5x lower than RTX 4090, 10x lower than A100 |
| Networking | 10GbE RJ45 | Bottleneck for distributed training |
| Power | 15W - 60W configurable | Extremely efficient |
| CUDA Compute | SM 8.7 | Compatible with PyTorch, slight differences from datacenter SM 8.0 |

### Can It Run QLoRA on Granite-8B?

**Yes.** Memory breakdown:

| Component | Memory Required |
|-----------|----------------|
| Model weights (4-bit NF4) | ~4-5 GB |
| LoRA adapters (rank 32, FP16) | ~0.3 GB |
| Optimizer states (paged AdamW 8-bit) | ~0.5 GB |
| Gradients | ~0.3 GB |
| Activations (batch=2, seq=4096, grad ckpt) | ~4-8 GB |
| **Total** | **~10-14 GB** |
| Available on Jetson (64GB - 6GB OS) | **~58 GB** |
| **Headroom** | **~44-48 GB** |

### Estimated Training Speed on Single Jetson

| Metric | Estimate | Comparison |
|--------|----------|------------|
| Training throughput | ~2-5 tokens/sec | A100: ~60-120 tok/s |
| Time per epoch (10K samples, 4096 seq) | ~8-20 hours | A100: ~30-90 min |
| Time for 5 epochs | **2-5 days** | A100: ~3-8 hours |
| Time for full pipeline (5 epochs + distillation) | **3-7 days** | Cloud: ~12-24 hours |

### Power & Electricity Cost

| Mode | Power | Annual Cost (24/7, $0.12/kWh) | Annual Cost ($0.15/kWh) |
|------|-------|-------------------------------|------------------------|
| MAXN (full training) | 55-60W | $58-$63 | $72-$79 |
| 50W (moderate) | 45-50W | $47-$53 | $59-$66 |
| Idle | 10-15W | $10-$16 | $13-$20 |

**Per-run electricity cost** (single Jetson, 5-day training run):
- 60W x 120 hours = 7.2 kWh x $0.15/kWh = **$1.08**

---

## 5. Distributed Jetson Cluster Feasibility

### Can You Scale TOPS by Networking Multiple Boards?

**Short answer: Not effectively for training. Here's why:**

#### The Networking Bottleneck

| Interconnect | Bandwidth | Gradient Sync Time (8B model, FP16) |
|-------------|-----------|-------------------------------------|
| Jetson 10GbE | 1.25 GB/s | ~13 seconds per step |
| Jetson 1GbE (default) | 125 MB/s | ~128 seconds per step |
| A100 NVLink | 600 GB/s | ~0.03 seconds per step |
| A100 InfiniBand | 200 GB/s | ~0.08 seconds per step |

For **full 8B model** gradient sync across 2 boards: **13 seconds at 10GbE** (each training step takes ~2-5 seconds of compute). Communication overhead **exceeds compute time by 3-6x**, making multi-board scaling **counterproductive**.

#### QLoRA Exception: LoRA Gradients Are Small

With QLoRA (rank 32), only LoRA adapter weights need synchronization:
- LoRA parameters: ~67M params (vs 8B total) = ~134MB in FP16
- Sync at 10GbE: ~0.1 seconds per step
- **This IS practical for 2-4 boards**

#### NCCL Support Issues

Per NVIDIA Developer Forums:
- PyTorch on Jetson **does not ship with NCCL support by default**
- Building PyTorch from source with `USE_NCCL=1` is required
- Multi-node NCCL over Ethernet on Jetson is **experimental and poorly tested**
- The **Gloo backend** is the more reliable option for Jetson distributed training

#### Practical Cluster Sizing

| Cluster Size | Total TOPS (INT8) | Effective Training TFLOPS (FP16) | Scaling Efficiency | Practical? |
|-------------|-------------------|--------------------------------|-------------------|------------|
| 1 board | 275 | 5.3 | 100% | **Yes** |
| 2 boards | 550 | ~8-9 | ~75-85% (with LoRA sync) | **Marginal** |
| 4 boards | 1,100 | ~14-16 | ~65-75% | **Not recommended** |
| 8 boards | 2,200 | ~20-24 | ~45-55% | **Wasteful** |

**Recommendation: 1-2 boards maximum for training. Beyond 2, the overhead isn't worth it.**

---

## 6. Per-Step Cost Comparison: Cloud vs Hybrid

### Single Pipeline Run

| Pipeline Step | Cloud (Spot) | Hybrid (Jetson) | Savings | % Saved |
|--------------|-------------|-----------------|---------|---------|
| Data Preparation | $0.15 | **$0.02** (electricity) | $0.13 | 87% |
| Teacher Generation (Bedrock) | $29.00 | **$29.00** (still Bedrock) | $0.00 | 0% |
| Training (5 epochs) | $25.56 | **$1.08** (electricity) | $24.48 | 96% |
| Distillation Corrections (Bedrock) | $34.43 | **$34.43** (still Bedrock) | $0.00 | 0% |
| Storage | $4.60/mo | **$0** (local NVMe) | $4.60 | 100% |
| Monitoring | $0.50 | **$0** (local logs) | $0.50 | 100% |
| **Total per run** | **$94.24** | **$64.53** | **$29.71** | **31.5%** |

### Time Comparison

| Metric | Cloud (ml.g5.12xlarge) | Hybrid (1x Jetson) | Hybrid (2x Jetson) |
|--------|----------------------|--------------------|--------------------|
| Data Prep | 2-4 hours | 3-6 hours | 3-6 hours |
| Training (5 epochs) | 8-16 hours | **2-5 days** | **1.5-3 days** |
| Distillation (API calls) | ~1-2 hours | ~2-4 hours (higher latency) | ~2-4 hours |
| **Total pipeline** | **~12-24 hours** | **~3-7 days** | **~2-4 days** |

### Annual Cost Comparison (Various Usage Patterns)

| Usage Pattern | Cloud (Spot) Annual | Hybrid Annual (excl. CapEx) | Annual Savings | Years to ROI (2-board) |
|--------------|--------------------|-----------------------------|----------------|----------------------|
| 1 run/month | $1,131 | $774 | $357 | 14.0 years |
| 2 runs/month | $2,262 | $1,549 | $713 | 7.0 years |
| 1 run/week | $4,900 | $3,355 | $1,545 | 3.2 years |
| 2 runs/week | $9,801 | $6,710 | $3,091 | 1.6 years |
| Daily runs | $34,398 | $23,554 | $10,844 | 0.5 years |

*2-board cluster CapEx: $5,000*

---

## 7. ROI & Break-Even Analysis

### Hardware Investment (2-Board Cluster)

| Item | Cost |
|------|------|
| 2x Jetson AGX Orin 64GB Dev Kit | $3,999.90 |
| 10GbE Switch + Cables | $300.00 |
| 2x 1TB NVMe SSD | $260.00 |
| UPS (850VA) | $150.00 |
| Rack/Cooling | $100.00 |
| **Total CapEx** | **$4,809.90** |

### Break-Even Formula

```
Break-Even (months) = CapEx / (Monthly Cloud Savings - Monthly OpEx)

Where:
- Monthly Cloud Savings = Cloud cost/month - Bedrock cost/month (stays same)
- Monthly OpEx = Electricity + Maintenance (~$33/month for 2-board cluster)
```

### Break-Even Table

| Runs/Month | Monthly Cloud Cost | Monthly Bedrock (fixed) | Monthly GPU Savings | Monthly OpEx | Net Monthly Savings | Break-Even |
|-----------|-------------------|------------------------|--------------------|--------------|--------------------|------------|
| 1 | $94 | $63 | $31 | $33 | **-$2** | **Never** |
| 2 | $188 | $127 | $61 | $33 | **$28** | **14.3 years** |
| 4 | $377 | $254 | $123 | $33 | **$90** | **4.5 years** |
| 8 (2x/week) | $754 | $508 | $246 | $33 | **$213** | **1.9 years** |
| 20 (daily) | $1,885 | $1,269 | $616 | $33 | **$583** | **8.2 months** |
| 30 | $2,827 | $1,903 | $924 | $33 | **$891** | **5.4 months** |

### Depreciation & Residual Value

| Year | Book Value (3-yr straight line) | Est. Resale Value | Effective Cost |
|------|--------------------------------|-------------------|---------------|
| 0 (purchase) | $4,810 | $4,810 | $0 |
| 1 | $3,207 | $3,600 | ~$1,210 |
| 2 | $1,603 | $2,400 | ~$2,410 |
| 3 | $0 | $1,200 | ~$3,610 |
| 4 | Fully depreciated | $700 | ~$4,110 |
| 5 | Fully depreciated | $300 | ~$4,510 |

---

## 8. Desktop GPU Alternative Comparison

### Why Consider This

An RTX 4090 desktop build costs roughly the same as **one** Jetson AGX Orin but trains **~15x faster**.

### Hardware Comparison

| Metric | Jetson AGX Orin 64GB | RTX 4090 Desktop | Cloud A10G (ml.g5.xlarge) | Cloud A100 (ml.p4d) |
|--------|---------------------|-------------------|--------------------------|-------------------|
| **Price** | $2,000 (board only) | $3,400 (full build) | $1.41/hr (spot ~$0.50) | $37.69/hr (spot ~$12) |
| **FP16 TFLOPS** | 5.3 | 82.6 | 31.2 | 312 |
| **CUDA Cores** | 2,048 | 16,384 | 9,216 | 6,912 |
| **GPU Memory** | 64GB (shared) | 24GB (dedicated) | 24GB (dedicated) | 80GB (dedicated) |
| **Memory BW** | 204.8 GB/s | 1,008 GB/s | 600 GB/s | 2,039 GB/s |
| **Power** | 60W | 450W | N/A | N/A |
| **QLoRA 8B Time (5 epochs)** | ~2-5 days | **~3-6 hours** | ~8-16 hours | ~3-8 hours |
| **Training $/run (electricity)** | $1.08 | $4.86 | $4-$8 (spot) | $36-$96 (spot) |

### RTX 4090 Build Cost

| Component | Price |
|-----------|-------|
| RTX 4090 (new/used) | $1,600-$2,000 |
| CPU (AMD Ryzen 7 7700X) | $280 |
| Motherboard (B650) | $180 |
| 64GB DDR5 RAM | $170 |
| 1000W PSU (80+ Gold) | $160 |
| 2TB NVMe SSD | $160 |
| Case + CPU Cooler | $150 |
| UPS (1500VA) | $200 |
| **Total** | **$2,900-$3,300** |

### Performance Per Dollar

| Platform | FP16 TFLOPS/$ (CapEx) | QLoRA runs to break even vs cloud |
|----------|----------------------|----------------------------------|
| Jetson AGX Orin 64GB | 0.0027 TFLOPS/$ | ~175 runs (vs g5.12xlarge spot) |
| RTX 4090 Desktop | **0.025 TFLOPS/$** | ~123 runs (vs g5.12xlarge spot) |
| RTX 3090 Used (~$2,100 build) | **0.017 TFLOPS/$** | ~75 runs (vs g5.12xlarge spot) |

---

## 9. Recommendation & Proposed Architecture

### Decision Matrix

| Factor | Jetson Cluster | RTX 4090 Desktop | Cloud (SageMaker) |
|--------|---------------|-------------------|-------------------|
| **Training Speed** | Slow (2-5 days) | Fast (3-6 hours) | Fast (8-16 hours) |
| **Upfront Cost** | $5,000 (2-board) | $3,300 | $0 |
| **Per-Run Cost** | $64 (mostly Bedrock) | $68 (mostly Bedrock) | $94 (Budget spot) |
| **Power Consumption** | 120W (2 boards) | 550W (system) | N/A |
| **Form Factor** | Compact (desk-sized) | Tower PC | N/A |
| **Noise** | Silent/quiet | Moderate (fan) | N/A |
| **Edge Inference** | Excellent | Not portable | N/A |
| **Scale Up** | Very limited | Limited (1 GPU) | Unlimited |
| **Data Privacy** | Full local control | Full local control | AWS has access |
| **Maintenance** | Minimal | Standard PC | Zero |
| **Break-Even** | 1.9 years (2x/week) | 1.3 years (2x/week) | N/A |

### Recommended Architecture: Tiered Approach

#### Tier 1: Immediate (No Hardware Purchase)

Use **ml.g5.xlarge spot** ($0.42-$0.50/hr) instead of ml.g5.12xlarge ($2.13/hr) or ml.p4d.24xlarge ($12.23/hr). Your QLoRA 8B workload fits on a single A10G 24GB GPU.

**Impact**: Reduces training cost from $25-$73 to **$2-$4 per run**. Total pipeline drops to **~$68-$72/run**.

#### Tier 2: Short-Term (If Running >2x/Week)

Build an **RTX 4090 desktop** ($3,300). Train locally, call Bedrock API from your machine.

**Impact**: Eliminates $2-$4 cloud training cost per run. Break-even at ~100-150 runs. Training is actually **faster** than cloud g5.xlarge.

#### Tier 3: Long-Term (If Edge Deployment Needed)

Purchase **1x Jetson AGX Orin 64GB** ($2,000) specifically for:
- Deploying the fine-tuned model for inference at the edge
- Testing inference latency/throughput in the target deployment environment
- Running the model in automotive/embedded prototype systems

**Do NOT use the Jetson for training.** Use it for what it was designed for: edge inference.

### Forward Infrastructure Cost Projection (3 Years)

#### Option A: Cloud Only (Optimized)

| Year | Runs | Cloud Cost | Bedrock Cost | Total |
|------|------|-----------|-------------|-------|
| 1 | 52 (weekly) | $156 | $3,298 | **$3,454** |
| 2 | 104 (2x/week) | $312 | $6,596 | **$6,908** |
| 3 | 156 (3x/week) | $468 | $9,894 | **$10,362** |
| **3-Year Total** | | | | **$20,724** |

#### Option B: RTX 4090 + Cloud Bedrock (Recommended)

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | RTX 4090 Build (CapEx) | $3,300 |
| 1 | Electricity (52 runs x $4.86) | $253 |
| 1 | Bedrock API (52 runs x $63) | $3,298 |
| 1 | **Year 1 Total** | **$6,851** |
| 2 | Electricity (104 runs x $4.86) | $506 |
| 2 | Bedrock API | $6,596 |
| 2 | **Year 2 Total** | **$7,102** |
| 3 | Electricity (156 runs) | $758 |
| 3 | Bedrock API | $9,894 |
| 3 | **Year 3 Total** | **$10,652** |
| | **3-Year Total** | **$24,605** |
| | Less: Residual hardware value | -$1,500 |
| | **Net 3-Year Cost** | **$23,105** |

#### Option C: 2x Jetson + Cloud Bedrock

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | Jetson Cluster (CapEx) | $5,000 |
| 1 | Electricity (52 runs x $1.08) | $56 |
| 1 | Bedrock API | $3,298 |
| 1 | Maintenance | $200 |
| 1 | **Year 1 Total** | **$8,554** |
| 2 | Electricity + Maintenance | $306 |
| 2 | Bedrock API | $6,596 |
| 2 | **Year 2 Total** | **$6,902** |
| 3 | Electricity + Maintenance | $458 |
| 3 | Bedrock API | $9,894 |
| 3 | **Year 3 Total** | **$10,352** |
| | **3-Year Total** | **$25,808** |
| | Less: Residual hardware value | -$1,200 |
| | **Net 3-Year Cost** | **$24,608** |

### 3-Year TCO Summary

| Option | 3-Year Net Cost | Training Speed | Risk |
|--------|----------------|----------------|------|
| **A: Cloud Only (Optimized)** | **$20,724** | Fast (hours) | Spot interruptions |
| **B: RTX 4090 + Bedrock** | **$23,105** | Fast (hours) | Hardware failure |
| **C: Jetson Cluster + Bedrock** | **$24,608** | **Slow (days)** | Experimental software |

### Key Takeaway

**The Bedrock API costs ($63/run) dominate all scenarios.** Reducing API costs via batch pricing (50% off) or prompt caching (up to 90% off for repeated system prompts) will have a **far greater impact** than any hardware choice.

If you can reduce Bedrock costs by 50% via batch processing:
- **$31.50/run API cost** x 312 runs over 3 years = **$9,828** in Bedrock savings
- This exceeds the total savings from any local hardware investment

---

## Sources

- [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) - Claude Sonnet 4.5: $3/1M input, $15/1M output tokens
- [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing) - Confirmed $3/$15 per 1M tokens
- [Claude Sonnet 4.5 Pricing 2026](https://pricepertoken.com/pricing-page/model/anthropic-claude-sonnet-4.5) - $3/$15, 73 tok/s, last updated Feb 15, 2026
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) - ml.g5.12xlarge: $7.09/hr, ml.p4d.24xlarge: $37.69/hr
- [g5.12xlarge specs (Vantage)](https://instances.vantage.sh/aws/ec2/g5.12xlarge) - $5.672/hr EC2, 4x A10G, 96GB VRAM
- [NVIDIA Jetson AGX Orin Datasheet](https://openzeka.com/en/wp-content/uploads/2022/08/Jetson-AGX-Orin-Module-Series-Datasheet.pdf) - 2048 CUDA, 64 Tensor, 5.3 FP16 TFLOPS
- [Jetson AGX Orin TOPS/CUDA Explained (NVIDIA Forums)](https://forums.developer.nvidia.com/t/jetson-agx-orin-tops-cuda-cores-explained/252426) - FP16: 5.3 TFLOPS confirmed
- [Jetson AGX Orin Power Consumption (NVIDIA Forums)](https://forums.developer.nvidia.com/t/agx-orin-power-consumption/223580) - 15-60W configurable
- [PyTorch NCCL on Jetson Issues (NVIDIA Forums)](https://forums.developer.nvidia.com/t/pytorch-is-not-compiled-with-nccl-support/262529) - NCCL not included by default
- [RTX 4090 vs Jetson AGX Orin](https://www.lowtouch.ai/rtx-vs-jetson-agx-orin/) - FP16: 82.6 vs 5.3 TFLOPS
- [Jetson AGX Orin LLM Performance (DFRobot)](https://www.dfrobot.com/blog-13496.html) - LLaMA2-7b at 1.3-1.8 tok/s inference
- [RTX 4090 Price Tracker](https://bestvaluegpu.com/history/new-and-used-rtx-4090-price-history-and-specs/) - Current $1,983-$3,590 new
- [RTX 4090 for AI Training (Fluence)](https://www.fluence.network/blog/nvidia-rtx-4090/) - MSRP $1,599, rivals datacenter GPUs
- [Jetson LLM Fine-tuning Guide (Medium)](https://medium.com/@michaelyuan_88928/how-to-set-up-your-jetson-device-for-llm-inference-and-fine-tuning-682e36444d43)
