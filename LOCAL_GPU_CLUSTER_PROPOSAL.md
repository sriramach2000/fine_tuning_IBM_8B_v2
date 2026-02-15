# Local GPU Workstation: Cost Basis Report & ROI Analysis

**Project**: Granite-8B Fine-Tuning Pipeline — Local Infrastructure Migration
**Author**: Sriram Acharya, Excelfore Corporation
**Date**: February 15, 2026
**Hardware**: NVIDIA GeForce RTX 5090 32GB (Blackwell Architecture)
**Recommendation**: **BUILD — Single RTX 5090 Office Workstation**

---

## Executive Summary

This report evaluates replacing cloud GPU compute (AWS SageMaker / EC2) with a **local office GPU workstation** using an NVIDIA RTX 5090. This document provides the **final recommended build**, complete bill of materials, self-contained cooling infrastructure, and ROI analysis.

### The Build

| | |
|---|---|
| **GPU** | NVIDIA RTX 5090 32GB GDDR7 — 210 TFLOPS FP16, 1,792 GB/s bandwidth |
| **Total Build Cost** | **$5,745-$7,845** (with GPU AIO liquid cooling) |
| **Training Time** | **2-4 hours** per full pipeline run (5 epochs QLoRA Granite-8B) |
| **Full Pipeline Time** | **~6-12 hours** end-to-end (incl. Bedrock API calls) |
| **Per-Run Cost** | **~$65** (electricity $0.69 + Bedrock API $63.43 + internet $0.50) |
| **Per-Run Savings vs Cloud** | **$24/run** vs ml.g5.12xlarge spot, **$60/run** vs ml.p4d.24xlarge spot |
| **Break-Even** | **~2.7 years** at 2 runs/week vs g5.12xlarge spot, **~13 months** vs p4d spot |
| **3-Year TCO** | **$24,360** (vs $27,948 cloud g5.12xlarge — saves ~$3,588) |
| **Cooling** | Self-contained GPU AIO liquid cooling — no external AC needed |
| **Office-Friendly** | Yes — 30-40 dBA, standard 15A circuit, fits under desk |

### Why RTX 5090

1. **32GB VRAM** — QLoRA 8B uses ~12-16GB, leaving 16-20GB headroom for larger batches (batch=4-8) and future models up to 20B.
2. **210 TFLOPS FP16** — 2-4x faster than cloud A10G instances, competitive with A100. Completes 5 training epochs in 2-4 hours.
3. **$1,999 MSRP** ($3,400-$5,500 street) — best performance-per-dollar of any current GPU for this workload.
4. **Single GPU simplicity** — no multi-GPU complexity, no NVLink needed, no tensor parallelism. One GPU, one training script, it just works.
5. **Bedrock API is 67% of every cloud run's cost** — the GPU choice only affects the other 33%. A single RTX 5090 eliminates that 33% almost entirely ($0.69 electricity vs $25+ cloud compute), making Bedrock ~98% of the local per-run cost.

### Why NOT Dual GPU / PRO 6000

| Option | Verdict | Reason |
|--------|---------|--------|
| **Dual RTX 5090** | Not needed for 8B | Saves ~1-2 hrs/run but costs $5,000+ more. No NVLink = PCIe gradient sync overhead. Only justified if running daily+ or parallel experiments. |
| **RTX PRO 6000 Blackwell (96GB)** | Overkill for 8B | $8,000 GPU alone. Only justified if scaling to 34B-70B QLoRA models. |
| **RTX 5080 (16GB)** | Too tight | 0-4GB headroom. OOM risk at batch>2 or seq>4096. Valid as Phase 1 only if 5090 unavailable. |

---

## Table of Contents

1. [GPU Hardware Options & Specifications](#1-gpu-hardware-options--specifications)
2. [Workstation Build Configurations & Costs](#2-workstation-build-configurations--costs)
3. [Cloud Cost Baseline (Updated)](#3-cloud-cost-baseline-updated)
4. [Per-Run Cost Comparison: Local vs Cloud](#4-per-run-cost-comparison-local-vs-cloud)
5. [ROI & Break-Even Analysis](#5-roi--break-even-analysis)
6. [Affordable Office Setup (Best ROI)](#6-affordable-office-setup-best-roi)
7. [Power, Cooling & Practical Office Considerations](#7-power-cooling--practical-office-considerations)
8. [Depreciation & Residual Value](#8-depreciation--residual-value)
9. [3-Year TCO Comparison](#9-3-year-tco-comparison)
10. [Recommendation: Build the RTX 5090 Workstation](#10-recommendation-build-the-rtx-5090-workstation)
11. [Full RTX 5090 Workstation Build (Bill of Materials)](#11-full-rtx-5090-workstation-build-bill-of-materials)

---

## 1. GPU Hardware Options & Specifications

### Current-Generation GPU Comparison (February 2026)

| Specification | RTX 5090 (Consumer) | RTX PRO 6000 Blackwell (Workstation) | RTX 4090 (Previous Gen) | Cloud A100 80GB | Cloud H100 80GB |
|--------------|---------------------|--------------------------------------|------------------------|-----------------|-----------------|
| **Street Price** | **$3,400-$5,500** | **$7,999** | **$1,600-$2,000** | $12.23/hr spot (p4d) | $22/hr spot (p5) |
| **MSRP** | $1,999 | $8,565 | $1,599 | N/A | N/A |
| **Architecture** | Blackwell | Blackwell | Ada Lovelace | Ampere | Hopper |
| **CUDA Cores** | 21,760 | 24,064 | 16,384 | 6,912 | 14,592 |
| **Tensor Cores** | 680 (5th Gen) | 752 (5th Gen) | 512 (4th Gen) | 432 (3rd Gen) | 528 (4th Gen) |
| **VRAM** | **32 GB GDDR7** | **96 GB GDDR7 ECC** | **24 GB GDDR6X** | **80 GB HBM2e** | **80 GB HBM3** |
| **Memory Bandwidth** | 1,792 GB/s | 1,792 GB/s | 1,008 GB/s | 2,039 GB/s | 3,352 GB/s |
| **FP16/BF16 (Tensor Core, dense)** | ~210 TFLOPS | **252 TFLOPS** | ~165 TFLOPS* | 312 TFLOPS | ~495 TFLOPS** |
| **FP32** | 125 TFLOPS | 126 TFLOPS | 82.6 TFLOPS | 19.5 TFLOPS | 67 TFLOPS |
| **TDP** | 575W | 600W | 450W | N/A | N/A |
| **NVLink (workstation)** | No | No | No | Yes (600 GB/s) | Yes (900 GB/s) |
| **ECC Memory** | No | Yes | No | Yes | Yes |
| **PCIe** | Gen 5 x16 | Gen 5 x16 | Gen 4 x16 | Gen 4 x16 | Gen 5 x16 |

*\*RTX 4090 FP16: 82.6 TFLOPS shader, ~165 TFLOPS Tensor Core dense (330 TFLOPS with sparsity). Table uses Tensor Core dense for consistency with other GPUs.*
*\*\*H100 FP16: ~495 TFLOPS dense (989 TFLOPS with sparsity). Table uses dense for consistency across all GPUs.*

### Can Each GPU Run QLoRA on Granite-8B?

| Component | Memory Required | RTX 5090 (32GB) | RTX PRO 6000 (96GB) | RTX 4090 (24GB) |
|-----------|----------------|-----------------|---------------------|-----------------|
| Model weights (4-bit NF4) | ~4-5 GB | Yes | Yes | Yes |
| LoRA adapters (rank 32, FP16) | ~0.3 GB | Yes | Yes | Yes |
| Optimizer states (paged AdamW 8-bit) | ~0.5 GB | Yes | Yes | Yes |
| Gradients | ~0.3 GB | Yes | Yes | Yes |
| Activations (batch=4, seq=4096, grad ckpt) | ~6-10 GB | Yes | Yes | Yes |
| **Total** | **~12-16 GB** | **16 GB headroom** | **80 GB headroom** | **8 GB headroom** |
| Larger batch (batch=8) | ~18-24 GB | Possible | Yes | **Tight** |

**All three GPUs can run QLoRA on Granite-8B.** The RTX 5090 has significant headroom for larger batch sizes, longer sequences, and models up to ~20B QLoRA. The RTX PRO 6000 (96GB) extends this to 70B QLoRA models.

### Estimated Training Speed (QLoRA Granite-8B, 5 epochs, 10K samples)

| GPU | Est. Throughput (tok/s) | Time per Epoch | Time for 5 Epochs | Full Pipeline |
|-----|------------------------|----------------|-------------------|---------------|
| RTX 4090 | ~40-80 | 1-2 hours | 3-6 hours | ~6-10 hours |
| RTX 5090 | ~80-150 | 30-60 min | **2-4 hours** | **~4-7 hours** |
| RTX PRO 6000 Blackwell | ~90-170 | 25-50 min | **2-3.5 hours** | **~4-6 hours** |
| 2x RTX 5090 (data parallel) | ~140-260 | 20-35 min | **1.5-2.5 hours** | **~3-5 hours** |
| Cloud A10G (ml.g5.xlarge) | ~25-50 | 2-4 hours | 8-16 hours | ~12-24 hours |
| Cloud A100 (ml.p4d) | ~60-120 | 30-90 min | 3-8 hours | ~6-12 hours |

**Key insight**: A single RTX 5090 is **faster** than a cloud A10G (ml.g5.xlarge) and competitive with a single A100. For single-GPU QLoRA workloads, multi-GPU instances waste capacity: a g5.12xlarge (4x A10G) and p4d (8x A100) each use only 1 of their GPUs — meaning you pay for 4 or 8 GPUs but use 1. This is why g5.12xlarge shows the same training time as g5.xlarge despite costing ~4x more per hour. A dual RTX 5090 setup matches that single-A100 throughput at a tiny fraction of the hourly cost.

---

## 2. Workstation Build Configurations & Costs

### Option A: Base Office Build — Single RTX 5090 ($5,100 - $6,200)

**Base configuration without GPU liquid cooling. See [Section 11](#11-full-rtx-5090-workstation-build-bill-of-materials) for the final recommended build ($5,745-$7,845 with GPU AIO cooling, different CPU/motherboard/case optimized for sustained AI workloads).**

| Component | Specification | Price |
|-----------|--------------|-------|
| GPU | NVIDIA RTX 5090 32GB (street price) | $3,400-$4,500 |
| CPU | AMD Ryzen 9 7900X (12-core) | $350 |
| Motherboard | ASUS ProArt X670E-Creator (PCIe 5.0) | $250 |
| RAM | 64GB DDR5-5600 (2x32GB) | $170 |
| Storage | 2TB PCIe Gen4 NVMe SSD | $150 |
| PSU | Corsair HX1500i (1500W, 80+ Platinum) | $280 |
| Case | Fractal Design Torrent (excellent airflow) | $190 |
| CPU Cooler | Noctua NH-D15 | $100 |
| UPS | CyberPower 1500VA / 900W | $200 |
| **Total** | | **$5,090-$6,190** |

**Realistic estimate accounting for RTX 5090 street pricing: ~$5,500**

### Option B: Mid-Range Build — Dual RTX 5090 ($10,600 - $12,800)

**For parallel experiments or future-proofing.**

| Component | Specification | Price |
|-----------|--------------|-------|
| GPUs | 2x NVIDIA RTX 5090 32GB (64GB total) | $6,800-$9,000 |
| CPU | AMD Ryzen Threadripper 7960X (24-core) | $800 |
| Motherboard | ASUS Pro WS WRX90E-SAGE (4x PCIe 5.0 x16) | $900 |
| RAM | 128GB DDR5 ECC (4x32GB) | $400 |
| Storage | 4TB PCIe Gen4 NVMe (2x2TB) | $300 |
| PSU | be quiet! Dark Power Pro 13 2000W | $400 |
| Case | be quiet! Dark Base Pro 901 | $250 |
| CPU Cooler | Noctua NH-U14S TR5 | $120 |
| UPS | APC Smart-UPS 2200VA | $650 |
| **Total** | | **$10,620-$12,820** |

**Realistic estimate: ~$11,000**

### Option C: Professional Build — Single RTX PRO 6000 Blackwell ($12,000 - $15,000)

**For 70B+ models, professional drivers, ECC memory.**

| Component | Specification | Price |
|-----------|--------------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell 96GB | $7,999 |
| CPU | AMD Threadripper PRO 7965WX (24-core) | $1,500 |
| Motherboard | ASUS Pro WS WRX90E-SAGE | $900 |
| RAM | 256GB DDR5 ECC (8x32GB) | $900 |
| Storage | 4TB PCIe Gen5 NVMe | $400 |
| PSU | EVGA SuperNOVA 2000 P6 (2000W) | $450 |
| Case | Fractal Design Define 7 XL | $220 |
| CPU Cooler | Noctua NH-U14S TR5 | $120 |
| UPS | APC Smart-UPS 3000VA | $1,200 |
| **Total** | | **$13,689** |

**Realistic estimate: ~$14,000**

### Option D: RTX 4090 Build — Previous Gen Budget ($2,900 - $3,300)

**Still viable if RTX 5090 is hard to find.**

| Component | Specification | Price |
|-----------|--------------|-------|
| GPU | RTX 4090 (used market / new remaining stock) | $1,600-$2,000 |
| CPU | AMD Ryzen 7 7700X | $280 |
| Motherboard | B650 | $180 |
| RAM | 64GB DDR5 | $170 |
| Storage | 2TB NVMe | $160 |
| PSU | 1000W 80+ Gold | $160 |
| Case + Cooler | | $150 |
| UPS | 1500VA | $200 |
| **Total** | | **$2,900-$3,300** |

### Option E: Pre-Built AI Workstation (Lambda Labs / Puget Systems)

| Vendor | Configuration | Price |
|--------|--------------|-------|
| Lambda Scalar (1x RTX 5090) | Ryzen 9, 64GB, 2TB NVMe | ~$7,000-$9,000 |
| Lambda Vector (4x GPU) | Threadripper, 256GB, 8TB | ~$30,000-$60,000 |
| Puget Systems (1x RTX PRO 6000 BW) | Threadripper, 128GB | ~$15,000-$20,000 |
| BOXX APEXX T4 (2x RTX PRO 6000 BW) | Threadripper PRO, 256GB | ~$28,000-$35,000 |

**Pre-builts carry a 30-60% markup** over DIY but include warranty, support, and professional assembly. Not recommended unless hardware expertise is unavailable.

---

## 3. Cloud Cost Baseline (Updated)

### Relevant AWS Instances for Granite-8B QLoRA

| Instance | GPUs | VRAM | SageMaker $/hr | EC2 On-Demand $/hr | EC2 Spot $/hr (est.) |
|----------|------|------|----------------|--------------------|--------------------|
| **ml.g5.xlarge** | 1x A10G 24GB | 24 GB | $1.41 | $1.006 | **$0.42-$0.50** |
| **ml.g5.12xlarge** | 4x A10G 24GB | 96 GB | $7.09 | $5.672 | **$2.13** |
| **ml.p4d.24xlarge** | 8x A100 80GB | 640 GB | $37.69 | $21.96 | **$12.23** |
| **ml.p5.48xlarge** | 8x H100 80GB | 640 GB | $63.30 | $55.04 | **$22.00** |

### Per-Run Cloud Costs (Single Full Pipeline)

| Scenario | Instance | Training Time | Training Cost | Bedrock API | Total/Run |
|----------|----------|---------------|---------------|-------------|-----------|
| **Budget (Spot)** | ml.g5.xlarge spot | 8-16 hrs | **$3.36-$8.00** | $63.43 | **$67-$71** |
| **Standard (Spot)** | ml.g5.12xlarge spot | 8-16 hrs | **$17-$34** | $63.43 | **$80-$97** |
| **Performance (Spot)** | ml.p4d.24xlarge spot | 3-8 hrs | **$37-$98** | $63.43 | **$100-$161** |
| **Maximum (On-Demand)** | ml.p4d.24xlarge OD | 3-8 hrs | **$66-$176** | $63.43 | **$129-$239** |

*S3 storage and CloudWatch monitoring add ~$61/year as fixed overhead (not included in per-run totals).*

**Note**: The budget scenario (ml.g5.xlarge spot) is the most cost-effective cloud option for QLoRA 8B.

---

## 4. Per-Run Cost Comparison: Local vs Cloud

### Electricity Cost Per Run (Local GPU)

| GPU Config | Power Draw (system) | Training Time | kWh | Electricity Cost ($0.30/kWh) |
|------------|-------------------|---------------|-----|------------------------------|
| 1x RTX 4090 | ~550W | 5 hrs | 2.75 | **$0.83** |
| 1x RTX 5090 | ~770W | 3 hrs | 2.31 | **$0.69** |
| 2x RTX 5090 | ~1,350W | 2 hrs | 2.70 | **$0.81** |
| 1x RTX PRO 6000 BW | ~800W | 2.75 hrs | 2.20 | **$0.66** |

### Full Pipeline Cost Per Run

| Configuration | Training Cost | Bedrock API (fixed) | Other | **Total/Run** |
|--------------|--------------|--------------------|-----------------------|--------------|
| **Cloud: ml.g5.xlarge spot** | $5.00 | $63.43 | — | **$68.43** |
| **Cloud: ml.g5.12xlarge spot** | $25.56 | $63.43 | — | **$88.99** |
| **Cloud: ml.p4d.24xlarge spot** | $61.15 | $63.43 | — | **$124.58** |
| **Local: 1x RTX 4090** | $0.83 | $63.43 | $0.50 (internet) | **$64.76** |
| **Local: 1x RTX 5090** | $0.69 | $63.43 | $0.50 | **$64.62** |
| **Local: 2x RTX 5090** | $0.81 | $63.43 | $0.50 | **$64.74** |
| **Local: 1x RTX PRO 6000 BW** | $0.66 | $63.43 | $0.50 | **$64.59** |

*Note: Cloud S3 storage and CloudWatch monitoring are fixed overhead (~$61/year) regardless of run count, so they are excluded from the per-run comparison and accounted for separately in the 3-Year TCO (Section 9).*

### Per-Run Savings (Local vs Cloud)

| Local Config | vs g5.xlarge spot | vs g5.12xlarge spot | vs p4d.24xlarge spot |
|-------------|-------------------|--------------------|--------------------|
| **1x RTX 5090** | **$3.81/run** | **$24.37/run** | **$59.96/run** |
| **2x RTX 5090** | **$3.69/run** | **$24.25/run** | **$59.84/run** |
| **1x RTX PRO 6000 BW** | **$3.84/run** | **$24.40/run** | **$59.99/run** |

**Critical observation**: Even the cheapest cloud option (g5.xlarge spot) costs ~$4 more per run than local. This per-run gap is pure GPU compute savings — the Bedrock API is identical in all scenarios. Against the g5.12xlarge (the original pipeline), savings are ~$24/run.

---

## 5. ROI & Break-Even Analysis

### Break-Even Formula

```
Break-Even (months) = CapEx / (Monthly Savings - Monthly Electricity Overhead)

Where:
- Monthly Savings = (Cloud cost/run - Local cost/run) x runs/month
- Monthly Electricity Overhead = Idle power draw (if kept running)
  Assume workstation is only on during training: overhead ≈ $0
  Assume workstation runs 24/7 for availability: overhead ≈ $25-$50/month
```

### Break-Even vs ml.g5.xlarge Spot (Budget Cloud — $3.81 savings/run)

| Runs/Month | Monthly Savings | RTX 5090 ($6,850) | 2x RTX 5090 ($11,000) | RTX PRO 6000 ($14,000) | RTX 4090 ($3,300) |
|-----------|----------------|---------------------------|-------------------------------|--------------------------------|---------------------------|
| 4 (1x/week) | $15.24 | **37.5 years** | Never | Never | **18.7 years** |
| 8 (2x/week) | $30.48 | **18.7 years** | 31.1 years | Never | **9.4 years** |
| 20 (5x/week) | $76.20 | **7.5 years** | 12.4 years | 15.2 years | **3.7 years** |
| 30 (daily) | $114.30 | **5.0 years** | 8.3 years | 10.1 years | **2.5 years** |

**Against the cheapest cloud option, ROI is very slow** because the savings per run are only ~$4. This is the honest reality — if you're already using g5.xlarge spot instances optimally, the per-run savings from local hardware are minimal. The case for local hardware rests on comparison to g5.12xlarge or p4d instances.

### Break-Even vs ml.g5.12xlarge Spot (Standard Cloud — $24.37 savings/run)

This is the more realistic comparison if you're using 4x A10G instances as originally spec'd in FULL_PROCEDURE.md.

| Runs/Month | Monthly Savings | RTX 5090 ($6,850) | 2x RTX 5090 ($11,000) | RTX PRO 6000 ($14,000) | RTX 4090 ($3,300) |
|-----------|----------------|---------------------------|-------------------------------|--------------------------------|---------------------------|
| 2 | $48.74 | **11.7 years** | 18.9 years | Never | **5.7 years** |
| 4 (1x/week) | $97.48 | **5.9 years** | 9.5 years | 12.0 years | **2.8 years** |
| 8 (2x/week) | $194.96 | **2.9 years** | 4.7 years | 6.0 years | **1.4 years** |
| 12 (3x/week) | $292.44 | **2.0 years** | 3.1 years | 4.0 years | **11.3 months** |
| 20 (5x/week) | $487.40 | **1.2 years** | 1.9 years | 2.4 years | **6.8 months** |
| 30 (daily) | $731.10 | **9.4 months** | 1.3 years | 1.6 years | **4.5 months** |

### Break-Even vs ml.p4d.24xlarge Spot (Performance Cloud — $59.96 savings/run)

| Runs/Month | Monthly Savings | RTX 5090 ($6,850) | 2x RTX 5090 ($11,000) | RTX PRO 6000 ($14,000) | RTX 4090 ($3,300) |
|-----------|----------------|---------------------------|-------------------------------|--------------------------------|---------------------------|
| 4 (1x/week) | $239.84 | **2.4 years** | 3.8 years | 4.9 years | **1.1 years** |
| 8 (2x/week) | $479.68 | **1.2 years** | 1.9 years | 2.4 years | **6.9 months** |
| 12 (3x/week) | $719.52 | **9.5 months** | 1.3 years | 1.6 years | **4.6 months** |
| 20 (5x/week) | $1,199.20 | **5.7 months** | 7.7 months | 9.7 months | **2.8 months** |
| 30 (daily) | $1,798.80 | **3.8 months** | 5.1 months | 6.5 months | **1.8 months** |

---

## 6. Affordable Office Setup (Best ROI)

### The Question: Can You Build Something Affordable That Beats Cloud ROI?

**Yes.** Here's how, broken down by budget tier:

### Tier 1: Ultra-Budget — RTX 4090 Build ($3,300)

If you can find an RTX 4090 (previous gen, still excellent):

| Metric | Value |
|--------|-------|
| **Total build cost** | $3,300 |
| QLoRA Granite-8B training time | 3-6 hours |
| Per-run savings vs g5.12xlarge spot | $24.23/run |
| **Break-even at 2x/week** | **1.4 years** |
| **Break-even at daily** | **4.5 months** |
| VRAM | 24 GB (tight but sufficient for 8B QLoRA) |
| Risk | Stock dwindling, no new production |

### Tier 2: Best Value — Single RTX 5090 Build ($6,850)

The sweet spot for an affordable office setup (includes GPU AIO liquid cooling — see Section 11 BOM):

| Metric | Value |
|--------|-------|
| **Total build cost** | ~$6,850 (with AIO cooling) |
| QLoRA Granite-8B training time | 2-4 hours (faster than cloud g5.xlarge) |
| Per-run savings vs g5.12xlarge spot | $24.37/run |
| **Break-even at 2x/week** | **2.7 years** |
| **Break-even at 3x/week** | **1.8 years** |
| **Break-even at daily** | **9.4 months** |
| VRAM | 32 GB (comfortable headroom, can try 13B-20B models too) |
| Training speed vs cloud | **2-4x faster than g5.xlarge, competitive with p4d** |
| Power at wall | ~770W during training |
| Monthly electricity (2 runs/week, ~6 hrs each) | ~$7.20 |

### Tier 3: Parallel Experiments — Dual RTX 5090 ($11,000)

For running training + inference simultaneously, or halving training time:

| Metric | Value |
|--------|-------|
| **Total build cost** | ~$11,000 |
| QLoRA Granite-8B training time | 1.5-2.5 hours |
| Total VRAM | 64 GB across 2 GPUs |
| **Break-even at 2x/week vs g5.12xlarge** | **4.7 years** |
| **Break-even at daily vs g5.12xlarge** | **1.3 years** |
| **Break-even at 2x/week vs p4d** | **1.9 years** |

### The Honest Assessment: When Local Makes Sense

```
                        Cloud is cheaper    |    Local is cheaper
                     (within 3-year life)   | (within 3-year life)
                        ◄──────────────────►|◄──────────────────►
                                            |
RTX 4090 ($3,300)      <4 runs/month       |    >4 runs/month      (vs g5.12xlarge)
RTX 5090 ($6,850)      <8 runs/month       |    >8 runs/month      (vs g5.12xlarge)
2x RTX 5090 ($11,000)  <13 runs/month      |    >13 runs/month     (vs g5.12xlarge)
RTX PRO 6000 ($14,000) <16 runs/month      |    >16 runs/month     (vs g5.12xlarge)

RTX 4090 ($3,300)      <2 runs/month       |    >2 runs/month      (vs p4d.24xlarge)
RTX 5090 ($6,850)      <3 runs/month       |    >3 runs/month      (vs p4d.24xlarge)
```

**If you're running the pipeline more than twice a week against g5.12xlarge (or more than once every two weeks against p4d), local hardware pays for itself within the useful life of the GPU.**

---

## 7. Power, Cooling & Practical Office Considerations

### Power Requirements

| Configuration | GPU TDP | System Draw (at wall) | Circuit Required | Standard US Office? |
|--------------|---------|----------------------|-----------------|-------------------|
| 1x RTX 4090 | 450W | ~550-650W | 15A/120V (standard) | **Yes** |
| 1x RTX 5090 | 575W | ~700-800W | 15A/120V (standard) | **Yes** (on dedicated circuit) |
| 2x RTX 5090 | 1,150W | ~1,300-1,500W | **20A/120V or 15A/240V** | **Needs dedicated circuit** |
| 1x RTX PRO 6000 BW | 600W | ~750-850W | 15A/120V (standard) | **Yes** (on dedicated circuit) |

**Standard US office outlet**: 15A at 120V = 1,800W max (derate to 1,440W continuous at 80% rule).
- Single-GPU builds: Fine on a standard circuit
- Dual-GPU builds: Need a dedicated 20A circuit or separate circuits for workstation + monitors

### Heat Output & Room Impact

| Configuration | Heat (BTU/hr) | Equivalent to... | Room Temp Rise (10x12 office) |
|--------------|---------------|------------------|-------------------------------|
| 1x RTX 5090 system | ~2,700 BTU/hr | A small space heater | +3-5°F |
| 2x RTX 5090 system | ~5,100 BTU/hr | A large space heater | +7-10°F |
| 1x RTX PRO 6000 BW system | ~2,900 BTU/hr | A small space heater | +4-6°F |

### Self-Contained Cooling Infrastructure

**Since the workstation will be assembled and operated in-office, a self-contained cooling strategy is essential.** Below are the options ranked by cost and practicality.

#### Option 1: GPU AIO Liquid Cooling + High-Airflow Case (Recommended — $530-$650)

The most practical self-contained approach: liquid-cool the GPU to keep heat managed at the source.

| Component | Product | Cost |
|-----------|---------|------|
| GPU AIO Cooler | **LYNK+ Modular AIO for RTX 5090** (360mm rad, drip-free quick-disconnect) | ~$380 |
| | *Alternative*: Alphacool Eiswolf AIO RTX 5090 | ~$300-$350 |
| High-Airflow Case | Fractal Design Torrent (best-in-class airflow, fits 360mm rad) | ~$190 |
| Additional Fans | 2-3x 120mm case fans (if not included) | ~$30-$60 |
| **Total** | | **$530-$650** |

**Benefits**: Cuts GPU temps by ~25°C, reduces noise by ~6 dBA vs stock air cooler. GPU stays under 60°C even during sustained multi-hour training runs. No external cooling infrastructure needed.

**Maintenance**: AIO is sealed — zero maintenance for 5+ years. No coolant changes needed.

*Note: LYNK+ currently supports reference PCB cards from Inno3D, MSI, Palit, Gainward, PNY, and ZOTAC. Check compatibility with your specific card.*

#### Option 2: Full Custom Loop Liquid Cooling ($900-$1,800)

For maximum cooling headroom, especially if planning to upgrade to dual-GPU later.

| Component | Budget | Mid-Range | High-End |
|-----------|--------|-----------|----------|
| GPU Water Block (EKWB, Alphacool, or Bykski) | $200-$210 | $300-$342 | $380-$400 |
| CPU Water Block | $60-$80 | $100-$150 | $150-$200 |
| Pump + Reservoir (combo unit) | $100-$150 | $150-$200 | $200-$300 |
| Radiators (2x 360mm recommended) | $80-$120 | $120-$200 | $200-$300 |
| Fittings (12-16 pieces) | $60-$100 | $100-$150 | $150-$250 |
| Tubing (soft or hardline) | $20-$40 | $30-$60 | $40-$80 |
| Coolant + Biocide | $15-$25 | $20-$35 | $25-$50 |
| Fans (6x 120mm for 2 radiators) | $50-$80 | $80-$150 | $120-$200 |
| **Total** | **$585-$805** | **$900-$1,287** | **$1,265-$1,780** |

**Maintenance**: Coolant change every 6-12 months (~$30 + 1-2 hours labor). Annual loop inspection recommended.

**GPU water blocks available for RTX 5090**: EKWB EK-Quantum Vector3 FE (~$342), Alphacool Core (~$210), Alphacool ES 1-slot (~$315).

**GPU water blocks for RTX PRO 6000 Blackwell**: Watercool HEATKILLER INOX Pro (~$380), Bykski N-RTXPRO6000-WS-SR (~$250), Optimus PC block (~$350), EK-Pro GPU WB (~$350+).

#### Option 3: Portable Spot AC for the Office ($300-$1,500)

If you want to keep stock air cooling on the GPU but need to manage room heat:

| Unit | BTU Rating | Price | Best For |
|------|-----------|-------|----------|
| Consumer portable AC (Midea, LG, Whynter) | 10,000-14,000 BTU | $300-$600 | Budget — occasional training |
| **Tripp Lite SRCOOL12K** (server-room grade) | 12,000 BTU (3.5 kW) | $1,130-$1,444 | **Sustained training — recommended** |
| Tripp Lite SRCOOL12KE (Gen 2) | 13,000 BTU (3.8 kW) | $1,200-$1,500 | Newer, more efficient |

**Sizing guide** (1W = 3.412 BTU/hr, size AC at 1.5-2x raw need):

| Workstation Heat | BTU/hr Generated | Recommended AC |
|-----------------|-----------------|----------------|
| ~770W (1x RTX 5090) | 2,627 BTU/hr | 5,000-8,000 BTU (budget consumer unit works) |
| ~1,350W (2x RTX 5090) | 4,606 BTU/hr | 10,000-12,000 BTU |
| ~1,500W (multi-GPU workstation) | 5,118 BTU/hr | 12,000-14,000 BTU |

**Important**: All portable AC units require an exhaust hose to a window or ceiling plenum. There is no truly ductless portable AC. If you cannot access a window, consider a mini-split system ($700-$2,000 + $500-$1,500 installation, requires a 3-inch hole in an exterior wall).

#### Option 4: Rackmount with Integrated Cooling ($2,500-$8,000)

For a more "server-like" setup that isolates noise and heat:

| Solution | Price | Notes |
|----------|-------|-------|
| 42U Rack + 4U Chassis (SilverStone RM44) + Portable AC | $2,400-$4,100 | DIY approach |
| **Eaton SmartRack Self-Cooling 42U** | $5,000-$8,000 | Fully integrated 5.5 kW cooling |
| Sysracks Air-Conditioned Rack | $2,500-$4,000 | Rear-door cooling module |

**Overkill for a single workstation**, but makes sense if you plan to add a second machine or NAS in the future.

#### Option 5: Immersion Cooling (NOT Recommended)

| Solution | Cost | Verdict |
|----------|------|---------|
| Enermax Cirrus Mk1 (4-GPU) | $50,000+ (cooling only) | Absurdly expensive for this use case |
| Thermaltake IX700 (enthusiast tank) | ~$5,000-$10,000 with fluid | Not yet in production, bleeding-edge |

**Custom loop liquid cooling provides 95%+ of the thermal benefit at 5-10% of the cost.**

#### Recommended Cooling Strategy by Build

| Build | Recommended Cooling | Total Cooling Cost | Self-Contained? |
|-------|--------------------|--------------------|-----------------|
| **Single RTX 5090 (office build)** | GPU AIO (LYNK+) + high-airflow case | **$530-$650** | **Yes** |
| **Dual RTX 5090** | Custom loop (CPU + 2 GPUs) + spot AC | **$1,200-$2,000** | Needs window exhaust |
| **RTX PRO 6000 Blackwell** | Custom loop + spot AC | **$1,300-$2,200** | Needs window exhaust |
| **Budget (RTX 4090)** | Stock air cooling + high-airflow case | **$190 (case only)** | **Yes** |

### Noise Levels

| Configuration | Cooling Method | Noise Level | Office-Friendly? |
|--------------|---------------|-------------|-----------------|
| 1x RTX 5090 + GPU AIO | Liquid (GPU) + Air (case) | **30-40 dBA** | **Yes — quiet** |
| 1x RTX 4090 / 5090 | Air cooling only | 40-50 dBA under load | Yes (headphones recommended) |
| 2x RTX 5090 | Custom loop | 35-45 dBA | Yes (with liquid cooling) |
| 2x RTX 5090 | Air cooling only | 45-55 dBA under load | Borderline (dedicated room ideal) |
| 1x RTX PRO 6000 BW | Blower-style stock | 45-55 dBA under load | Borderline (blower GPUs are louder) |

### Ambient Temperature Requirements

| Range | Status | Notes |
|-------|--------|-------|
| 18-22°C (64-72°F) | **Ideal** | Full GPU performance, no throttling |
| 22-25°C (72-77°F) | Acceptable | Minor clock reduction possible under peak load |
| 25-30°C (77-86°F) | **Problematic** | Expect thermal throttling during sustained training |
| 30°C+ (86°F+) | Dangerous | Significant throttling, potential instability |

**Office HVAC typically maintains 22-24°C. With a GPU AIO cooler, a single RTX 5090 workstation will operate comfortably in this range without supplemental cooling.**

### Internet Requirements

The only cloud component remaining is Bedrock API calls:
- ~5-10 MB per API batch (JSON payloads)
- Standard office internet (50+ Mbps) is more than sufficient
- Initial S3 data download: 5-50 GB one-time transfer

---

## 8. Depreciation & Residual Value

### GPU Depreciation Schedule

| Year | RTX 5090 Est. Resale | RTX PRO 6000 BW Est. Resale | RTX 4090 Est. Resale |
|------|---------------------|-----------------------------|--------------------|
| 0 (purchase) | $3,400-$5,500 | $7,999 | $1,600-$2,000 |
| 1 | $2,800-$4,000 | $6,000-$7,000 | $1,200-$1,500 |
| 2 | $1,800-$2,500 | $4,000-$5,000 | $800-$1,000 |
| 3 | $1,000-$1,500 | $2,500-$3,500 | $500-$700 |
| 4 | $600-$900 | $1,500-$2,000 | $300-$500 |
| 5 | $300-$500 | $800-$1,200 | $150-$300 |

**Historical pattern**: High-end GPUs retain ~50-60% value at year 1, ~35-45% at year 2, ~20-30% at year 3. Professional GPUs (Quadro/RTX PRO line) depreciate slower due to enterprise demand.

### System (Non-GPU) Depreciation

The rest of the workstation (CPU, RAM, storage, PSU, case): ~$1,500-$3,000 depending on build.
- Year 1: 80% residual
- Year 2: 60% residual
- Year 3: 40% residual
- These components have longer useful lives (5-7 years) and can be reused with future GPU upgrades.

---

## 9. 3-Year TCO Comparison

### Assumptions

- **Year 1**: 1 run/week (52 runs) — proof of concept, iteration
- **Year 2**: 2 runs/week (104 runs) — scaling experiments
- **Year 3**: 3 runs/week (156 runs) — production iteration
- **Total runs over 3 years**: 312
- **Bedrock API cost**: $63.43/run (all scenarios)
- **Electricity**: $0.30/kWh (Fremont, CA — PG&E residential rate, Feb 2026). Per-run training electricity is $0.69 (GPU load only); total yearly estimates include full pipeline system-on time and idle overhead (~$60/year fixed + per-run system cost).

### TCO-1: Cloud Only — ml.g5.xlarge Spot (Optimized)

| Year | Runs | GPU Compute | Bedrock API | S3 + Monitoring | Total |
|------|------|-------------|-------------|-----------------|-------|
| 1 | 52 | $260 | $3,298 | $61 | **$3,619** |
| 2 | 104 | $520 | $6,596 | $61 | **$7,177** |
| 3 | 156 | $780 | $9,894 | $61 | **$10,735** |
| **3-Year** | **312** | **$1,560** | **$19,789** | **$183** | **$21,532** |

### TCO-2: Cloud Only — ml.g5.12xlarge Spot (Original Pipeline)

| Year | Runs | GPU Compute | Bedrock API | S3 + Monitoring | Total |
|------|------|-------------|-------------|-----------------|-------|
| 1 | 52 | $1,329 | $3,298 | $61 | **$4,688** |
| 2 | 104 | $2,658 | $6,596 | $61 | **$9,315** |
| 3 | 156 | $3,988 | $9,894 | $61 | **$13,943** |
| **3-Year** | **312** | **$7,975** | **$19,789** | **$183** | **$27,947** |

### TCO-3: Local RTX 5090 Full Build + Cloud Bedrock (RECOMMENDED)

*Full workstation with GPU AIO liquid cooling — see Section 11 for complete BOM.*

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | Workstation CapEx (full build with AIO cooling) | $6,850 |
| 1 | Electricity (52 runs x $0.69 + idle) | $120 |
| 1 | Bedrock API (52 x $63.43) | $3,298 |
| 1 | Internet (allocated) | $0 (existing) |
| 1 | **Year 1 Total** | **$10,268** |
| 2 | Electricity (104 runs) | $240 |
| 2 | Bedrock API | $6,596 |
| 2 | **Year 2 Total** | **$6,836** |
| 3 | Electricity (156 runs) | $360 |
| 3 | Bedrock API | $9,894 |
| 3 | **Year 3 Total** | **$10,254** |
| | **3-Year Gross** | **$27,360** |
| | Less: Hardware residual value (year 3) | -$3,000 |
| | **3-Year Net** | **$24,360** |

### TCO-4: Local RTX 4090 Build + Cloud Bedrock (Budget)

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | Workstation CapEx | $3,300 |
| 1 | Electricity (52 runs x $0.83 + idle) | $127 |
| 1 | Bedrock API | $3,298 |
| 1 | **Year 1 Total** | **$6,725** |
| 2 | Electricity + Bedrock API | $6,849 |
| 3 | Electricity + Bedrock API | $10,274 |
| | **3-Year Gross** | **$23,848** |
| | Less: Hardware residual value | -$1,200 |
| | **3-Year Net** | **$22,648** |

### TCO-5: Local Dual RTX 5090 + Cloud Bedrock

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | Workstation CapEx | $11,000 |
| 1 | Electricity (52 runs x $0.81) | $125 |
| 1 | Bedrock API | $3,298 |
| 1 | **Year 1 Total** | **$14,423** |
| 2 | Electricity + Bedrock API | $6,846 |
| 3 | Electricity + Bedrock API | $10,269 |
| | **3-Year Gross** | **$31,538** |
| | Less: Hardware residual value | -$5,000 |
| | **3-Year Net** | **$26,538** |

### TCO-6: Local RTX PRO 6000 Blackwell + Cloud Bedrock

| Year | Cost Type | Amount |
|------|-----------|--------|
| 1 | Workstation CapEx | $14,000 |
| 1 | Electricity (52 runs x $0.66) | $118 |
| 1 | Bedrock API | $3,298 |
| 1 | **Year 1 Total** | **$17,416** |
| 2 | Electricity + Bedrock API | $6,832 |
| 3 | Electricity + Bedrock API | $10,250 |
| | **3-Year Gross** | **$34,498** |
| | Less: Hardware residual value | -$6,000 |
| | **3-Year Net** | **$28,498** |

### 3-Year TCO Summary

| Option | 3-Year Net Cost | GPU Training Speed | CapEx | Risk Level |
|--------|----------------|-------------------|-------|------------|
| **TCO-1: Cloud g5.xlarge spot** | **$21,532** | Moderate (8-16 hrs) | $0 | Spot interruptions |
| **TCO-2: Cloud g5.12xlarge spot** | **$27,947** | Moderate (8-16 hrs) | $0 | Spot interruptions |
| **► TCO-3: Local 1x RTX 5090 (RECOMMENDED)** | **$24,360** | **Fast (2-4 hrs)** | **$6,850** | Hardware failure |
| TCO-4: Local 1x RTX 4090 | $22,648 | Fast (3-6 hrs) | $3,300 | Stock scarcity |
| TCO-5: Local 2x RTX 5090 | $26,538 | Fastest (1.5-2.5 hrs) | $11,000 | Higher complexity |
| TCO-6: Local RTX PRO 6000 BW | $28,498 | Fast (2-3.5 hrs) | $14,000 | Overkill for 8B |

### Key TCO Insights

```
3-Year Cost Ranking (lowest to highest):
────────────────────────────────────────
1. Cloud g5.xlarge spot:     $21,532  ◄── Cheapest but slowest, spot interruption risk
2. Local RTX 4090:           $22,648  ◄── Good budget option but stock dwindling
3. ► Local RTX 5090 (PICK):  $24,360  ◄── RECOMMENDED — fast, future-proof, self-contained
4. Local 2x RTX 5090:        $26,538  ◄── Only if parallel experiments needed
5. Cloud g5.12xlarge spot:   $27,947  ◄── Original pipeline — most expensive cloud
6. Local RTX PRO 6000:       $28,498  ◄── Only if scaling to 70B+ models
```

**The RTX 5090 build is ~$3,588 cheaper than the original cloud pipeline (g5.12xlarge) over 3 years, while delivering 4-8x faster training, zero spot interruption risk, and local training compute. It's ~$2,828 more than the cheapest cloud option but completes training in hours instead of half a day.**

---

## 10. Recommendation: Build the RTX 5090 Workstation

### Decision: GO

| Factor | Assessment |
|--------|------------|
| **Model** | Granite-8B QLoRA — fits easily in 32GB VRAM with 16-20GB headroom |
| **Build cost** | **$6,850** (mid-range, with GPU AIO liquid cooling) |
| **Training speed** | **2-4 hours** for 5 epochs (vs 8-16 hrs cloud g5.xlarge) |
| **Full pipeline** | **~6-12 hours** end-to-end including Bedrock API calls |
| **Per-run cost** | **$64.62** (electricity $0.69 + Bedrock $63.43 + internet $0.50) |
| **Per-run savings** | **$24.37/run** vs g5.12xlarge spot, **$59.96/run** vs p4d spot |
| **Break-even (2x/week)** | **~2.7 years** vs g5.12xlarge spot, **~13 months** vs p4d spot |
| **Break-even (daily)** | **~9.4 months** vs g5.12xlarge spot, **~3.8 months** vs p4d spot |
| **3-Year TCO** | **$24,360** (vs $27,948 cloud g5.12xlarge — saves $3,588) |
| **Office-friendly** | Yes — 30-40 dBA with AIO cooling, standard 15A circuit |
| **Future-proof** | 32GB VRAM handles up to ~20B QLoRA models |
| **Data privacy** | Training compute is fully local — training data stays on-premises. Bedrock API calls (teacher model) still transit to AWS. |
| **Dual GPU needed?** | **No** — single RTX 5090 is sufficient for 8B. Saves ~1 hr/run but costs $5K+ more. |
| **PRO 6000 needed?** | **No** — 96GB is overkill unless scaling to 34B-70B models. |

### Architecture

```
EXCELFORE OFFICE (RTX 5090 Workstation)    CLOUD (AWS)
┌─────────────────────────────┐            ┌────────────────────────┐
│                             │            │                        │
│  1. Data Preparation        │            │                        │
│     - S3 download (once)    │◄── sync ──│  S3 Bucket (source)    │
│     - Function extraction   │            │                        │
│     - JSONL generation      │            │                        │
│     [CPU: ~1-2 hrs]         │            │                        │
│                             │            │                        │
│  2. QLoRA Training          │            │                        │
│     - Granite-8B 4-bit NF4  │            │                        │
│     - RTX 5090 (32GB VRAM)  │            │                        │
│     - batch=4-8, seq=4096   │            │                        │
│     - 5 epochs: 2-4 hrs     │── API ───►│  3. Bedrock Claude     │
│     - Student inference     │            │     Sonnet 4.5         │
│     - Quality evaluation    │◄─ resp ───│     (Teacher Model)    │
│     [GPU: ~3-5 hrs]         │            │  [API: ~2-4 hrs]       │
│                             │            │                        │
│  3. Model Export            │            │                        │
│     - LoRA adapters → S3    │── upload ─►│  S3 (model artifacts)  │
│     - Local checkpoints     │            │                        │
│                             │            │                        │
│  ─────────────────────────  │            │                        │
│  GPU AIO Liquid Cooled      │            │                        │
│  Power: ~770W under load    │            │                        │
│  Electricity: ~$0.69/run    │            │  Bedrock: ~$63/run     │
│  Noise: 30-40 dBA           │            │                        │
│  Total pipeline: ~6-12 hrs   │            │                        │
└─────────────────────────────┘            └────────────────────────┘

Per-run: ~$65 (vs $89 cloud g5.12xlarge, vs $125 cloud p4d)
Savings: $24-$60 per run
Break-even: ~2.7 years at 2 runs/week vs g5.12xlarge
```

### Cost Reduction Priority Order

Since Bedrock API costs dominate (~71% of every cloud run, ~98% of every local run), optimize in this order:

```
Priority 1: Enable Bedrock Batch Inference (50% off)
             Savings: ~$31/run → $3,224/year at 2x/week
             Cost: $0 (configuration change)
             ROI: Immediate

Priority 2: Enable Prompt Caching (90% off cached input)
             Savings: ~$3-5/run → $312-520/year at 2x/week
             Cost: $0 (code change)
             ROI: Immediate

Priority 3: Build the RTX 5090 workstation (this proposal)
             Savings: ~$24/run → $2,534/year at 2x/week
             Cost: $6,850 CapEx
             ROI: ~2.7 years at 2x/week vs g5.12xlarge
```

### What About Scaling to Larger Models?

| Model Size | RTX 5090 (32GB) | RTX PRO 6000 (96GB) | 2x RTX 5090 (64GB) | Cloud p4d (640GB) |
|-----------|-----------------|---------------------|--------------------|--------------------|
| 8B QLoRA | Easy | Overkill | Overkill | Overkill |
| 13B QLoRA | Comfortable | Overkill | Overkill | Overkill |
| 20B QLoRA | Tight (batch=1-2) | Easy | Comfortable | Overkill |
| 34B QLoRA | **Won't fit** | Comfortable | Tight | Easy |
| 70B QLoRA | **Won't fit** | Tight (batch=1) | **Won't fit** | Comfortable |
| 70B Full FT | **Won't fit** | **Won't fit** | **Won't fit** | Possible |

If there's a clear roadmap to 70B+ models, the RTX PRO 6000 (96GB) becomes compelling. For 8B-20B work, the RTX 5090 is the better value.

---

## 11. Full RTX 5090 Workstation Build (Bill of Materials)

### Complete Build — Ready to Order

This is the **recommended full workstation** to be assembled at the Excelfore office. Based on the gaming build blueprint with targeted upgrades for sustained AI training workloads.

*Prices verified February 15, 2026 via Amazon, Best Buy, Newegg, and manufacturer sites.*

| # | Component | Product | Spec Highlights | Cost |
|---|-----------|---------|----------------|------|
| 1 | **GPU** | **NVIDIA RTX 5090 32GB** | 210 TFLOPS FP16, 1,792 GB/s, Blackwell | **$3,400-$5,500** |
| 2 | **CPU** | AMD Ryzen 7 9800X3D | 8-core/16-thread, AM5, 120W TDP | $450 |
| 3 | **Motherboard** | GIGABYTE X870 Eagle WIFI7 | PCIe 5.0 x16, DDR5, Wi-Fi 7 | $230 |
| 4 | **RAM** | G.SKILL Trident Z5 DDR5 **64GB (2x32GB)** | DDR5-5600 CL36, dual channel | $170 |
| 5 | **Storage** | Crucial P310 2TB NVMe | PCIe Gen4, 7,100 MB/s read | $130 |
| 6 | **Case** | **Corsair 5000D Airflow** | Full tower, dual 360mm rad support, dual-chamber | $175 |
| 7 | **PSU** | **Corsair HX1500i 1500W 80+ Platinum** | 1500W ATX 3.1, 80% rule = 1200W continuous, 440W headroom | $370 |
| 8 | **CPU Cooler** | CORSAIR Nautilus 360 RS AIO | 360mm radiator, front-mounted | $120 |
| 9 | **GPU Cooler** | **LYNK+ Modular AIO 360mm** | Drip-free quick-disconnect, top-mounted | $380 |
| 10 | **Case Fans** | 2x CORSAIR RS120 ARGB | Additional exhaust fans | $50 |
| 11 | **UPS** | CyberPower GX1500U 1500VA / 900W | Sine wave battery backup for training runs | $240 |
| 12 | **Misc** | Thermalright AM5 holder, PSU cables | | $30 |
| | | **WORKSTATION TOTAL** | | **$5,745-$7,845** |
| | | | | |
| 13 | **Monitor** | ASUS ROG Strix 34" XG349C | 34" ultrawide, 180Hz, IPS | $1,100 |
| 14 | **Mouse** | Logitech G502 X Plus | Wireless | $130 |
| | | **WITH PERIPHERALS** | | **$6,975-$9,075** |

**Mid-range estimate (RTX 5090 at ~$4,500 street): ~$6,845 workstation, ~$8,075 with peripherals**

### Why Each Upgrade from the Gaming Blueprint

| Component | Gaming Build | Workstation Build | Why |
|-----------|-------------|-------------------|-----|
| **GPU** | RTX 5080 16GB ($1,000) | **RTX 5090 32GB** ($3,400-$5,500) | 16GB = 0-4GB headroom, OOM risk. 32GB = batch=4-8, future models to 20B |
| **RAM** | 32GB DDR5 ($100) | **64GB DDR5** ($170) | Data preprocessing loads datasets into memory. 32GB is tight. |
| **PSU** | 1050W ($140) | **1500W Platinum ATX 3.1** ($370) | RTX 5090 TDP 575W + spikes to 650W+. 1050W leaves only 80W headroom (dangerous). 1500W = 440W headroom. |
| **Case** | Corsair 3500X ($110) | **Corsair 5000D Airflow** ($175) | Full tower fits CPU AIO (front) + GPU AIO (top). Dual-chamber isolates PSU heat. 3500X is undersized for sustained 575W load. |
| **GPU Cooling** | Stock air | **LYNK+ AIO** ($380) | Cuts GPU temp by ~25°C. GPU stays <60°C during multi-hour training. Noise drops to 30-40 dBA. Self-contained, zero maintenance. |
| CPU, Mobo, SSD, CPU AIO | Same | **Same** | No change needed. 8-core CPU is sufficient — GPU does 95%+ of training work. |

### Power Budget Verification

```
RTX 5090 TDP:             575W
RTX 5090 transient spike:  ~650W (brief, <1ms)
CPU (9800X3D full load):   ~105W
Motherboard + RAM:         ~50W
SSD + Fans + AIO pumps:    ~40W
                           ─────
System total (sustained):  ~770W
System total (peak):       ~845W

PSU: 1500W 80+ Platinum
  At 770W load: 51% utilization → peak efficiency (~94%)
  80% rule capacity: 1,200W → 430W headroom ✓
  Handles transient spikes with ease ✓
```

### Cooling Layout in Case

```
Corsair 5000D Airflow — Dual Chamber Layout
┌─────────────────────────────────────────┐
│  TOP: GPU AIO Radiator (360mm)          │ ← Exhaust (hot GPU coolant out)
│  ┌───────────────────────────────────┐  │
│  │                                   │  │
│  │  GPU (RTX 5090 + LYNK+ block)    │  │
│  │  CPU (9800X3D + Nautilus block)   │  │
│  │  Motherboard (X870 Eagle)         │  │
│  │  RAM (64GB DDR5)                  │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│  FRONT: CPU AIO Radiator (360mm)        │ ← Intake (fresh air in)
├─────────────────────────────────────────┤
│  REAR CHAMBER (hidden):                 │
│  PSU (1500W) + cables + SSD             │ ← Isolated from GPU heat
└─────────────────────────────────────────┘
```

**Self-contained cooling**: CPU AIO pulls fresh air through front radiator → over components → GPU AIO exhausts through top radiator. No external AC unit needed. Office HVAC (22-24°C) is sufficient.

### Training Performance

| Metric | Value |
|--------|-------|
| QLoRA Granite-8B throughput | ~80-150 tokens/sec |
| Batch size | 4-8 (with 16-20GB headroom) |
| Time per epoch (10K samples, seq=4096) | 30-60 min |
| **5 epochs training** | **2-4 hours** |
| Student inference (per epoch) | ~20-40 min |
| Quality eval (syntax, MISRA, protocol) | ~10-20 min |
| Bedrock API calls (teacher + corrections) | ~2-4 hours (network-bound) |
| **Full pipeline end-to-end** | **~6-12 hours** |

### ROI Summary

*Note: This ROI summary assumes a flat 2 runs/week from Year 1 for simplicity. The 3-Year TCO in Section 9 uses a ramping schedule (1→2→3 runs/week) which yields the same 312 total runs but with a different cash flow profile.*

| Metric | vs Cloud g5.12xlarge spot | vs Cloud p4d.24xlarge spot |
|--------|--------------------------|---------------------------|
| **Cloud cost per run** | $88.99 | $124.58 |
| **Local cost per run** | $64.62 | $64.62 |
| **Savings per run** | **$24.37** | **$59.96** |
| **CapEx** | $6,850 | $6,850 |
| | | |
| Break-even at 1 run/week | 5.4 years | 2.2 years |
| **Break-even at 2 runs/week** | **2.7 years** | **13.2 months** |
| Break-even at 3 runs/week | 1.8 years | 8.8 months |
| Break-even at daily runs | 9.4 months | 3.8 months |
| | | |
| **Year 1 savings (2x/week)** | -$4,316 (net cost due to CapEx) | -$614 (nearly break-even) |
| **Year 2 savings (2x/week)** | -$1,781 cumulative | +$5,622 cumulative |
| **Year 3 savings (2x/week)** | +$753 cumulative | +$11,858 cumulative |

### Fallback: Phase 1 with RTX 5080

If the RTX 5090 is unavailable or budget is constrained, build with the 5080 now and swap later:

| Phase | Build | Cost | Training Time | Break-Even |
|-------|-------|------|---------------|------------|
| **Phase 1 (now)** | Same build but RTX 5080 + 1050W PSU + 3500X case | $2,635 | 4-8 hrs | **~10 months** (2x/week vs g5.12xlarge) |
| **Phase 2 (6-12 mo)** | Swap GPU → 5090, PSU → 1500W, Case → 5000D, add GPU AIO | +$2,850 (sell 5080 for ~$800) | 2-4 hrs | Already ROI-positive |
| **Total** | | **~$5,485** | | |

This phased approach starts training immediately for $2,635 and reaches full RTX 5090 performance for ~$5,485 total — potentially cheaper than buying a 5090 at today's inflated street prices.

---

## Sources

- [NVIDIA RTX PRO 6000 Blackwell Official](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) — 96GB GDDR7, 252 TFLOPS FP16, 600W TDP
- [RTX PRO 6000 Blackwell Price ($7,999)](https://videocardz.com/newz/nvidia-flagship-rtx-pro-6000-is-now-rtx-5080-cheaper-as-card-price-drops-to-7999) — VideoCardz, price drop from $8,565
- [RTX PRO 6000 Blackwell Specs (WareDB)](https://waredb.com/processor/nvidia-rtx-pro-6000-blackwell) — 24,064 CUDA, 752 Tensor, 1,792 GB/s bandwidth
- [RTX PRO 6000 NVLink Status](https://pcpartpicker.com/forums/topic/492576-does-nvidia-rtx-pro-6000-blackwell-support-nvlink) — No NVLink on workstation edition
- [RTX PRO 6000 vs RTX 6000 Ada (Technical.city)](https://technical.city/en/video/RTX-6000-Ada-Generation-vs-RTX-PRO-6000-Blackwell) — 96GB vs 48GB, 600W vs 300W
- [RTX PRO 6000 Newegg Listing](https://www.newegg.com/nvidia-blackwell-rtx-pro-6000-96gb-graphic-card/p/N82E16814132106) — Currently available
- [NVIDIA RTX 5090 Official](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) — 32GB GDDR7, $1,999 MSRP, 575W TDP
- [RTX 5090 Street Pricing](https://videocardz.com/newz/nvidia-geforce-rtx-5090-prices-already-pushing-toward-4000) — $3,400-$5,500 due to supply shortage
- [RTX 5090 BF16 TFLOPS (NVIDIA Forums)](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543) — ~210 TFLOPS dense
- [AWS EC2 p4d Pricing (Vantage)](https://instances.vantage.sh/aws/ec2/p4d.24xlarge) — $21.96/hr on-demand
- [AWS EC2 p5 Pricing (Vantage)](https://instances.vantage.sh/aws/ec2/p5.48xlarge) — $55.04/hr on-demand
- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) — ml.p4d.24xlarge: $37.69/hr training
- [AWS GPU Price Increase (The Register)](https://www.theregister.com/2026/01/05/aws_price_increase) — 15% increase on p5e/p5en in Jan 2026
- [Fremont, CA Electricity Rates 2026 (EnergySage)](https://www.energysage.com/local-data/electricity-cost/ca/alameda-county/fremont/) — Fremont, CA average $0.30/kWh (PG&E)
- [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — Claude Sonnet 4.5: $3/1M input, $15/1M output
- [RTX 4090 Price Tracker](https://bestvaluegpu.com/history/new-and-used-rtx-4090-price-history-and-specs/) — $1,600-$2,000
- [LYNK+ RTX 5090 Modular AIO Review (TechPowerUp)](https://www.techpowerup.com/review/lynk-geforce-rtx-5090-modular-aio-liquid/) — $380, cuts temps by ~25°C
- [Alphacool RTX 5090 AIO Coolers](https://www.alphacool.com/en/news/new-rtx-5080-5090-cooler) — Eiswolf AIO, CES 2025
- [EKWB RTX 5090 Water Block ($342)](https://www.ekwb.com/shop/ek-quantum-vector3-fe-rtx-5090-plexi) — Full-cover custom loop block
- [Watercool HEATKILLER for RTX PRO 6000 Blackwell](https://shop.watercool.de/HEATKILLER-INOX-Pro-for-NVIDIA-RTX-6000-Blackwell) — Industrial-grade workstation block
- [Bykski RTX PRO 6000 Block (24/7 continuous use)](https://www.bykski.us/products/bykski-durable-metal-pom-gpu-water-block-and-backplate-for-nvidia-rtx-pro-6000-blackwell-workstation-edition-n-rtxpro6000-ws-sr-continuous-usage)
- [Tripp Lite SRCOOL12K Server Room AC](https://tripplite.eaton.com/smartrack-12000-btu-120v-portable-air-conditioning-unit-small-server-rooms-network-closets~SRCOOL12K) — 12,000 BTU, $1,130-$1,444
- [Fractal Design Torrent Review (Tom's Hardware)](https://www.tomshardware.com/reviews/fractal-design-torrent-review) — Best-in-class airflow
- [RTX 5090 Thermals (GamersNexus)](https://gamersnexus.net/gpus/nvidia-geforce-rtx-5090-founders-edition-review-benchmarks-gaming-thermals-power) — 72°C GPU / 89°C VRAM at stock
- [Custom Loop Cost Discussion (PCPartPicker)](https://pcpartpicker.com/forums/topic/189202-how-much-does-a-decent-custom-water-cooling-loop-cost)
- [Coolant Maintenance Guide (Corsair)](https://www.corsair.com/us/en/explorer/diy-builder/custom-cooling/how-often-should-i-change-the-coolant-in-my-custom-cooling-loop/) — Change every 6-12 months
- FULL_PROCEDURE.md — Pipeline step definitions and original cloud cost estimates
