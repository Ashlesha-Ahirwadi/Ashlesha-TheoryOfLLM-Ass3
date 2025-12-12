# RLHF Assignment: PPO, GRPO, and DPO Implementation

**Course**: Reinforcement Learning from Human Feedback  
**Due Date**: December 12, 2024 at 5pm  
**Authors**: Assignment 3 Submission

---

## Overview

This repository contains a complete implementation of three Reinforcement Learning from Human Feedback (RLHF) approaches:
- **PPO** (Proximal Policy Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **DPO** (Direct Preference Optimization)

All methods are implemented from scratch, trained on the Anthropic HH-RLHF dataset, and evaluated comprehensively.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Compute Requirements](#compute-requirements)
- [Results](#results)
- [Repository Contents](#repository-contents)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Project Structure

```
Assignment_3/
├── Dockerfile                      # Container configuration
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ANALYSIS.md                     # Comprehensive analysis (Part 4)
├── main.py                         # Main training script
├── src/                           # Source code
│   ├── __init__.py
│   ├── utils.py                   # Utility functions
│   ├── data_preparation.py        # Dataset loading and preprocessing (Part 1.1)
│   ├── reward_model.py            # Reward model implementation (Part 1.2)
│   ├── ppo_trainer.py             # PPO implementation (Part 2.1)
│   ├── grpo_trainer.py            # GRPO implementation (Part 2.2)
│   ├── dpo_trainer.py             # DPO implementation (Part 3)
│   └── evaluation.py              # Evaluation framework (Part 4)
└── outputs/                       # Generated outputs
    ├── models/                    # Trained models
    │   ├── reward_model/
    │   ├── ppo_model/
    │   ├── grpo_model/
    │   └── dpo_model/
    ├── results/                   # Evaluation results
    │   ├── evaluation_results.json
    │   ├── failure_analysis.json
    │   ├── pareto_frontier.png
    │   └── reward_distributions.png
    └── samples/                   # Generated samples (20+ per model)
        ├── base_samples.json
        ├── ppo_samples.json
        ├── grpo_samples.json
        └── dpo_samples.json
```

---

## Requirements

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB VRAM (tested on V100)
- CPU: 4+ cores
- RAM: 16GB
- Storage: 20GB free space

**Recommended:**
- GPU: NVIDIA GPU with 16GB+ VRAM (V100, A100, RTX 3090)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 50GB free space

### Software Requirements

- Docker with NVIDIA Container Runtime
- CUDA 11.8+
- Python 3.10 (provided in Docker container)

**Note**: All Python dependencies are managed through Docker. No local Python installation required.

---

## Setup Instructions

### Option 1: Using Docker (Recommended)

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd Assignment_3
```

#### 2. Build Docker Container
```bash
docker build -t rlhf-assignment .
```

**Build time**: ~5-10 minutes (downloads PyTorch, Transformers, etc.)

#### 3. Run Container with GPU
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace rlhf-assignment bash
```

**Verify GPU access:**
```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Option 2: Local Installation (Not Recommended)

If Docker is not available:

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training
python3 main.py --mode all --output_dir outputs
```

**Note**: Docker is required for grading. Local installation is for development only.

---

## Quick Start

### Run Complete Pipeline

Once inside the Docker container:

```bash
# Run all training steps (data prep → reward model → PPO → GRPO → DPO → evaluation)
python3 main.py --mode all \
    --data_subset_size 1000 \
    --max_length 256 \
    --batch_size 2 \
    --num_epochs 1 \
    --output_dir outputs
```

**Total time**: ~2.5 hours on V100 GPU

---

## Usage

### Training Individual Components

#### Step 1: Data Preparation (Part 1.1)
```bash
python3 main.py --mode prepare_data \
    --output_dir outputs \
    --data_subset_size 1000 \
    --max_length 256
```
**Time**: ~3 minutes  
**Output**: `outputs/dataset_stats.json`, `outputs/dataset_distribution.png`

---

#### Step 2: Train Reward Model (Part 1.2)
```bash
python3 main.py --mode train_reward \
    --output_dir outputs \
    --batch_size 2 \
    --num_epochs 1 \
    --data_subset_size 1000 \
    --max_length 256
```
**Time**: ~30 minutes  
**Output**: `outputs/models/reward_model/`  
**Metrics**: Training/validation accuracy, loss curves, error analysis

---

#### Step 3: Train PPO Model (Part 2.1)
```bash
python3 main.py --mode train_ppo \
    --reward_model_path outputs/models/reward_model \
    --output_dir outputs \
    --batch_size 1 \
    --num_epochs 1 \
    --data_subset_size 1000 \
    --max_length 256 \
    --kl_coef 0.05 \
    --clip_ratio 0.2
```
**Time**: ~45 minutes  
**Output**: `outputs/models/ppo_model/`

---

#### Step 4: Train GRPO Model (Part 2.2)
```bash
python3 main.py --mode train_grpo \
    --reward_model_path outputs/models/reward_model \
    --output_dir outputs \
    --batch_size 1 \
    --num_epochs 1 \
    --data_subset_size 1000 \
    --group_size 2 \
    --max_length 256
```
**Time**: ~35 minutes  
**Output**: `outputs/models/grpo_model/`

---

#### Step 5: Train DPO Model (Part 3)
```bash
python3 main.py --mode train_dpo \
    --output_dir outputs \
    --batch_size 1 \
    --num_epochs 1 \
    --data_subset_size 1000 \
    --max_length 256 \
    --dpo_beta 0.1
```
**Time**: ~30 minutes  
**Output**: `outputs/models/dpo_model/`

---

#### Step 6: Evaluate All Models (Part 4)
```bash
python3 main.py --mode evaluate \
    --output_dir outputs \
    --eval_prompts 100
```
**Time**: ~10 minutes  
**Output**: 
- `outputs/results/evaluation_results.json`
- `outputs/results/pareto_frontier.png`
- `outputs/samples/*.json` (20+ examples per model)

---

### Command Line Arguments

#### General Arguments
- `--mode`: Training mode (`prepare_data`, `train_reward`, `train_ppo`, `train_grpo`, `train_dpo`, `evaluate`, `all`)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory (default: `outputs`)
- `--use_gpu`: Use GPU if available (default: True)

#### Data Arguments
- `--dataset_name`: Dataset name (default: `Anthropic/hh-rlhf`)
- `--max_length`: Maximum sequence length (default: 256)
- `--data_subset_size`: Use subset for quick testing (default: None = full dataset)

#### Training Arguments
- `--batch_size`: Training batch size (default: 4, recommended: 1-2 for 8GB GPU)
- `--num_epochs`: Number of epochs (default: 3, used 1 for faster training)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)

#### PPO/GRPO Specific
- `--kl_coef`: KL divergence coefficient (default: 0.05)
- `--clip_ratio`: PPO clip ratio (default: 0.2)
- `--entropy_coef`: Entropy coefficient (default: 0.01)
- `--group_size`: GRPO group size (default: 4, used 2 for memory)

#### DPO Specific
- `--dpo_beta`: DPO temperature parameter (default: 0.1)

#### Evaluation Arguments
- `--eval_prompts`: Number of evaluation prompts (default: 100)

---

## Compute Requirements

### Training Configuration Used

Our submission was trained with these settings (optimized for 8GB GPU):

```bash
# Data
--data_subset_size 1000        # 1,000 training samples
--max_length 256               # 256 tokens max sequence length

# Batch Sizes
Reward Model: --batch_size 2
PPO/GRPO/DPO: --batch_size 1

# Epochs
--num_epochs 1                 # 1 epoch for all models

# GRPO
--group_size 2                 # Reduced from default 4
```

### Resource Usage

| Component | GPU Memory | Training Time | Total Time |
|-----------|-----------|---------------|------------|
| Data Prep | <1 GB | 3 min | 3 min |
| Reward Model | 6-7 GB | 30 min | 33 min |
| PPO | 6-7 GB | 45 min | 1h 18min |
| GRPO | 6-7 GB | 35 min | 1h 53min |
| DPO | 6-7 GB | 30 min | 2h 23min |
| Evaluation | 3-4 GB | 10 min | 2h 33min |

**Total**: ~2.5 hours on single V100 GPU (16GB VRAM)

### Scaling Recommendations

**For 16GB+ GPU:**
```bash
--batch_size 4                 # Increase batch size
--max_length 512               # Longer sequences
--data_subset_size 5000        # More training data
--num_epochs 3                 # More epochs
```

**For 32GB+ GPU:**
```bash
--batch_size 8
--max_length 512
--data_subset_size 10000
--num_epochs 5
--group_size 4                 # Default GRPO group size
```

---

## Results

### Model Performance Summary

| Model | Val Accuracy | Avg Reward | KL Divergence | Training Time |
|-------|-------------|-----------|---------------|---------------|
| Reward Model | 47.5% | - | - | 30 min |
| Base (GPT-2) | - | 2.5 | 0.00 | - |
| PPO | - | 3.8 | 0.15 | 45 min |
| GRPO | - | 3.6 | 0.13 | 35 min |
| DPO | - | 3.4 | 0.10 | 30 min |

### Key Findings

1. **All methods improved** over baseline (55-65% win rate vs base model)
2. **PPO**: Highest reward, most aggressive optimization
3. **GRPO**: Best balance of quality and efficiency
4. **DPO**: Fastest, most conservative, lowest KL drift

### Generated Samples

Sample outputs from each model are available in:
- `outputs/samples/base_samples.json` (20+ examples)
- `outputs/samples/ppo_samples.json` (20+ examples)
- `outputs/samples/grpo_samples.json` (20+ examples)
- `outputs/samples/dpo_samples.json` (20+ examples)

Each sample includes:
- Original prompt
- Generated response
- Reward score
- KL divergence from reference

---

## Repository Contents

### Source Code Files

**Core Implementation:**
- `src/data_preparation.py` (Part 1.1): Dataset loading, preprocessing, analysis
- `src/reward_model.py` (Part 1.2): Reward model with pairwise ranking loss
- `src/ppo_trainer.py` (Part 2.1): PPO with clipped objective, KL penalty, entropy bonus
- `src/grpo_trainer.py` (Part 2.2): GRPO with group-based advantages
- `src/dpo_trainer.py` (Part 3): DPO without explicit reward modeling
- `src/evaluation.py` (Part 4): Comprehensive evaluation framework
- `src/utils.py`: Utility functions (logging, seeding, etc.)

**Training Script:**
- `main.py`: Main entry point with all training modes

**Configuration:**
- `Dockerfile`: Container definition
- `requirements.txt`: Python dependencies

### Documentation

- `README.md`: This file (setup and usage instructions)
- `ANALYSIS.md`: Comprehensive analysis addressing Part 4 requirements

### Generated Outputs

All outputs are in the `outputs/` directory:

**Models** (`outputs/models/`):
- `reward_model/`: Trained reward model + training curves + error analysis
- `ppo_model/`: PPO-trained policy model
- `grpo_model/`: GRPO-trained policy model
- `dpo_model/`: DPO-trained policy model

**Results** (`outputs/results/`):
- `evaluation_results.json`: Quantitative metrics for all models
- `failure_analysis.json`: Failure mode analysis
- `pareto_frontier.png`: Reward vs KL divergence plot
- `reward_distributions.png`: Reward distribution comparison

**Samples** (`outputs/samples/`):
- `*.json`: 20+ generated examples per model with prompts, responses, and metrics

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory Error
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size and sequence length
```bash
--batch_size 1
--max_length 128
--data_subset_size 500
```

#### Issue 2: Docker Build Fails
```
ERROR: failed to solve: process "/bin/sh -c ..." did not complete successfully
```

**Solution**: Check internet connection and retry
```bash
docker build --no-cache -t rlhf-assignment .
```

#### Issue 3: GPU Not Available
```
CUDA available: False
```

**Solution**: Verify NVIDIA Docker runtime
```bash
# Check NVIDIA Docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### Issue 4: Training Takes Too Long

**Solution**: Use smaller subset for testing
```bash
--data_subset_size 100    # Very fast, for testing only
--num_epochs 1
```

#### Issue 5: Empty Training Set
```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**Solution**: Ensure you're using the latest `src/data_preparation.py` that correctly parses HH-RLHF format

---

### Getting Help

1. **Check logs**: Training logs are saved to `outputs/training_*.log`
2. **Review outputs**: Check `outputs/` directory for generated files
3. **Verify setup**: Run `nvidia-smi` and `python3 -c "import torch; print(torch.cuda.is_available())"`
4. **GPU memory**: Monitor with `watch -n 1 nvidia-smi`

---

## Implementation Highlights

### Part 1.1: Dataset Preparation
- ✅ Loads Anthropic HH-RLHF dataset
- ✅ Analyzes distribution and identifies patterns
- ✅ Implements preprocessing pipeline with tokenization
- ✅ Creates balanced train/validation splits (1000/200)
- ✅ Handles edge cases (empty responses filtered)

### Part 1.2: Reward Model
- ✅ Fine-tunes GPT-2 (~124M parameters)
- ✅ Implements pairwise ranking loss: `L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))`
- ✅ Tracks accuracy, loss curves, gradient norms
- ✅ Reports validation accuracy: 47.5%
- ✅ Error analysis on 105 incorrect predictions (20+ examples)

### Part 2.1: PPO Implementation
- ✅ Clipped surrogate objective
- ✅ KL divergence penalty from reference policy
- ✅ Entropy bonus for exploration
- ✅ Trains policy with reward model
- ✅ Hyperparameter tuning (clip ratio, KL coefficient)

### Part 2.2: GRPO Implementation
- ✅ Group-based advantage estimation
- ✅ Samples multiple responses per prompt (group size 2)
- ✅ Computes advantages relative to group mean
- ✅ Simplified policy gradient (no clipping)
- ✅ Compares stability, efficiency, sample quality vs PPO

### Part 3: DPO Implementation
- ✅ Direct preference optimization without reward model
- ✅ Trains on preference pairs directly
- ✅ Most stable and efficient approach
- ✅ Compares with PPO and GRPO

### Part 4: Analysis and Evaluation
- ✅ **4.1 Quantitative**: Win rates, reward scores, KL divergence, Pareto frontier
- ✅ **4.2 Qualitative**: Failure mode analysis, training curve analysis, alignment types
- ✅ Comprehensive ANALYSIS.md with all findings

---

## Reproducibility

### Exact Reproduction

To reproduce our exact results:

```bash
# 1. Build container
docker build -t rlhf-assignment .

# 2. Run container
docker run --gpus all -it --rm -v $(pwd):/workspace rlhf-assignment bash

# 3. Run training with exact parameters
python3 main.py --mode all \
    --seed 42 \
    --data_subset_size 1000 \
    --max_length 256 \
    --batch_size 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --kl_coef 0.05 \
    --clip_ratio 0.2 \
    --entropy_coef 0.01 \
    --group_size 2 \
    --dpo_beta 0.1 \
    --eval_prompts 100 \
    --output_dir outputs
```

**Expected outputs**: All files in `outputs/` directory matching structure above

---

## Citation

### Dataset
```bibtex
@misc{bai2022training,
  title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
  author={Bai, Yuntao and Jones, Andy and Ndousse, Kamal and Askell, Amanda and Chen, Anna and DasSarma, Nova and Drain, Dawn and Fort, Stanislav and Ganguli, Deep and Henighan, Tom and others},
  journal={arXiv preprint arXiv:2204.05862},
  year={2022}
}
```

### Methods
- **PPO**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- **GRPO**: Shao et al. (2024) "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
- **DPO**: Rafailov et al. (2023) "Direct Preference Optimization"

---

## License

This code is submitted as part of an academic assignment. All rights reserved.

---

## Submission Checklist

- ✅ All source code (`src/*.py`, `main.py`)
- ✅ Dockerfile with container configuration
- ✅ README.md with setup and compute requirements
- ✅ ANALYSIS.md with comprehensive Part 4 analysis
- ✅ Generated samples (20+ examples per model in `outputs/samples/`)
- ✅ Trained models in `outputs/models/`
- ✅ Evaluation results in `outputs/results/`
- ✅ All code properly documented
- ✅ Reproducible with provided instructions

---

## Contact



**Submission Date**: December 12, 2024  
**Course**: Reinforcement Learning from Human Feedback