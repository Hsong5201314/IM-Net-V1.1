# 🚀 IM-Net: Proactive Interference Mitigation in Multi-Task Recommendation

> Official PyTorch Implementation of the Paper
> *Model-agnostic proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization*

## 📖 Introduction

> This repository contains the official implementation of IM-Net, a novel proactive meta-learning framework designed to mitigate gradient interference in multi-task recommendation systems. By leveraging high-order derivatives (Hessian-vector products) and a multi-step lookahead mechanism, IM-Net dynamically resolves gradient conflicts before they harm the backbone model (e.g., LightGCN, NCF).

## 🛠️ Environment Setup

> Our code is implemented and tested on Python 3.9+ and PyTorch 2.0+.

### Step 1: Create a virtual environment (Recommended)

```bash
conda create -n imnet python=3.10
conda activate imnet
```

### Step 2: Install PyTorch with CUDA support

> Important: Please install the correct PyTorch version corresponding to your hardware's CUDA version. Below is an installation example for CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install other dependencies

```bash
pip install -r requirements.txt
```

## 📁 Dataset Preparation

> We evaluate our model on two public datasets: Amazon Books and Yelp.
Please place the processed datasets into the data/ directory, following the structure below:
```text
IM-Net/
│── data/
│   ├── amazon_books_processedDataV3/
│   └── yelp_processed_for_meta/
```
**Note for Reviewers: The pre-processed datasets are already included in our submitted supplementary materials.**

## 🚀 One-Click Reproducibility (For Reviewers)

> We provide integrated shell scripts to reproduce all core experimental results in the paper.
> First, grant execution permission to script files:
```bash
chmod +x scripts/*.sh
```

### 🏆 RQ1: Main Performance Comparison

> Our main evaluation (RQ1) is comprehensively designed across two distinct dimensions to prove the superiority of IM-Net. 

As presented in the paper, the evaluation is divided into:
* **Category A (Architecture & Augmentation):** Evaluating different structural backbones and data-centric methods (LightGCN, HINE, NCF, SimGCL).
* **Category B (Optimization Strategies):** Evaluating gradient-centric Multi-Objective Optimization (MOO) strategies (Scalarization, GradNorm, PCGrad, MGDA, and our **IM-Net**). To ensure a strictly fair comparison, all strategies in this category are uniformly applied to the LightGCN backbone.

> The provided bash scripts are pre-configured to reproduce the core results of our proposed method (**LightGCN + IM-Net**), which achieves the state-of-the-art performance in Category B.

**To reproduce the IM-Net results:**
```bash
# Reproduce LightGCN + IM-Net on Yelp2018 (Target: Recall ~0.0844, NDCG ~0.0627)
bash scripts/run_rq1_yelp_meta.sh

# Reproduce LightGCN + IM-Net on Amazon-Book (Target: Recall ~0.0428, NDCG ~0.0324)
bash scripts/run_rq1_amazon_meta.sh
```

> *(Note: To reproduce other baseline methods from the table, such as SimGCL or PCGrad, you can directly modify the --model_name and --mode arguments inside these bash scripts.)*

### 🔬 RQ2: Ablation Study

> Verify the effectiveness of each core module:
```bash
bash scripts/run_rq2_ablation.sh
```

### ⏱️ RQ3: Training Efficiency Analysis

> Evaluate training speed and computational overhead:
⚠️ **Note: Please run this script on CUDA-enabled GPU devices. Running on CPU will lead to inaccurate time statistics.**

```bash
bash scripts/run_rq3_efficiency.sh
```


### 🌪️ RQ4: Controlled Stress Test (Digital Twin Simulation)

> As stated in our paper, the extreme gradient interference stress test is conducted using our standalone evaluation framework: **Digital_Twin_Stress-Test**. 
> 
> To reproduce RQ4, please navigate to the sibling directory `../Digital_Twin_Stress-Test/` provided in the supplementary materials and refer to its specific documentation.


## 📂 Repository Structure

```text
IM-Net/
│
├── core/                       # Core Algorithms & Training Engine
│   ├── imnet.py                # IM-Net meta network implementation
│   ├── train_engine.py         # Meta training logic & Hessian calculation
│   └── optimizers_utils.py     # Meta-learning optimizer tools
│
├── models/                     # Plug-and-Play Backbone Networks
│   └── backbone.py             # LightGCN, NCF and other base models
│
├── data_utils/                 # Data Processing Module
│   └── data_loaderMeta.py      # Dataset loading & sparse graph construction
│
├── scripts/                    # One-click running scripts
│   ├── run_rq1_amazon_meta.sh  
│   ├── run_rq1_yelp_meta.sh
│   ├── run_rq2_ablation.sh     
│   └── run_rq3_efficiency.sh   
│
├── data/                       # Processed datasets folder
│
├── main_gpu.py                 # Main entry for overall performance experiments
├── run_table2_ablationBestTest.py  # Entry for ablation experiments
├── run_rq3_efficiency.py       # Entry for efficiency analysis
├── run_rq4_stress_test.py      # Entry for stress test experiments
├── utils.py                    # Common tools & random seed fixing
├── requirements.txt            # Environment dependency list
└── README.md                   # Project instructions
```

## 📝 Citation

> If our work is helpful to your research, please cite our paper:
```bibtex
@inproceedings{imnet2024,
  title={Model-agnostic proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization},
  author={Anonymous Authors},
  booktitle={Proceedings of the XXth International Conference},
  year={2024}
}
```

## ✉️ Contact

> If you have any questions about the code or paper, feel free to submit an issue or contact the authors.