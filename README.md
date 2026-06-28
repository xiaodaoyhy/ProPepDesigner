# ProPepDesigner

**ProPepDesigner: AI driving  de novo design of long-acting GIPR/GLP-1R/GCGR triple agonists for obesity therapy**

[![Python 3.8](https://img.shields.io/badge/python-3.8.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


ProPepDesigner is a hierarchical computational workflow that integrates deep generative modeling, multi-task quantitative structure–activity relationship (QSAR) models, and reinforcement learning. ProPepDesigner employs a ‘backbone-first, modification-later’ strategy, leveraging a primary multi-task model tailored for core backbone profiles (supporting both natural and non-natural amino acids), complemented by a secondary specialized model to predict receptor-activation potency following fatty acid modification.

| Module | Description |
|--------|-------------|
| `GenePeptide/` | mol2mol generative model: pre-training, finetuning, reinforcement learning (RL), sampling |
| `pepQSAR/Backbone_QSAR/` | Backbone multi-task MLP QSAR (RL scoring) |
| `pepQSAR/Full_QSAR/` | Full-sequence multi-task Transformer QSAR |
| `data/` | Data preprocessing scripts and datasets |
| `configurations/` | JSON configs for training, RL, and sampling |
| `myutils/` | SMILES ↔ sequence conversion, sampling evaluation |

---

## Installation

```bash
git clone https://github.com/xiaodaoyhy/ProPepDesigner.git
cd ProPepDesigner
conda env create -f environment.yml
conda activate propepdesigner
conda install -c conda-forge umap-learn hdbscan
pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-linux-x64
```

**Requirements:** Linux · Python 3.8.13 · CUDA GPU (recommended). See [`environment.yml`](environment.yml) for full dependencies.

---

## Quick Start

### Generative model (`cd GenePeptide/`)

```bash
# Pre-training
nohup python -m torch.distributed.run --nproc_per_node=<N> --master_port=<PORT> \
  input_to_training.py --config ../configurations/train.json > train.log 2>&1 &

# Finetuning
nohup python -m torch.distributed.run --nproc_per_node=<N> --master_port=<PORT> \
  input_to_training.py --config ../configurations/finetune.json > finetune.log 2>&1 &

# Reinforcement learning
nohup python -m torch.distributed.run --nproc_per_node=<N> --master_port=<PORT> \
  input_to_reinforcement_learning.py ../configurations/RL_finetune_model.json > RL_finetune_model.log 

# Sampling
CUDA_VISIBLE_DEVICES=0 nohuppython -m torch.distributed.run --nproc_per_node=1 --master_port=<PORT> \
  input_to_sampling.py ../configurations/sampling_finetune.json > sampling_finetune.log 2>&1 &
```



# Train QSAR models
```bash
conda activate propepdesigner
cd pepQSAR/Backbone_QSAR/ && python train_config.py    # Backbone MLP
cd pepQSAR/Full_QSAR/      && python train_config.py    # Full-sequence Transformer
```

---

## Bundled Models and Data

| Content | Path |
|---------|------|
| Base generative model | `GenePeptide/model/base/checkpoint/model_42.ckpt` |
| Finetuned generative model | `GenePeptide/model/finetune/checkpoint/model_46.ckpt` |
| Backbone QSAR | `pepQSAR/Backbone_QSAR/` |
| Full-sequence QSAR | `pepQSAR/Full_QSAR/` |
| Raw data | `data/process/GIPR_GLP1R_GCGR_summary_seq.csv` |
| RL / sampling results | `GenePeptide/model/RL/results*/` |

---

## Citation

Please cite the following paper if you use this repository in your work:
Manuscript in preparation/under review. The citation will be updated shortly upon publication.

## License

[MIT License](LICENSE)

## Code Availability
The source code is provided for academic use.