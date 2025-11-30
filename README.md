# PRC Data Challenge 2025 - Team Sincere Glacier

**Team Name:** sincere-glacier  
**Competition:** Performance Review Commission (PRC) Data Challenge 2025  
**Topic:** Aircraft Fuel Burn Prediction using Hybrid Stacking (GBM + LSTM)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Project Overview

This repository contains the solution developed by Team **sincere-glacier** for the 2025 PRC Data Challenge. Our approach combines physics-based feature engineering with deep learning to predict aircraft fuel consumption for flight intervals.

### Methodology

Our solution uses a **2-stage stacked generalization** approach:

1. **Physics-Aware LightGBM**: Gradient Boosting Machine trained on aggregated flight statistics and aerodynamic features (energy rate, parasitic power, ISA atmosphere properties)
2. **Sequence-Aware LSTM**: PyTorch LSTM network processing raw trajectory time-series to capture temporal dynamics
3. **Meta-Learner**: Non-negative Ridge Regression combining Out-of-Fold (OOF) predictions for final fuel estimation


## Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/johntad110/sincere-glacier-prc2025.git
cd sincere-glacier-prc2025
```

### 2. Install deps

**Option A: Quick Install**

```bash
pip install -r requirements.txt
```

**Option B: Development Install (Recommended)**

```bash
pip install -e .
```

This installs the `fuel_prediction` package in editable mode, enabling the CLI.

### 3. Data Setup

Ensure the competition datasets are in the project directory:

- `prc-2025-datasets/fuel_train.parquet`
- `prc-2025-datasets/flights_train/`
- `prc-2025-datasets/flightlist_train.parquet`
- `prc-2025-datasets/apt.parquet`
- etc.

---

## Usage

### Quick Start: CLI Commands

After installation, you can use the `fuel-predict` command:

```bash
# Run full pipeline (train all models + inference)
fuel-predict pipeline

# Train individual models
fuel-predict train gbm
fuel-predict train lstm
fuel-predict train stacking

# Run inference
fuel-predict predict stacking --dataset rank
fuel-predict predict stacking --dataset final

# Generate OOF predictions
fuel-predict oof lstm
```

### Alternative: Direct Script Execution

```bash
# Training
python pipelines/01_train_gbm.py
python pipelines/02_train_lstm.py
python pipelines/05_lstm_oof.py
python pipelines/06_stacking_train.py

# Inference
python pipelines/03_gbm_inference.py
python pipelines/04_lstm_inference.py
python pipelines/07_stacking_inference.py
```

### Using Makefile (Windows/Unix)

```bash
# Install package
make install

# Run full pipeline
make pipeline

# Train all models
make train

# Run inference
make predict

# Clean generated files
make clean
```

---

## Reproduction Steps

### Phase 1: Model Training

**Step 1: Train LightGBM**

```bash
python pipelines/01_train_gbm.py
```

- Extracts physics features from trajectories
- Trains with 5-fold GroupKFold CV
- **Outputs**: `models/gbm/lgb_model_fold_*.txt`, `models/gbm/ac_means.parquet`, `models/gbm/oof_predictions.parquet`

**Step 2: Train LSTM**

```bash
python pipelines/02_train_lstm.py
```

- Preprocesses trajectories to fixed-length sequences (SEQ_LEN=32)
- Trains PyTorch LSTM with aircraft embeddings
- **Outputs**: `models/lstm/model_fold_*.pth`, `models/lstm/stats.npz`

### Phase 2: Out-of-Fold Predictions

**Step 3: Generate LSTM OOF**

```bash
python pipelines/05_lstm_oof.py
```

- Generates LSTM predictions on training data
- **Output**: `models/lstm/lstm_oof.parquet`

### Phase 3: Stacking Ensemble

**Step 4: Train Meta-Model**

```bash
python pipelines/06_stacking_train.py
```

- Combines GBM and LSTM OOF predictions
- Trains Ridge regression with positive constraints
- **Output**: `models/stacking/meta_model.pkl`

### Phase 4: Inference

**Step 5: Generate Submissions**

```bash
python pipelines/07_stacking_inference.py
```

- Runs inference on ranking and final datasets
- **Outputs**: `submissions/submission_stacking_rank.parquet`, `submissions/submission_stacking_final.parquet`

---

## 🔧 Configuration

Model parameters can be customized via `src/fuel_prediction/config/default.yaml`:

```yaml
gbm:
  params:
    learning_rate: 0.05
    num_leaves: 63
  num_boost_round: 2000

lstm:
  seq_len: 32
  hidden_dim: 256
  num_layers: 3
  batch_size: 256
```

---

## Proj Quirks

- Modular Design: Reusable components (features, models, data, utils)
- Configuration Management: YAML-based configs for easy experimentation
- CLI Interface: User-friendly commands for all operations (nice to have)
- Comprehensive Logging: Structured logging throughout
- Documentation : Docstrings and inline comments


---

## Model Performance

| Model        | CV RMSE (kg) | Description            |
| ------------ | ------------ | ---------------------- |
| GBM          | ~219         | Physics-aware features |
| LSTM         | ~222         | Sequence learning      |
| **Stacking** | **~214**     | **Final ensemble**     |

---

## License

This project is licensed under the **GNU GPLv3** license.

---

## Acknowledgments

- PRC 2025 Data Challenge organizers
- OpenAP library for aviation performance modeling inspiration
- Competition participants for insights

---

Team Sincere Glacier | Built with ❤️ for aviation analytics
