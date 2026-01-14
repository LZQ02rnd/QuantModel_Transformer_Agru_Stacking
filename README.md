## Project: Transformer + AGRU Stacking for Equity Selection

This repository contains a deep learning model for quantitative equity selection.  
The core architecture is a **Transformer + AGRU Stacking ensemble**, optimized mainly with **IC Loss (Information Coefficient Loss)**.

### Structure

- **`model/transformer_agru_stacking.py`**: Full code for the model, dataset, training loop and score generation (includes `MyModel`, `StockDataset`, etc.).
- **`model/generate_score_df.sh`**: Bash script (Linux) to load a trained model and generate `score_df_2022-2023.parquet`.
- **`backtest_metrics.csv`**: Example backtest performance metrics.
- **`nav_curve.csv`**: Example NAV (net asset value) curve.
- **`score_df_2022-2023.parquet`**: Example scores for backtesting (kept locally; ignored by Git via `.gitignore`).

### Environment

Recommended: Python 3.10+

Core dependencies:

- `torch`
- `pandas`
- `numpy`
- `tqdm`

You can create a `conda` or `venv` environment and install:

```bash
pip install torch pandas numpy tqdm
```

### How to train the model

1. Make sure the data directory `quanthw_202509/` is in the project root (same level as `model/`).
2. From the project root, run:

```bash
python model/transformer_agru_stacking.py
```

The script will:

- Train on data from 2006–2021
- Validate on 2022–2023 and save the best model by **IC**
- After training, generate `score_df_2022-2023.parquet` for backtesting

### Data

This repository does **not** include any raw data for legal and size reasons.

The model expects a directory named `quanthw_202509/` in the project root, containing yearly
time‑series tensors and index files with the following structure (one triplet per year):

- `<YEAR>_xs.pt`  — input features, shape `[N, seq_len, feature_dim]`
- `<YEAR>_ys.pt`  — target returns, shape `[N, 1]`
- `<YEAR>_indices.csv` — metadata with at least:
  - `date`: trading date
  - `code`: instrument identifier

Examples: `2006_xs.pt`, `2006_ys.pt`, `2006_indices.csv`, …, `2023_xs.pt`, `2023_ys.pt`, `2023_indices.csv`.

The data itself is assumed to come from a proprietary or competition dataset and is **not publicly
redistributable**. To run this project, users should prepare their own dataset in the same format
and place it under `quanthw_202509/`.

### Generate `score_df` using a pre-trained model

If you already have a trained model checkpoint (e.g. `transformer_agru_model.pth`), in a Linux environment:

1. Place the model code and checkpoint in the same directory (e.g. `/root`).
2. Run:

```bash
bash model/generate_score_df.sh
```

The script will:

- Load the model definition and weights
- Run inference on the validation set (2022–2023)
- Save `score_df_2022-2023.parquet` in the current environment


