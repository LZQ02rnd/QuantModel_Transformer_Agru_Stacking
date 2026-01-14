## Project: Transformer + AGRU Stacking for Equity Selection

This repository contains a deep learning model for quantitative equity selection.  
The core architecture is a **Transformer + AGRU Stacking ensemble**, optimized mainly with **IC Loss (Information Coefficient Loss)**.

### Structure

- **`model/transformer_agru_stacking.py`**: Full code for the model, dataset, training loop and score generation (includes `MyModel`, `StockDataset`, etc.).
- **`model/generate_score_df.sh`**: Bash script (Linux) to load a trained model and generate `score_df_2022-2023.parquet`.
- **`backtest_metrics.csv`**: Example backtest performance metrics.
- **`nav_curve.csv`**: Example NAV (net asset value) curve.
- **`score_df_2022-2023.parquet`**: Example scores for backtesting (kept locally; ignored by Git via `.gitignore`).
- **`quanthw_202509/`**: Full dataset directory (ignored by Git; not pushed to GitHub).

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

### GitHub usage

- The `.gitignore` file is configured to ignore:
  - Large datasets such as `quanthw_202509/`
  - Model weights `*.pt` / `*.pth`
  - Output files like `*.parquet`
  - Documents `*.docx`

You can initialize and push this repository with:

```bash
git init
git add .
git commit -m "Add Transformer + AGRU Stacking quant model"
git branch -M main
git remote add origin <your GitHub repo URL>
git push -u origin main
```

This way, GitHub only shows the core code and example backtest results, while large data and model weights stay local.
