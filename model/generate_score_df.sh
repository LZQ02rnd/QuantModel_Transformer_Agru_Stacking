#!/bin/bash
# 生成 score_df - 纯bash脚本

cd /root

python3 << 'PYTHON_SCRIPT'
import torch
import pandas as pd
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入模型定义
sys.path.insert(0, '/root')

# 尝试导入模型（支持多种文件名）
model_imported = False
try:
    from 学员_姓名_code import MyModel, StockDataset, mymodel_params
    model_imported = True
    print("✓ 导入模型: 学员_姓名_code")
except ImportError:
    try:
        from 学号_姓名_code import MyModel, StockDataset, mymodel_params
        model_imported = True
        print("✓ 导入模型: 学号_姓名_code")
    except ImportError:
        print("ERROR: 无法导入模型定义")
        print("请确保训练代码文件在 /root/ 目录下")
        exit(1)

print("=" * 60)
print("生成 score_df")
print("=" * 60)

# 检查模型文件
model_files = [
    '/root/学号_姓名_model.pth',
    '/root/学员_姓名_model.pth',
    '/root/学号_姓名_code_model.pth',
    '/root/学员_姓名_code_model.pth'
]
model_path = None
for f in model_files:
    if os.path.exists(f):
        model_path = f
        break

if not model_path:
    print("ERROR: 找不到模型文件")
    for f in model_files:
        print(f"  尝试: {f}")
    exit(1)

print(f"✓ 模型文件: {model_path}")

# 加载模型
print("\n加载模型...")
checkpoint = torch.load(model_path, map_location='cpu')
model_params = checkpoint.get('model_params', mymodel_params)

# 检查模型类型
has_transformer = 'transformer' in model_params
has_agru = 'agru' in model_params
has_meta = 'meta_model' in model_params

if has_transformer and (has_agru or has_meta):
    print("检测到 Stacking 模型（Transformer + AGRU）")
    model = MyModel(
        seq_len=model_params["seq_len"],
        feature_dim=model_params["feature_dim"],
        transformer_config=model_params.get("transformer", {}),
        agru_config=model_params.get("agru", {}),
        meta_model_config=model_params.get("meta_model", {}),
        output_dim=model_params.get("output_dim", 1)
    )
else:
    print("检测到单独的 Transformer 模型")
    if has_transformer:
        tc = model_params.get("transformer", {})
        simple_params = {
            "seq_len": model_params.get("seq_len", 30),
            "feature_dim": model_params.get("feature_dim", 17),
            "d_model": tc.get("d_model", 128),
            "nhead": tc.get("nhead", 8),
            "num_layers": tc.get("num_layers", 3),
            "dim_feedforward": tc.get("dim_feedforward", 512),
            "dropout": tc.get("dropout", 0.1),
        }
        model = MyModel(**simple_params)
    else:
        model = MyModel(**model_params)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ 模型加载成功")

# 准备数据
print("\n准备验证集数据...")
data_dir = '/root/autodl-tmp/quanthw_202509'
if not os.path.exists(data_dir):
    data_dir = 'quanthw_202509'

val_dataset = StockDataset([2022, 2023], data_dir, mode='val')
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
print(f"✓ 验证集: {len(val_dataset)} 个样本")

# 预测
print("\n开始预测...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"使用设备: {device}")

predictions = []
with torch.no_grad():
    for batch_x, _ in tqdm(val_loader, desc="预测中"):
        batch_x = batch_x.to(device)
        pred = model(batch_x)
        predictions.append(pred.cpu().numpy())

predictions = np.concatenate(predictions, axis=0).flatten()
print(f"✓ 预测完成: {len(predictions)} 个预测值")

# 生成 score_df
print("\n生成 score_df...")
score_df = pd.DataFrame({
    'date': val_dataset.indices['date'].values,
    'code': val_dataset.indices['code'].values,
    'score': predictions
})
score_df['date'] = pd.to_datetime(score_df['date'])

print(f"score_df shape: {score_df.shape}")
print(f"Date range: {score_df['date'].min()} to {score_df['date'].max()}")
print(f"Unique codes: {score_df['code'].nunique()}")

# 保存
output_path = '/root/score_df_2022-2023.parquet'
score_df.to_parquet(output_path, index=False)
file_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"\n✓ 已保存: {output_path} ({file_size:.2f} MB)")
print("=" * 60)
print("完成！现在可以运行回测: python3 run_backtest.py")
print("=" * 60)
PYTHON_SCRIPT
