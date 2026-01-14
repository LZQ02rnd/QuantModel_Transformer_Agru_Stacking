"""
量化投资理论与实务 - Transformer + AGRU Stacking集成模型
核心架构：
1. Transformer：全局视野，捕捉长期依赖
2. AGRU (Attention-based GRU)：结合注意力机制的GRU，自适应关注重要时间步
3. Stacking：集成两种架构的优势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import os
import math
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==================== 模型参数配置 ====================
mymodel_params = {
    "seq_len": 30,
    "feature_dim": 17,
    "transformer": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.1
    },
    "agru": {
        "hidden_dim": 128,  # AGRU隐藏维度
        "num_layers": 2,  # GRU层数
        "dropout": 0.2
    },
    "meta_model": {
        "hidden_dims": [64, 32],
        "dropout": 0.3
    },
    "output_dim": 1
}


# ==================== Warmup Cosine Annealing Scheduler ====================
class WarmupCosineAnnealingLR:
    """
    带Warmup的Cosine Annealing学习率调度器
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine Annealing阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# ==================== IC Loss (信息系数损失) ====================
class ICLoss(nn.Module):
    """
    信息系数损失函数 (IC Loss)
    混合损失：IC Loss (70%) + MSE Loss (30%)
    """
    def __init__(self, ic_weight=0.7, mse_weight=0.3, reduction='mean'):
        super(ICLoss, self).__init__()
        self.ic_weight = ic_weight
        self.mse_weight = mse_weight
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        # 展平
        pred_flat = pred.squeeze(-1)
        target_flat = target.squeeze(-1)
        
        # 计算皮尔逊相关系数（IC）
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = (pred_centered * target_centered).sum()
        pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
        
        correlation = numerator / (pred_std * target_std + 1e-8)
        ic_loss = 1.0 - correlation
        
        # MSE Loss作为辅助损失
        mse_loss = self.mse_loss(pred, target)
        
        # 混合损失
        total_loss = self.ic_weight * ic_loss + self.mse_weight * mse_loss
        return total_loss


# ==================== 位置编码（时序优化版）====================
class TimeSeriesPositionalEncoding(nn.Module):
    """时序位置编码 - 针对金融时序数据优化"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeSeriesPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # 计算div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        
        # 填充位置编码（标准Transformer位置编码）
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model//2]
        
        # 添加衰减因子，让近期时间步权重更大
        decay = torch.exp(-position / max_len * 2)  # [max_len, 1]
        pe = pe * decay  # [max_len, d_model]
        
        # 注册为buffer，形状为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        batch_size, seq_len, d_model = x.size()
        
        # 确保维度匹配
        if d_model != self.d_model:
            raise ValueError(f"Input d_model {d_model} doesn't match expected {self.d_model}")
        
        # 确保seq_len不超过max_len
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        # 广播相加: [1, seq_len, d_model] + [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ==================== Transformer Base Model ====================
class TransformerBaseModel(nn.Module):
    """
    Transformer基模型 - 捕捉全局依赖关系
    """
    def __init__(self, 
                 seq_len=30, 
                 feature_dim=17,
                 d_model=128,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=512,
                 dropout=0.1,
                 output_dim=1):
        super(TransformerBaseModel, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 时序位置编码
        self.pos_encoder = TimeSeriesPositionalEncoding(d_model, dropout, seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多尺度时序特征融合
        self.temporal_pooling = nn.ModuleDict({
            'mean': nn.AdaptiveAvgPool1d(1),
            'max': nn.AdaptiveMaxPool1d(1),
        })
        
        # 输出层
        pool_dim = d_model * 3  # mean + max + last
        self.output_layer = nn.Sequential(
            nn.Linear(pool_dim, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # 多尺度时序特征融合
        encoded_t = encoded.transpose(1, 2)  # [batch, d_model, seq_len]
        
        mean_pool = self.temporal_pooling['mean'](encoded_t).squeeze(-1)  # [batch, d_model]
        max_pool = self.temporal_pooling['max'](encoded_t).squeeze(-1)  # [batch, d_model]
        last_step = encoded[:, -1, :]  # [batch, d_model]
        
        # 融合特征
        fused_features = torch.cat([mean_pool, max_pool, last_step], dim=1)  # [batch, d_model*3]
        
        # 输出预测
        pred = self.output_layer(fused_features)  # [batch, 1]
        return pred


# ==================== AGRU (Attention-based GRU) Base Model ====================
class AGRUBaseModel(nn.Module):
    """
    AGRU (Attention-based GRU) 基模型
    结合注意力机制的GRU，自适应关注重要的时间步
    
    架构：
    1. GRU层：捕捉时序依赖关系
    2. 注意力机制：自适应地关注重要的时间步
    3. 特征融合：结合GRU输出和注意力权重
    
    优势：
    - 注意力机制帮助模型关注关键时间步
    - GRU捕捉时序模式
    - 两者结合提升预测准确性
    """
    def __init__(self,
                 seq_len=30,
                 feature_dim=17,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.2,
                 output_dim=1):
        super(AGRUBaseModel, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力后的特征融合
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # GRU输出 + 注意力输出
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_dim]
        
        Returns:
            pred: [batch, 1]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        
        # GRU处理
        gru_out, hidden = self.gru(x)  # [batch, seq_len, hidden_dim]
        
        # 注意力机制：自适应关注重要的时间步
        # 使用GRU输出作为query, key, value
        attn_out, attn_weights = self.attention(
            gru_out,  # query
            gru_out,  # key
            gru_out   # value
        )  # [batch, seq_len, hidden_dim]
        
        # 残差连接和归一化
        attn_out = self.attention_norm(attn_out + gru_out)  # [batch, seq_len, hidden_dim]
        
        # 特征融合
        # 使用最后一个时间步的GRU输出和注意力输出
        gru_last = gru_out[:, -1, :]  # [batch, hidden_dim]
        attn_last = attn_out[:, -1, :]  # [batch, hidden_dim]
        
        # 也可以使用平均池化或加权平均
        # gru_mean = gru_out.mean(dim=1)  # [batch, hidden_dim]
        # attn_mean = attn_out.mean(dim=1)  # [batch, hidden_dim]
        
        # 融合GRU和注意力特征
        fused_features = torch.cat([gru_last, attn_last], dim=1)  # [batch, hidden_dim * 2]
        
        # 输出预测
        pred = self.output_layer(fused_features)  # [batch, 1]
        return pred


# ==================== MyModel: Transformer + HRU Stacking ====================
class MyModel(nn.Module):
    """
    Transformer + AGRU Stacking集成模型
    
    架构：
    1. 第一层（Base Models）：
       - Transformer：捕捉全局依赖关系
       - AGRU：结合注意力机制的GRU，自适应关注重要时间步
    2. 第二层（Meta Model）：
       - MLP：学习如何组合两个基模型的预测
    """
    def __init__(self, 
                 seq_len=30,
                 feature_dim=17,
                 transformer_config=None,
                 agru_config=None,
                 meta_model_config=None,
                 output_dim=1):
        super(MyModel, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 第一层：基模型
        transformer_config = transformer_config or {}
        agru_config = agru_config or {}
        
        # Transformer基模型
        self.transformer = TransformerBaseModel(
            seq_len=seq_len,
            feature_dim=feature_dim,
            output_dim=output_dim,
            **transformer_config
        )
        
        # AGRU基模型
        self.agru = AGRUBaseModel(
            seq_len=seq_len,
            feature_dim=feature_dim,
            output_dim=output_dim,
            **agru_config
        )
        
        # 第二层：元模型（学习如何组合基模型的预测）
        meta_config = meta_model_config or {}
        hidden_dims = meta_config.get("hidden_dims", [64, 32])
        meta_dropout = meta_config.get("dropout", 0.3)
        
        # 输入是2个基模型的预测（2个值）
        meta_input_dim = 2
        dims = [meta_input_dim] + hidden_dims + [output_dim]
        
        meta_layers = []
        for i in range(len(dims) - 1):
            meta_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                meta_layers.append(nn.LayerNorm(dims[i+1]))
                meta_layers.append(nn.GELU())
                meta_layers.append(nn.Dropout(meta_dropout))
        
        self.meta_model = nn.Sequential(*meta_layers)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_dim]
        
        Returns:
            pred: [batch, 1]
        """
        # 第一层：基模型预测
        transformer_pred = self.transformer(x)  # [batch, 1]
        agru_pred = self.agru(x)  # [batch, 1]
        
        # 组合基模型的预测
        base_preds = torch.cat([transformer_pred, agru_pred], dim=1)  # [batch, 2]
        
        # 第二层：元模型学习如何组合
        final_pred = self.meta_model(base_preds)  # [batch, 1]
        
        return final_pred


# ==================== Dataset类 ====================
class StockDataset(Dataset):
    """股票数据集"""
    def __init__(self, years, data_dir='data', mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        
        self.xs_list = []
        self.ys_list = []
        self.indices_list = []
        
        print(f"Loading {mode} data...")
        for year in tqdm(years, desc=f"Loading {mode} years"):
            xs_path = os.path.join(data_dir, f'{year}_xs.pt')
            ys_path = os.path.join(data_dir, f'{year}_ys.pt')
            indices_path = os.path.join(data_dir, f'{year}_indices.csv')
            
            if os.path.exists(xs_path) and os.path.exists(ys_path):
                try:
                    xs_size = os.path.getsize(xs_path)
                    ys_size = os.path.getsize(ys_path)
                    if xs_size < 1024 * 1024:
                        print(f"Warning: {year}_xs.pt is too small ({xs_size} bytes), skipping...")
                        continue
                    
                    xs = torch.load(xs_path, map_location='cpu', weights_only=False)
                    ys = torch.load(ys_path, map_location='cpu', weights_only=False)
                    indices = pd.read_csv(indices_path)
                except Exception as e:
                    print(f"Error loading {year} data: {e}")
                    continue
                
                self.xs_list.append(xs)
                self.ys_list.append(ys)
                self.indices_list.append(indices)
        
        if self.xs_list:
            self.xs = torch.cat(self.xs_list, dim=0)
            self.ys = torch.cat(self.ys_list, dim=0)
            self.indices = pd.concat(self.indices_list, ignore_index=True)
            print(f"{mode} dataset loaded: {len(self.xs)} samples")
        else:
            raise ValueError(f"No data found for years: {years}")
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


# ==================== 保存模型函数 ====================
def save_model(model, model_params, filepath):
    """保存模型参数和权重到 .pth 文件"""
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    save_dict = {
        'model_state_dict': model_state_dict,
        'model_params': model_params
    }
    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, rank=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Training")
    else:
        pbar = dataloader
    
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        
        loss.backward()
        # 进一步增强梯度裁剪，防止训练不稳定（IC下降问题）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{current_lr:.2e}'
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ==================== 验证函数 ====================
def validate(model, dataloader, criterion, device, rank=0):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validating")
        else:
            pbar = dataloader
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 计算IC（信息系数）
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0).squeeze(-1).numpy()
        all_targets = torch.cat(all_targets, dim=0).squeeze(-1).numpy()
        ic = np.corrcoef(all_preds, all_targets)[0, 1]
        if rank == 0:
            print(f"  IC (Information Coefficient): {ic:.4f}")
    else:
        ic = 0.0
    
    return avg_loss, ic


# ==================== 预测函数 ====================
def predict(model, dataloader, device, rank=0):
    """批量预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Predicting")
        else:
            pbar = dataloader
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    return predictions


# ==================== 生成score_df ====================
def generate_score_df(model, val_years, data_dir, device, batch_size=512, rank=0, world_size=1):
    """生成用于回测的score_df"""
    val_dataset = StockDataset(val_years, data_dir, mode='val')
    
    if world_size > 1:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    predictions = predict(model, val_loader, device, rank)
    
    if world_size > 1:
        gathered_predictions = [None] * world_size
        dist.all_gather_object(gathered_predictions, predictions.numpy())
        
        if rank == 0:
            all_predictions = np.concatenate(gathered_predictions, axis=0)
            all_indices = val_dataset.indices
    else:
        all_predictions = predictions.numpy()
        all_indices = val_dataset.indices
    
    if rank == 0:
        score_df = pd.DataFrame({
            'date': all_indices['date'].values,
            'code': all_indices['code'].values,
            'score': all_predictions.flatten()
        })
        return score_df
    else:
        return None


# ==================== 主训练函数 ====================
def main():
    # ========== 多GPU初始化 ==========
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        if rank == 0:
            print(f"Distributed training: {world_size} GPUs")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Single GPU training on: {device}")
    
    # ========== 配置参数 ==========
    data_dir = '/root/autodl-tmp/data'
    if not os.path.exists(data_dir):
        data_dir = 'data'
    
    train_years = list(range(2006, 2022))
    val_years = [2022, 2023]
    
    batch_size = 512
    num_epochs = 30  # 增加epochs，给模型更多时间学习
    learning_rate = 1e-4  # 进一步降低学习率，提高训练稳定性（从3e-4降到1e-4，防止IC下降）
    weight_decay = 1e-5
    
    # ========== 创建数据集 ==========
    if rank == 0:
        print("Creating datasets...")
        print("=" * 60)
        print("Transformer + AGRU Stacking集成模型")
        print("1. Transformer: 全局视野，捕捉长期依赖")
        print("2. AGRU: 结合注意力机制的GRU，自适应关注重要时间步")
        print("3. Stacking: 集成两种架构的优势")
        print("=" * 60)
    
    train_dataset = StockDataset(train_years, data_dir, mode='train')
    val_dataset = StockDataset(val_years, data_dir, mode='val')
    
    # ========== 创建DataLoader ==========
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler, 
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    
    # ========== 创建模型 ==========
    if rank == 0:
        print("\nCreating Transformer + HRU Stacking model...")
    
    model = MyModel(
        seq_len=mymodel_params["seq_len"],
        feature_dim=mymodel_params["feature_dim"],
        transformer_config=mymodel_params["transformer"],
        agru_config=mymodel_params["agru"],
        meta_model_config=mymodel_params["meta_model"],
        output_dim=mymodel_params["output_dim"]
    )
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # ========== 损失函数和优化器 ==========
    # 进一步调整IC权重：从0.8降到0.7，更保守更稳定（防止IC下降）
    criterion = ICLoss(ic_weight=0.7, mse_weight=0.3, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # ========== Warmup + Cosine Annealing学习率调度器 ==========
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.15)  # 增加warmup比例，从10%到15%，更稳定的训练初期
    
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    
    if rank == 0:
        print(f"Using IC Loss (IC: 70%, MSE: 30%)")
        print(f"Using Warmup + Cosine Annealing LR scheduler")
        print(f"  Initial LR: {learning_rate}")
        print(f"  Min LR: 1e-6")
        print(f"  Warmup steps: {warmup_steps} ({warmup_steps/total_steps*100:.1f}%)")
        print(f"  Total steps: {total_steps}")
    
    # ========== 训练循环 ==========
    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}\n")
    
    best_val_loss = float('inf')
    best_val_ic = -1.0
    patience = 3  # 早停：如果IC连续3个epoch不提升，停止训练（更早停止，防止IC下降）
    no_improve_count = 0
    ic_drop_count = 0  # 记录IC连续下降的次数
    last_ic = -1.0
    
    for epoch in range(num_epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, rank)
        val_loss, val_ic = validate(model, val_loader, criterion, device, rank)
        
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss).to(device)
            val_loss_tensor = torch.tensor(val_loss).to(device)
            val_ic_tensor = torch.tensor(val_ic).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_ic_tensor, op=dist.ReduceOp.SUM)
            train_loss = train_loss_tensor.item() / world_size
            val_loss = val_loss_tensor.item() / world_size
            val_ic = val_ic_tensor.item() / world_size
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val IC: {val_ic:.4f}, LR: {current_lr:.2e}")
            
            # 检查IC是否下降
            if last_ic > 0 and val_ic < last_ic:
                ic_drop_count += 1
                print(f"  ⚠️  IC下降: {last_ic:.4f} → {val_ic:.4f} (连续下降{ic_drop_count}次)")
            else:
                ic_drop_count = 0  # 重置下降计数器
            
            last_ic = val_ic
            
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_val_loss = val_loss
                no_improve_count = 0  # 重置计数器
                save_model(model, mymodel_params, "学号_姓名_model.pth")
                print(f"Best model saved! Val IC: {val_ic:.4f}, Val Loss: {val_loss:.6f}")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"\nEarly stopping: IC has not improved for {patience} epochs")
                    print(f"Best IC: {best_val_ic:.4f}")
                    break
            
            # 如果IC连续下降2次，提前停止
            if ic_drop_count >= 2:
                print(f"\n⚠️  Early stopping: IC has dropped for {ic_drop_count} consecutive epochs")
                print(f"Best IC: {best_val_ic:.4f} (saved at epoch {epoch+1 - no_improve_count})")
                break
    
    # ========== 生成score_df用于回测 ==========
    if rank == 0:
        print("\nGenerating score_df for backtest...")
        score_df = generate_score_df(model, val_years, data_dir, device, batch_size, rank, world_size)
        
        if score_df is not None:
            score_df.to_parquet('score_df_2022-2023.parquet', index=False)
            print("score_df saved to: score_df_2022-2023.parquet")
            
            print("\n" + "=" * 60)
            print("Training Summary:")
            print(f"Best Val IC: {best_val_ic:.4f}")
            print(f"Best Val Loss: {best_val_loss:.6f}")
            print("=" * 60)
    
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print("\nTraining completed!")


if __name__ == "__main__":
    main()

