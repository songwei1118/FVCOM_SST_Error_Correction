#!/usr/bin/env python
# coding: utf-8

"""
完整 SHAP 分析 - 全测试集
✔ 不重新训练模型
✔ 直接加载最优模型并去除temp通道
✔ 保存SHAP结果供后续调用
"""
import torch
import os
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import shap
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import pickle
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📱 使用设备:", device)


# ------------------------
# 固定随机种子
# ------------------------
def set_seed(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子设置为: {seed}")


# ------------------------
# 数据加载
# ------------------------
def load_ocean_data(file_path, normalize=True):
    print(f"📂 正在加载数据: {file_path}")
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    data = np.transpose(data, (3, 2, 1, 0))  # T,C,H,W

    # 1. 去掉 temp 通道（原第0个通道），保留其他9个通道
    X = data[:, [1, 2, 3, 4, 6, 7, 8, 9, 13], :, :]  # 9个通道
    Y = data[:, 0, :, :]  # SST目标
    mask = ~np.isnan(Y)
    Y = np.where(mask, Y, 0)
    mask = mask.astype(np.float32)

    if normalize:
        means_X = np.zeros(X.shape[1])
        stds_X = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            xi = X[:, i]
            valid = ~np.isnan(xi)
            means_X[i] = np.nanmean(xi)
            stds_X[i] = np.nanstd(xi)
            xi[~valid] = means_X[i]
            X[:, i] = xi
        X = (X - means_X.reshape(1, -1, 1, 1)) / (stds_X.reshape(1, -1, 1, 1) + 1e-8)
        mean_Y = np.nanmean(Y[mask == 1])
        std_Y = np.nanstd(Y[mask == 1])
        Y = (Y - mean_Y) / (std_Y + 1e-8)
    else:
        means_X, stds_X, mean_Y, std_Y = None, None, None, None

    return X, Y, mask, means_X, stds_X, mean_Y, std_Y


def split_data_sequential(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)
    return (X[train_idx], Y[train_idx], mask[train_idx]), \
        (X[val_idx], Y[val_idx], mask[val_idx]), \
        (X[test_idx], Y[test_idx], mask[test_idx])


class OceanSSTDataset(Dataset):
    def __init__(self, X, Y, mask):
        self.X = X
        self.Y = Y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
            torch.tensor(self.Y[idx], dtype=torch.float32), \
            torch.tensor(self.mask[idx], dtype=torch.float32)


# ======================================================
class EarthSpecificPositionEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, height, width) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class MultiScaleCNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(128, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x3 = self.branch_3x3(x)
        x5 = self.branch_5x5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.fusion(x)
        return x


class MemoryEfficientOceanSSTModel(nn.Module):
    def __init__(self, in_channels=9, spatial_height=200, spatial_width=120, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, downscale_factor=2):
        super().__init__()
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        self.height = spatial_height // downscale_factor
        self.width = spatial_width // downscale_factor
        self.d_model = d_model
        self.input_conv = MultiScaleCNNEncoder(in_channels, 128)
        self.scale_enhance = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(128, d_model, kernel_size=downscale_factor, stride=downscale_factor),
            nn.ReLU(), nn.BatchNorm2d(d_model)
        )
        self.pos_encoding = EarthSpecificPositionEncoding(d_model, self.height, self.width)
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                       dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(d_model, 64, kernel_size=downscale_factor, stride=downscale_factor),
            nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        x_in = x
        x = self.input_conv(x)
        x = self.scale_enhance(x)
        x = self.downsample(x)
        x = self.pos_encoding(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.transformer:
            x = layer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.upsample(x)
        out = self.output_conv(x)
        residual = self.residual_conv(x_in)
        out = out + self.residual_alpha * residual
        return out.squeeze(1)


# ------------------------
# Wrapper + batch SHAP
# ------------------------
class Wrapper(nn.Module):
    """包装模型以输出合适的形状用于SHAP"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 模型输出形状为 [batch, H, W]
        output = self.model(x)
        # 对空间维度求平均，得到每个样本的标量预测值
        return output.mean(dim=(1, 2), keepdim=True)


def batch_shap(explainer, dataset, batch_size=5, save_interval=None):
    """
    批量计算SHAP值并可选保存中间结果

    Parameters:
    -----------
    explainer: SHAP explainer对象
    dataset: 数据集
    batch_size: 批次大小
    save_interval: 每隔多少批次保存一次中间结果

    Returns:
    --------
    shap_values: numpy数组 [n_samples, n_features, H, W]
    """
    all_shap = []
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    print(f"📊 开始计算SHAP值，共{len(dataset)}个样本，{n_batches}个批次")

    for batch_idx in tqdm(range(0, len(dataset), batch_size), desc="计算 SHAP"):
        batch_end = min(batch_idx + batch_size, len(dataset))

        # 准备批次数据
        batch_data = []
        for i in range(batch_idx, batch_end):
            batch_data.append(dataset[i][0])
        batch = torch.stack(batch_data, dim=0).to(device)

        # 计算SHAP值
        sv = explainer.shap_values(batch)

        # 处理返回格式
        if isinstance(sv, list):
            sv = sv[0]

        # 移除最后一维（如果是标量输出）
        if len(sv.shape) == 4 and sv.shape[-1] == 1:
            sv = np.squeeze(sv, axis=-1)

        all_shap.append(sv)

        # 可选：定期保存中间结果
        if save_interval and (batch_idx // batch_size + 1) % save_interval == 0:
            temp_save = np.concatenate(all_shap, axis=0)
            with open(f"shap_intermediate_{batch_idx // batch_size + 1}.pkl", 'wb') as f:
                pickle.dump(temp_save, f)
            print(f"💾 已保存中间结果，当前共{len(temp_save)}个样本")

    shap_values = np.concatenate(all_shap, axis=0)
    print(f"✅ SHAP计算完成，形状: {shap_values.shape}")
    return shap_values


# ------------------------
# 全局 SHAP 分析
# ------------------------
def global_shap_analysis(shap_values, masks, feature_names, save_path=None):
    """
    计算全局SHAP重要性

    Parameters:
    -----------
    shap_values: [n_samples, n_features, H, W]
    masks: [n_samples, H, W] 有效区域掩码
    feature_names: 特征名称列表
    save_path: 保存路径

    Returns:
    --------
    df: DataFrame包含特征重要性统计
    """
    # 只考虑有效区域的SHAP值
    masked_shap = shap_values * masks[:, np.newaxis, :, :]

    # 计算每个特征的重要性（绝对值中位数）
    importance_median = np.median(np.abs(masked_shap), axis=(0, 2, 3))
    importance_mean = np.mean(np.abs(masked_shap), axis=(0, 2, 3))
    importance_std = np.std(np.abs(masked_shap), axis=(0, 2, 3))

    # 计算百分比
    percent_median = importance_median / importance_median.sum() * 100

    # 创建DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_median": importance_median,
        "importance_mean": importance_mean,
        "importance_std": importance_std,
        "percentage": percent_median
    })

    # 按重要性排序
    df = df.sort_values('importance_median', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("全局特征重要性分析")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # 保存结果
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"💾 特征重要性已保存至: {save_path}")

    return df


# ------------------------
# 保存SHAP结果
# ------------------------
def save_shap_results(shap_values, masks, test_indices=None, save_dir="./shap_results"):
    """
    保存完整的SHAP结果

    Parameters:
    -----------
    shap_values: SHAP值数组
    masks: 掩码数组
    test_indices: 测试集索引
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存SHAP值（可能较大，使用numpy格式）
    np.save(os.path.join(save_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(save_dir, "masks.npy"), masks)

    if test_indices is not None:
        np.save(os.path.join(save_dir, "test_indices.npy"), test_indices)

    # 保存元数据
    metadata = {
        'n_samples': shap_values.shape[0],
        'n_features': shap_values.shape[1],
        'spatial_shape': shap_values.shape[2:],
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"💾 SHAP结果已保存至: {save_dir}")
    print(f"   - shap_values.npy: {shap_values.shape}")
    print(f"   - masks.npy: {masks.shape}")


# ------------------------
# 加载SHAP结果
# ------------------------
def load_shap_results(save_dir="./shap_results"):
    """加载之前保存的SHAP结果"""
    shap_values = np.load(os.path.join(save_dir, "shap_values.npy"))
    masks = np.load(os.path.join(save_dir, "masks.npy"))

    with open(os.path.join(save_dir, "metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    print(f"📂 加载SHAP结果:")
    print(f"   - 样本数: {metadata['n_samples']}")
    print(f"   - 特征数: {metadata['n_features']}")
    print(f"   - 空间维度: {metadata['spatial_shape']}")
    print(f"   - 保存时间: {metadata['save_time']}")

    return shap_values, masks, metadata


# ------------------------
# 主流程
# ------------------------
def main():
    set_seed(46)

    # ===== 配置参数 =====
    data_path = r"D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat"
    model_path = r"CNN-Transformer-多尺度35绝对位置编码-动态学习率2010-2023-有气压_best_model.pth"
    save_dir = r"./shap_results_full_test"
    feature_names = ["U10", "V10", "short_wave", "long_wave",
                     "air_pressure", "km", "u", "v", "air_temp"]

    # 可选：是否重新计算SHAP（如果已经保存过，可以跳过计算）
    force_recompute = True  # 设置为False可跳过计算直接加载

    # ===== 1. 加载数据 =====
    print("\n" + "=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    X, Y, mask, means_X, stds_X, mean_Y, std_Y = load_ocean_data(data_path)

    # 分割数据
    (train_X, train_Y, train_mask), \
        (val_X, val_Y, val_mask), \
        (test_X, test_Y, test_mask) = split_data_sequential(X, Y, mask)

    print(f"训练集: {train_X.shape[0]} 样本")
    print(f"验证集: {val_X.shape[0]} 样本")
    print(f"测试集: {test_X.shape[0]} 样本")

    # 创建测试集Dataset
    test_dataset = OceanSSTDataset(test_X, test_Y, test_mask)

    # ===== 2. 加载模型 =====
    print("\n" + "=" * 60)
    print("步骤2: 加载模型")
    print("=" * 60)
    model = MemoryEfficientOceanSSTModel(in_channels=9).to(device).eval()

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 剔除temp通道（原模型有10个通道，现在只有9个）
    def remove_temp_channel(w):
        """移除temp通道（原索引5，0-based）"""
        # w形状: [out_channels, in_channels, H, W]
        return torch.cat([w[:, :5, :, :], w[:, 6:, :, :]], dim=1)

    # 修改相关层的权重
    state_dict['input_conv.branch_3x3.0.weight'] = remove_temp_channel(
        state_dict['input_conv.branch_3x3.0.weight'])
    state_dict['input_conv.branch_5x5.0.weight'] = remove_temp_channel(
        state_dict['input_conv.branch_5x5.0.weight'])
    state_dict['residual_conv.weight'] = remove_temp_channel(
        state_dict['residual_conv.weight'])

    # 加载修改后的权重
    model.load_state_dict(state_dict)
    print("✅ 模型加载成功，已适配9个输入通道")

    # ===== 3. 检查是否已有SHAP结果 =====
    if not force_recompute and os.path.exists(os.path.join(save_dir, "shap_values.npy")):
        print("\n" + "=" * 60)
        print("步骤3: 加载已有SHAP结果")
        print("=" * 60)
        shap_values, masks, metadata = load_shap_results(save_dir)

        # 重新计算特征重要性
        df = global_shap_analysis(shap_values, test_mask, feature_names,
                                  save_path=os.path.join(save_dir, "feature_importance.csv"))

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.barh(df["feature"], df["importance_median"])
        plt.gca().invert_yaxis()
        plt.xlabel("SHAP重要性 (绝对值中位数)")
        plt.title(f"全局SHAP特征重要性 (全测试集, {len(test_dataset)}个样本)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_importance.png"), dpi=150, bbox_inches='tight')
        plt.show()

        print("✅ 分析完成")
        return

    # ===== 4. 准备SHAP背景样本 =====
    print("\n" + "=" * 60)
    print("步骤3: 准备SHAP背景样本")
    print("=" * 60)

    # 从训练集中随机选取200个样本作为背景（使用更多背景样本提高稳定性）
    n_background = min(100, len(train_X))
    bg_idx = np.random.choice(len(train_X), n_background, replace=False)
    background = torch.stack([torch.tensor(train_X[i], dtype=torch.float32)
                              for i in bg_idx]).to(device)
    print(f"背景样本数: {n_background}")

    # ===== 5. 创建SHAP解释器 =====
    print("\n" + "=" * 60)
    print("步骤4: 创建SHAP解释器")
    print("=" * 60)
    wrapped_model = Wrapper(model)
    explainer = shap.GradientExplainer(wrapped_model, background)
    print("✅ SHAP解释器创建完成")

    # ===== 6. 批量计算SHAP值 =====
    print("\n" + "=" * 60)
    print("步骤5: 计算SHAP值（全测试集）")
    print("=" * 60)

    start_time = time.time()
    shap_values = batch_shap(explainer, test_dataset, batch_size=10, save_interval=None)
    elapsed_time = time.time() - start_time
    print(f"⏱️ SHAP计算耗时: {elapsed_time / 60:.2f} 分钟")

    # ===== 7. 保存SHAP结果 =====
    print("\n" + "=" * 60)
    print("步骤6: 保存SHAP结果")
    print("=" * 60)
    save_shap_results(shap_values, test_mask, save_dir=save_dir)

    # ===== 8. 全局特征重要性分析 =====
    print("\n" + "=" * 60)
    print("步骤7: 全局特征重要性分析")
    print("=" * 60)
    df = global_shap_analysis(shap_values, test_mask, feature_names,
                              save_path=os.path.join(save_dir, "feature_importance.csv"))


    print("\n" + "=" * 60)
    print("✅ 全测试集SHAP分析完成！")
    print(f"📁 所有结果已保存至: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()