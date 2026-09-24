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
print("Device:", device)


def set_seed(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_ocean_data(file_path, normalize=True):
    print(f"Loading data: {file_path}")
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    data = np.transpose(data, (3, 2, 1, 0))

    X = data[:, [1, 2, 3, 4, 6, 7, 8, 9, 13], :, :]
    Y = data[:, 0, :, :]
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
    test_idx = np.arange(val_end, N)
    return (X[train_end:val_end], Y[train_end:val_end], mask[train_end:val_end]), \
        (X[val_end:], Y[val_end:], mask[val_end:])


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


class Wrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output.mean(dim=(1, 2), keepdim=True)


def batch_shap(explainer, dataset, batch_size=5, save_interval=None, save_dir="./shap_results"):
    all_shap = []
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    print(f"Starting SHAP computation: {len(dataset)} samples, {n_batches} batches")

    for batch_idx in tqdm(range(0, len(dataset), batch_size), desc="Computing SHAP"):
        batch_end = min(batch_idx + batch_size, len(dataset))

        batch_data = []
        for i in range(batch_idx, batch_end):
            batch_data.append(dataset[i][0])
        batch = torch.stack(batch_data, dim=0).to(device)

        sv = explainer.shap_values(batch)

        if isinstance(sv, list):
            sv = sv[0]

        if len(sv.shape) == 4 and sv.shape[-1] == 1:
            sv = np.squeeze(sv, axis=-1)

        all_shap.append(sv)

        if save_interval and (batch_idx // batch_size + 1) % save_interval == 0:
            temp_save = np.concatenate(all_shap, axis=0)
            intermediate_path = os.path.join(save_dir, f"shap_intermediate_秋季_{batch_idx // batch_size + 1}.pkl")
            with open(intermediate_path, 'wb') as f:
                pickle.dump(temp_save, f)
            print(f"Intermediate results saved, {len(temp_save)} samples so far: {intermediate_path}")

    shap_values = np.concatenate(all_shap, axis=0)
    print(f"SHAP computation finished, shape: {shap_values.shape}")
    return shap_values


def global_shap_analysis(shap_values, masks, feature_names, save_path=None):
    masked_shap = shap_values * masks[:, np.newaxis, :, :]

    importance_median = np.median(np.abs(masked_shap), axis=(0, 2, 3))
    importance_mean = np.mean(np.abs(masked_shap), axis=(0, 2, 3))
    importance_std = np.std(np.abs(masked_shap), axis=(0, 2, 3))

    percent_median = importance_median / importance_median.sum() * 100

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_median": importance_median,
        "importance_mean": importance_mean,
        "importance_std": importance_std,
        "percentage": percent_median
    })

    df = df.sort_values('importance_median', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Global Feature Importance Analysis - Autumn")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Feature importance saved to: {save_path}")

    return df


def save_shap_results(shap_values, masks, test_indices=None, save_dir="./shap_results"):
    os.makedirs(save_dir, exist_ok=True)

    shap_path = os.path.join(save_dir, "shap_values_秋季.npy")
    mask_path = os.path.join(save_dir, "masks_秋季.npy")

    np.save(shap_path, shap_values)
    np.save(mask_path, masks)

    if test_indices is not None:
        idx_path = os.path.join(save_dir, "test_indices_秋季.npy")
        np.save(idx_path, test_indices)

    metadata = {
        'n_samples': shap_values.shape[0],
        'n_features': shap_values.shape[1],
        'spatial_shape': shap_values.shape[2:],
        'data_season': 'autumn',
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = os.path.join(save_dir, "metadata_秋季.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"SHAP results saved to: {save_dir}")
    print(f"   - shap_values_秋季.npy: {shap_values.shape}")
    print(f"   - masks_秋季.npy: {masks.shape}")
    print(f"   - metadata_秋季.pkl: metadata saved")


def load_shap_results(save_dir="./shap_results"):
    shap_path = os.path.join(save_dir, "shap_values_秋季.npy")
    mask_path = os.path.join(save_dir, "masks_秋季.npy")
    metadata_path = os.path.join(save_dir, "metadata_秋季.pkl")

    shap_values = np.load(shap_path)
    masks = np.load(mask_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"Loading SHAP results (autumn):")
    print(f"   - Number of samples: {metadata['n_samples']}")
    print(f"   - Number of features: {metadata['n_features']}")
    print(f"   - Spatial dimensions: {metadata['spatial_shape']}")
    print(f"   - Data season: {metadata['data_season']}")
    print(f"   - Save time: {metadata['save_time']}")

    return shap_values, masks, metadata


def main():
    set_seed(46)

    data_path = r"D:\CNN\秋季数据处理\output_2010_2023_autumn\data_with_mask_2010_2023_autumn.mat"
    model_path = r"CNN-Transformer-多尺度35绝对位置编码-动态学习率2010-2023秋季-有气压_best_model.pth"
    save_dir = r"./shap_results_秋季_test"
    feature_names = ["U10", "V10", "short_wave", "long_wave",
                     "air_pressure", "km", "u", "v", "air_temp"]

    force_recompute = True

    print("\n" + "=" * 60)
    print("Step 1: Load autumn data")
    print("=" * 60)
    X, Y, mask, means_X, stds_X, mean_Y, std_Y = load_ocean_data(data_path)

    (val_X, val_Y, val_mask), \
        (test_X, test_Y, test_mask) = split_data_sequential(X, Y, mask)

    print(f"Validation set: {val_X.shape[0]} samples")
    print(f"Test set: {test_X.shape[0]} samples")

    test_dataset = OceanSSTDataset(test_X, test_Y, test_mask)

    print("\n" + "=" * 60)
    print("Step 2: Load model")
    print("=" * 60)
    model = MemoryEfficientOceanSSTModel(in_channels=9).to(device).eval()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    def remove_temp_channel(w):
        return torch.cat([w[:, :5, :, :], w[:, 6:, :, :]], dim=1)

    state_dict['input_conv.branch_3x3.0.weight'] = remove_temp_channel(
        state_dict['input_conv.branch_3x3.0.weight'])
    state_dict['input_conv.branch_5x5.0.weight'] = remove_temp_channel(
        state_dict['input_conv.branch_5x5.0.weight'])
    state_dict['residual_conv.weight'] = remove_temp_channel(
        state_dict['residual_conv.weight'])

    model.load_state_dict(state_dict)
    print("Model loaded successfully, adapted to 9 input channels")

    shap_path = os.path.join(save_dir, "shap_values_秋季.npy")
    if not force_recompute and os.path.exists(shap_path):
        print("\n" + "=" * 60)
        print("Step 3: Load existing SHAP results (autumn)")
        print("=" * 60)
        shap_values, masks, metadata = load_shap_results(save_dir)

        importance_path = os.path.join(save_dir, "feature_importance_秋季.csv")
        df = global_shap_analysis(shap_values, test_mask, feature_names,
                                  save_path=importance_path)

        plt.figure(figsize=(10, 6))
        plt.barh(df["feature"], df["importance_median"])
        plt.gca().invert_yaxis()
        plt.xlabel("SHAP重要性 (绝对值中位数)")
        plt.title(f"全局SHAP特征重要性 - 秋季数据 (全测试集, {len(test_dataset)}个样本)")
        plt.tight_layout()

        importance_plot_path = os.path.join(save_dir, "shap_importance_秋季.png")
        plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Feature importance plot saved to: {importance_plot_path}")

        print("Analysis complete")
        return

    print("\n" + "=" * 60)
    print("Step 3: Prepare SHAP background samples (autumn)")
    print("=" * 60)

    n_background = min(100, len(val_X))
    bg_idx = np.random.choice(len(val_X), n_background, replace=False)
    background = torch.stack([torch.tensor(val_X[i], dtype=torch.float32)
                              for i in bg_idx]).to(device)
    print(f"Number of background samples: {n_background}")

    print("\n" + "=" * 60)
    print("Step 4: Create SHAP explainer")
    print("=" * 60)
    wrapped_model = Wrapper(model)
    explainer = shap.GradientExplainer(wrapped_model, background)
    print("SHAP explainer created")

    print("\n" + "=" * 60)
    print("Step 5: Compute SHAP values (full test set, autumn)")
    print("=" * 60)

    start_time = time.time()
    shap_values = batch_shap(explainer, test_dataset, batch_size=10, save_interval=None, save_dir=save_dir)
    elapsed_time = time.time() - start_time
    print(f"SHAP computation time: {elapsed_time / 60:.2f} minutes")

    print("\n" + "=" * 60)
    print("Step 6: Save SHAP results (autumn)")
    print("=" * 60)
    save_shap_results(shap_values, test_mask, save_dir=save_dir)

    print("\n" + "=" * 60)
    print("Step 7: Global feature importance analysis (autumn)")
    print("=" * 60)
    importance_path = os.path.join(save_dir, "feature_importance_秋季.csv")
    df = global_shap_analysis(shap_values, test_mask, feature_names,
                              save_path=importance_path)


    print("\n" + "=" * 60)
    print("Full test set SHAP analysis for autumn completed!")
    print(f"All results saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
