import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import random
import torch.nn as nn
import h5py
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import pickle
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings('ignore')

plt.rcParams.update({"font.family": "serif","font.serif": ["Times New Roman"],})
plt.rcParams['axes.unicode_minus'] = False

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

def batch_shap(explainer, dataset, batch_size=5, save_interval=None):
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
            with open(f"shap_intermediate_春季_{batch_idx // batch_size + 1}.pkl", 'wb') as f:
                pickle.dump(temp_save, f)
            print(f"Intermediate results saved, {len(temp_save)} samples so far")

    shap_values = np.concatenate(all_shap, axis=0)
    print(f"SHAP computation finished, shape: {shap_values.shape}")
    return shap_values

def global_shap_analysis(shap_values, masks, feature_names, save_path=None):
    masked_shap = shap_values * masks[:, np.newaxis, :, :]

    importance = np.mean(np.abs(masked_shap), axis=(0, 2, 3))

    total = importance.sum()
    percentage = importance / total * 100 if total > 0 else np.zeros_like(importance)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "percentage": percentage
    })

    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Global Feature Importance Analysis")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print("mask mean:", np.mean(masks))
    print("mask unique:", np.unique(masks))

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Feature importance saved to: {save_path}")

    return df

def save_shap_results(shap_values, masks, test_indices=None, save_dir="./shap_results_春季"):
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "shap_values_春季.npy"), shap_values)
    np.save(os.path.join(save_dir, "masks_春季.npy"), masks)

    if test_indices is not None:
        np.save(os.path.join(save_dir, "test_indices_春季.npy"), test_indices)

    metadata = {
        'n_samples': shap_values.shape[0],
        'n_features': shap_values.shape[1],
        'spatial_shape': shap_values.shape[2:],
        'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'note': 'spring data'
    }

    with open(os.path.join(save_dir, "metadata_春季.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"SHAP results saved to: {save_dir}")
    print(f"   - shap_values_春季.npy: {shap_values.shape}")
    print(f"   - masks_春季.npy: {masks.shape}")
    print("SHAP min:", shap_values.min())
    print("SHAP max:", shap_values.max())
    print("SHAP mean:", np.mean(np.abs(shap_values)))


def load_shap_results(save_dir="./shap_results_春季"):
    shap_values = np.load(os.path.join(save_dir, "shap_values_春季.npy"))
    masks = np.load(os.path.join(save_dir, "masks_春季.npy"))

    if shap_values.ndim == 5 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)
        print("Automatically removed last SHAP dimension (...,1)")

    with open(os.path.join(save_dir, "metadata_春季.pkl"), 'rb') as f:
        metadata = pickle.load(f)

    print("Loading SHAP results:")
    print(f"   - Number of samples: {metadata['n_samples']}")
    print(f"   - Number of features: {metadata['n_features']}")
    print(f"   - Spatial dimensions: {metadata['spatial_shape']}")
    print(f"   - Save time: {metadata['save_time']}")

    note = metadata.get('note', metadata.get('备注', 'N/A (old metadata version)'))
    print(f"   - Note: {note}")

    return shap_values, masks, metadata
def plot_feature_importance(df, save_path):

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    features = df["feature"].values
    values = df["importance"].values
    values = values * 1e6
    percentages = df["percentage"].values

    cmap = LinearSegmentedColormap.from_list(
        'Nature',
        ['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027']
    )
    norm_vals = percentages / percentages.max()
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| (×10⁻⁶)', fontsize=24)

    for spine in ax.spines.values():
        spine.set_visible(True)

    x_max = values.max() * 1.15
    ax.set_xlim(0, x_max)

    for i, (v, pct) in enumerate(zip(values, percentages)):
        ax.text(
            v + values.max() * 0.01,
            i,
            f"{pct:.1f}%",
            va='center',
            fontsize=23,
            fontweight='bold'
        )

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
def main():
    set_seed(46)

    data_path = r"D:\CNN\春季数据处理\output_2010_2023_spring\data_with_mask_2010_2023_spring.mat"
    save_dir = r"./shap_results_春季_test"
    feature_names = ["U10", "V10", "short_wave", "long_wave",
                     "air_pressure", "km", "u", "v", "air_temperature"]

    X, Y, mask, _, _, _, _ = load_ocean_data(data_path)
    (_, _, _), (_, _, _), (test_X, test_Y, test_mask) = split_data_sequential(X, Y, mask)

    shap_values, masks, metadata = load_shap_results(save_dir)

    df = global_shap_analysis(shap_values, test_mask, feature_names,
                              save_path=os.path.join(save_dir, "feature_importance_春季.csv"))

    plot_feature_importance(
        df,
        save_path=os.path.join(save_dir, "shap_importance_bar_春季.png")
    )

    print("Spring data evaluation complete")

if __name__ == "__main__":
    main()
