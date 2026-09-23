import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONHASHSEED"] = "0"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16


def set_seed(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class OceanDataset(Dataset):
    def __init__(self, X, Y, mask):
        self.X = X
        self.Y = Y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.Y[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.float32),
        )


def load_data(file_path=r'D:\CNN\秋季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            raise KeyError("'data' dataset not found in the HDF5 file")
        data = f['data'][:]
    print(f"Raw data shape: {data.shape}")

    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :].astype(np.float32)
    Y = data[:, 0, :, :].astype(np.float32)
    lon = data[0, 10, :, 0].astype(np.float32)
    lat = data[0, 11, 0, :].astype(np.float32)
    time = data[:, 12, 0, 0].astype(np.float32)

    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    print(f"Y mean: {Y_mean:.4f}, Y std: {Y_std:.4f}")

    mask = ~np.isnan(Y)
    valid_samples = np.any(mask, axis=(1, 2))
    X = X[valid_samples]
    Y = Y[valid_samples]
    mask = mask[valid_samples]

    Y = np.where(np.isnan(Y), 0.0, Y)
    mask = mask.astype(np.float32)

    C = X.shape[1]
    means = np.zeros(C, dtype=np.float32)
    stds = np.zeros(C, dtype=np.float32)

    for i in range(C):
        xi = X[:, i, :, :]
        valid = ~np.isnan(xi)
        if np.any(valid):
            means[i] = np.nanmean(xi)
            stds[i] = np.nanstd(xi)
        else:
            means[i] = 0.0
            stds[i] = 1.0
        xi[~valid] = means[i]
        X[:, i, :, :] = xi

    X = (X - means.reshape(1, C, 1, 1)) / (stds.reshape(1, C, 1, 1) + 1e-8)

    Y = (Y - Y_mean) / (Y_std + 1e-8)

    return X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std


def split_data(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    np.random.shuffle(train_idx)

    X_train, Y_train, mask_train = X[train_idx], Y[train_idx], mask[train_idx]
    X_val, Y_val, mask_val = X[val_idx], Y[val_idx], mask[val_idx]
    X_test, Y_test, mask_test = X[test_idx], Y[test_idx], mask[test_idx]

    return (X_train, Y_train, mask_train), (X_val, Y_val, mask_val), (X_test, Y_test, mask_test), test_idx


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1,
                                  bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.cbam(out)
        res = self.residual(x)
        return self.relu(out + res)


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = ResDoubleConv(in_channels, 32)
        self.enc2 = ResDoubleConv(32, 64)
        self.enc3 = ResDoubleConv(64, 128)
        self.enc4 = ResDoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResDoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ResDoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResDoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResDoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResDoubleConv(64, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._pad_and_concat(self.up4(b), e4))
        d3 = self.dec3(self._pad_and_concat(self.up3(d4), e3))
        d2 = self.dec2(self._pad_and_concat(self.up2(d3), e2))
        d1 = self.dec1(self._pad_and_concat(self.up1(d2), e1))
        return self.final(d1)

    def _pad_and_concat(self, upsampled, bypass):
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = nn.functional.pad(
            upsampled,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([upsampled, bypass], dim=1)


def train_model(model, train_dataset, val_dataset, device, num_epochs=100, batch_size=4, lr=5e-4,
                save_dir="./results_CBAMunet_year_2010_2023_秋季", patience=10):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_resunet_model_cbamunet_秋季.pth")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            preds = model(batch_x).squeeze(1)
            diff = preds - batch_y
            loss = ((diff ** 2) * batch_mask).sum() / (batch_mask.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y, batch_mask in val_loader:
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                val_preds = model(batch_x).squeeze(1)
                diff = val_preds - batch_y
                val_loss += (((diff ** 2) * batch_mask).sum() / (batch_mask.sum() + 1e-8)).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Best Val Loss={best_val_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"\nValidation loss did not improve for {patience} consecutive epochs. Early stopping triggered.")
            break

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss (2010-2023_Autumn)")
    plt.savefig(os.path.join(save_dir, "loss_curve_cbamunet_year_2010_2023_秋季.png"), dpi=150)
    plt.close()

    print(f"Best model saved: {best_model_path}")
    print(f"Total epochs trained: {epoch}, best validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, best_model_path


def evaluate_model(model, test_ds, means, stds, Y_mean, Y_std, device,
                   save_path="evaluation_results_cbamUNET_year_2010_2023_秋季.txt"):
    model.eval()
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    fvcom_channel_in_X = 5

    sum_sq_err_fvcom = 0.0
    sum_sq_err_model = 0.0
    sum_abs_err_fvcom = 0.0
    sum_abs_err_model = 0.0
    n_valid = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            pred = model(x)

            pred_np = pred.squeeze(1).cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()
            x_np = x.cpu().numpy()
            mask_np = mask.squeeze(1).cpu().numpy()

            pred_denorm = pred_np * Y_std + Y_mean
            y_denorm = y_np * Y_std + Y_mean
            fvcom_denorm = x_np[:, fvcom_channel_in_X, :, :] * stds[fvcom_channel_in_X] + means[fvcom_channel_in_X]

            valid_mask = mask_np == 1
            if not np.any(valid_mask):
                continue

            fvcom_vals = fvcom_denorm[valid_mask]
            model_vals = pred_denorm[valid_mask]
            obs_vals = y_denorm[valid_mask]

            sum_sq_err_fvcom += np.sum((fvcom_vals - obs_vals) ** 2)
            sum_sq_err_model += np.sum((model_vals - obs_vals) ** 2)
            sum_abs_err_fvcom += np.sum(np.abs(fvcom_vals - obs_vals))
            sum_abs_err_model += np.sum(np.abs(model_vals - obs_vals))
            n_valid += fvcom_vals.size

    if n_valid == 0:
        print("No valid test samples available for evaluation statistics!")
        return None

    fvcom_rmse = np.sqrt(sum_sq_err_fvcom / n_valid)
    model_rmse = np.sqrt(sum_sq_err_model / n_valid)
    fvcom_mae = sum_abs_err_fvcom / n_valid
    model_mae = sum_abs_err_model / n_valid

    rmse_improvement = (fvcom_rmse - model_rmse) / fvcom_rmse * 100
    mae_improvement = (fvcom_mae - model_mae) / fvcom_mae * 100

    print("\nOverall test set evaluation results (all samples)")
    print(f"FVCOM RMSE : {fvcom_rmse:.4f} deg C")
    print(f"Model RMSE : {model_rmse:.4f} deg C")
    print(f"RMSE improvement : {rmse_improvement:.2f} %")
    print(f"FVCOM MAE  : {fvcom_mae:.4f} deg C")
    print(f"Model MAE  : {model_mae:.4f} deg C")
    print(f"MAE improvement  : {mae_improvement:.2f} %")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("========== Model Evaluation Results (2010-2023_Autumn) ==========\n")
        f.write(f"\nRMSE evaluation results:\n")
        f.write(f"  FVCOM RMSE: {fvcom_rmse:.4f} deg C\n")
        f.write(f"  Predicted RMSE: {model_rmse:.4f} deg C\n")
        f.write(f"  RMSE improvement: {rmse_improvement:.2f}%\n")
        f.write(f"\nMAE evaluation results:\n")
        f.write(f"  FVCOM MAE: {fvcom_mae:.4f} deg C\n")
        f.write(f"  Predicted MAE: {model_mae:.4f} deg C\n")
        f.write(f"  MAE improvement: {mae_improvement:.2f}%\n")
        f.write("\n" + "=" * 50)

    print(f"\nEvaluation results saved to {save_path}")

    return {
        'fvcom_rmse': fvcom_rmse,
        'model_rmse': model_rmse,
        'rmse_improvement': rmse_improvement,
        'fvcom_mae': fvcom_mae,
        'model_mae': model_mae,
        'mae_improvement': mae_improvement
    }


def plot_temperature_comparison_simple(model, test_ds, test_idx, means, stds, Y_mean, Y_std, device,
                                       num_samples=60):
    model.eval()
    loader = DataLoader(test_ds, batch_size=1)

    all_predictions = []
    all_fvcom = []
    all_sst = []

    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x).squeeze().cpu().numpy()

            pred = pred * Y_std + Y_mean
            y = y.squeeze().cpu().numpy() * Y_std + Y_mean
            fvcom = x[0, 5].cpu().numpy() * stds[5] + means[5]
            m = m.squeeze().cpu().numpy()

            valid_mask = (m == 1)
            if np.any(valid_mask):
                all_sst.append(np.nanmean(y[valid_mask]))
                all_fvcom.append(np.nanmean(fvcom[valid_mask]))
                all_predictions.append(np.nanmean(pred[valid_mask]))
            else:
                all_sst.append(np.nan)
                all_fvcom.append(np.nan)
                all_predictions.append(np.nan)

    all_sst = np.array(all_sst)
    all_fvcom = np.array(all_fvcom)
    all_predictions = np.array(all_predictions)

    n_samples = len(all_sst)
    if n_samples < num_samples:
        print(f"Warning: test set only contains {n_samples} samples, fewer than the requested {num_samples}")
        num_samples = n_samples

    start_idx = n_samples - num_samples

    sst_temps = all_sst[start_idx:]
    fvcom_temps = all_fvcom[start_idx:]
    adjusted_temps = all_predictions[start_idx:]

    valid = ~(np.isnan(sst_temps) | np.isnan(fvcom_temps) | np.isnan(adjusted_temps))
    sst_temps = sst_temps[valid]
    fvcom_temps = fvcom_temps[valid]
    adjusted_temps = adjusted_temps[valid]

    time_steps = np.arange(1, len(sst_temps) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, sst_temps, color='g', linewidth=2.5, label='Satellite SST (Obs)')
    plt.plot(time_steps, fvcom_temps, color='b', linewidth=2.5, label='FVCOM (Raw)')
    plt.plot(time_steps, adjusted_temps, color='r', linewidth=2.5, label='CBAMUnet Adjustment')

    plt.xlabel('Test Time Steps (Last 60 Samples)', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.title(f'Temperature Comparison of the Last {len(time_steps)} Test Samples (2010-2023_Autumn)', fontsize=15)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('temperature_comparison_last_60_samples_year_CBAMUnet_2010_2023_秋季.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print(
        f"Temperature comparison plot for the last {len(time_steps)} test samples saved: "
        f"temperature_comparison_last_60_samples_year_CBAMUnet_2010_2023_秋季.png")


def plot_mean_temperature_comparison(model, test_ds, lon, lat, means, stds, Y_mean, Y_std, device,
                                     output_path='mean_temperature_comparison_CBAMUnet_year_2010_2023_秋季.png'):
    model.eval()
    loader = DataLoader(test_ds, batch_size=1)

    fvcom_all = []
    pred_all = []
    sst_all = []

    print("Computing the mean temperature field over all test samples...")

    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            x, y, m = x.to(device), y.to(device), m.to(device)

            pred = model(x).squeeze().cpu().numpy() * Y_std + Y_mean
            y = y.squeeze().cpu().numpy() * Y_std + Y_mean
            fvcom = x[0, 5].cpu().numpy() * stds[5] + means[5]
            m = m.squeeze().cpu().numpy()

            fvcom_masked = np.where(m == 1, fvcom, np.nan)
            pred_masked = np.where(m == 1, pred, np.nan)
            sst_masked = np.where(m == 1, y, np.nan)

            fvcom_all.append(fvcom_masked)
            pred_all.append(pred_masked)
            sst_all.append(sst_masked)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} samples")

    fvcom_mean = np.nanmean(np.stack(fvcom_all), axis=0)
    pred_mean = np.nanmean(np.stack(pred_all), axis=0)
    sst_mean = np.nanmean(np.stack(sst_all), axis=0)

    print(f"Satellite SST mean temperature range: {np.nanmin(sst_mean):.3f} deg C to {np.nanmax(sst_mean):.3f} deg C")
    print(f"Before correction (FVCOM) temperature range: {np.nanmin(fvcom_mean):.3f} deg C to {np.nanmax(fvcom_mean):.3f} deg C")
    print(f"After correction (CBAMUnet) temperature range: {np.nanmin(pred_mean):.3f} deg C to {np.nanmax(pred_mean):.3f} deg C")

    diff_mean = pred_mean - fvcom_mean

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    temp_min = max(0, min(np.nanmin(fvcom_mean), np.nanmin(pred_mean), np.nanmin(sst_mean)) - 2)
    temp_max = min(30, max(np.nanmax(fvcom_mean), np.nanmax(pred_mean), np.nanmax(sst_mean)) + 2)

    diff_abs_max = max(abs(np.nanmin(diff_mean)), abs(np.nanmax(diff_mean)))
    diff_abs_max = min(diff_abs_max, 10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    titles = [
        'Mean Temperature Before Correction (°C)\n(FVCOM Raw Simulation)',
        'Mean Temperature After Correction (°C)\n(CBAMUnet Corrected Result)',
        'Temperature Difference (°C)\n(After - Before)'
    ]

    data_list = [fvcom_mean, pred_mean, diff_mean]
    cmaps = ['coolwarm', 'coolwarm', 'RdBu_r']
    vmins = [temp_min, temp_min, -diff_abs_max]
    vmaxs = [temp_max, temp_max, diff_abs_max]

    for i, (ax, title, data, cmap, vmin, vmax) in enumerate(zip(axes, titles, data_list, cmaps, vmins, vmaxs)):
        im = ax.pcolormesh(lon_grid, lat_grid, data.T,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', transform=ccrs.PlateCarree())

        ax.set_extent([116, 124, 35, 41], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                            fraction=0.046, pad=0.02, shrink=0.8)
        cbar.set_label('°C', rotation=0, fontsize=12)

        ax.text(0.02, 0.98, f'({chr(97 + i)})', transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('SST Correction Effect Comparison 2010-2023 - Regional All-Day Mean_Autumn',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Mean temperature comparison plot saved: {output_path}")

    print("\nStatistics:")
    print(f"Satellite observed mean temperature: {np.nanmean(sst_mean):.3f} +/- {np.nanstd(sst_mean):.3f} deg C")
    print(f"Mean temperature before correction: {np.nanmean(fvcom_mean):.3f} +/- {np.nanstd(fvcom_mean):.3f} deg C")
    print(f"Mean temperature after correction: {np.nanmean(pred_mean):.3f} +/- {np.nanstd(pred_mean):.3f} deg C")
    print(f"Mean temperature change: {np.nanmean(diff_mean):.3f} deg C")



def main():
    set_seed(46)
    DATA_PATH = r'D:\CNN\秋季数据处理\output_2010_2023_autumn\data_with_mask_2010_2023_autumn.mat'
    MODEL_PATH = r'D:\CNN\CBAM_Unet模型\results_UNet_CBAM_year_2010_2023_秋季\best_resunet_model_cbamunet_秋季.pth'
    SAVE_DIR = "./results_UNet_CBAM_year_2010_2023_秋季"

    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data(DATA_PATH)

    train, val, test, test_idx = split_data(X, Y, mask)

    test_ds = OceanDataset(*test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResUNet(in_channels=10, out_channels=1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    print("Model loaded successfully")

    evaluation_path_csv = os.path.join(SAVE_DIR, "evaluation_results_CBAMunet_year_2010_2023_autumn.csv")

    results = evaluate_model(
        model,
        test_ds,
        means,
        stds,
        Y_mean,
        Y_std,
        device,
        save_path=os.path.join(SAVE_DIR, "evaluation_results_CBAMunet_year_2010_2023_autumn.txt")
    )

    import pandas as pd

    if results is not None:
        df = pd.DataFrame([results])
        df.to_csv(evaluation_path_csv, index=False)
        print(f"Compact CSV saved: {evaluation_path_csv}")

    plot_temperature_comparison_simple(model, test_ds, test_idx, means, stds, Y_mean, Y_std, device, num_samples=60)

    plot_mean_temperature_comparison(model, test_ds, lon, lat, means, stds, Y_mean, Y_std, device,
                                     output_path='mean_temperature_comparison_cbamunet_year_2010_2023_秋季.png')


if __name__ == "__main__":
    main()
