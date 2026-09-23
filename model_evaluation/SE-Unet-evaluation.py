import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd
from io import BytesIO
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import torch


warnings.filterwarnings('ignore')

plt.rcParams.update({"font.family": "serif","font.serif": ["Times New Roman"]})
plt.rcParams['axes.unicode_minus'] = False

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


def load_data(file_path=r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            raise KeyError("'data' dataset not found in HDF5 file")
        data = f['data'][:]
    print(f"Raw data shape: {data.shape}")

    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :].astype(np.float32)
    Y = data[:, 0, :, :].astype(np.float32)
    lon = data[0, 10, :, 0].astype(np.float32)
    lat = data[0, 11, 0, :].astype(np.float32)
    time = data[:, 12, 0, 0].astype(np.float32)

    full_time = pd.date_range(
        start='2010-01-01',
        end='2023-12-31',
        freq='D'
    )

    time = full_time[~((full_time.month == 1) & (full_time.day == 1))]
    assert len(time) == data.shape[0], \
        f"Time length ({len(time)}) does not match data length ({data.shape[0]})!"

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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.avg_pool(x)
        se_weight = self.fc(se_weight)
        return x * se_weight


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
        self.se = SEBlock(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1,
                                  bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
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
                save_dir="./results_SEunet_year_2010_2023", patience=10):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_resunet_model_seunet.pth")
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
            print(f"\nValidation loss did not improve for {patience} consecutive epochs. Early stopping.")
            break

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss (2010-2023 Year)")
    plt.savefig(os.path.join(save_dir, "loss_curve_seunet_year_2010_2023.png"), dpi=150)
    plt.close()

    print(f"Best model saved: {best_model_path}")
    print(f"Total epochs: {epoch}, Best validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, best_model_path


def evaluate_model(model, test_ds, means, stds, Y_mean, Y_std, device,
                   save_path="evaluation_results_saUNET_year_2010_2023.txt"):
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

            pred_np = pred.cpu().numpy()[0, 0]
            y_np = y.cpu().numpy()[0]
            x_np = x.cpu().numpy()[0]
            mask_np = mask.cpu().numpy()[0]

            pred_denorm = pred_np * Y_std + Y_mean
            y_denorm = y_np * Y_std + Y_mean
            fvcom_denorm = x_np[fvcom_channel_in_X] * stds[fvcom_channel_in_X] + means[fvcom_channel_in_X]

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
        print("No valid test samples for evaluation statistics!")
        return None

    fvcom_rmse = np.sqrt(sum_sq_err_fvcom / n_valid)
    model_rmse = np.sqrt(sum_sq_err_model / n_valid)
    fvcom_mae = sum_abs_err_fvcom / n_valid
    model_mae = sum_abs_err_model / n_valid

    rmse_improvement = (fvcom_rmse - model_rmse) / fvcom_rmse * 100
    mae_improvement = (fvcom_mae - model_mae) / fvcom_mae * 100

    print("\nOverall test-set evaluation (all samples)")
    print(f"FVCOM RMSE : {fvcom_rmse:.4f} °C")
    print(f"Model RMSE : {model_rmse:.4f} °C")
    print(f"RMSE improvement : {rmse_improvement:.2f} %")
    print(f"FVCOM MAE  : {fvcom_mae:.4f} °C")
    print(f"Model MAE  : {model_mae:.4f} °C")
    print(f"MAE improvement  : {mae_improvement:.2f} %")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("========== Model Evaluation Results (2010-2023) ==========\n")
        f.write(f"\nRMSE results:\n")
        f.write(f"  FVCOM RMSE: {fvcom_rmse:.4f} °C\n")
        f.write(f"  Predicted RMSE: {model_rmse:.4f} °C\n")
        f.write(f"  RMSE improvement: {rmse_improvement:.2f}%\n")
        f.write(f"\nMAE results:\n")
        f.write(f"  FVCOM MAE: {fvcom_mae:.4f} °C\n")
        f.write(f"  Predicted MAE: {model_mae:.4f} °C\n")
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


def compute_spatial_rmse_maps(model, test_dataset,
                              mean_Y, std_Y,
                              means, stds,
                              device,
                              fvcom_channel=5):
    model.eval()

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sum_sq_model = None
    sum_sq_fvcom = None
    sum_err_model = None
    sum_err_fvcom = None
    count = None

    with torch.no_grad():
        for x, y, mask in loader:

            x = x.to(device)
            pred = model(x)

            pred = pred.cpu().numpy()[0, 0]
            y = y.cpu().numpy()[0]
            mask = mask.cpu().numpy()[0]
            x_np = x.cpu().numpy()[0]

            pred = pred * std_Y + mean_Y
            obs = y * std_Y + mean_Y
            fvcom = x_np[fvcom_channel] * stds[fvcom_channel] + means[fvcom_channel]

            valid = mask == 1

            if sum_sq_model is None:
                sum_sq_model = np.zeros_like(pred)
                sum_sq_fvcom = np.zeros_like(pred)
                sum_err_model = np.zeros_like(pred)
                sum_err_fvcom = np.zeros_like(pred)
                count = np.zeros_like(pred)

            diff_model = pred - obs
            diff_fvcom = fvcom - obs

            sum_sq_model[valid] += (diff_model[valid] ** 2)
            sum_sq_fvcom[valid] += (diff_fvcom[valid] ** 2)

            sum_err_model[valid] += diff_model[valid]
            sum_err_fvcom[valid] += diff_fvcom[valid]
            count[valid] += 1

    count_safe = count + 1e-8

    rmse_model = np.sqrt(sum_sq_model / count_safe)
    rmse_fvcom = np.sqrt(sum_sq_fvcom / count_safe)

    bias_model = sum_err_model / count_safe
    bias_fvcom = sum_err_fvcom / count_safe

    improvement = rmse_fvcom - rmse_model
    improvement_pct = improvement / (rmse_fvcom + 1e-8) * 100

    return rmse_model, rmse_fvcom, improvement, improvement_pct, bias_model, bias_fvcom


def plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model='SE-Unet-FVCOM-RMSE_model_crop.png',
                   final_path_fvcom='SE-Unet-FVCOM-RMSE_fvcom_crop.png'):

    global_vmin = 0
    global_vmax = max(np.nanmax(rmse_model), np.nanmax(rmse_fvcom))

    lon = np.linspace(117, 127, rmse_model.shape[0])
    lat = np.linspace(35.5, 41.5, rmse_model.shape[1])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    def draw_and_crop(data, title, final_path):
        vmin = global_vmin
        vmax = global_vmax

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.75],
                          projection=ccrs.PlateCarree())

        im = ax.pcolormesh(
            lon_grid, lat_grid, data.T,
            cmap='turbo',
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree(),
            zorder=1
        )

        ax.set_extent([117, 127, 35.5, 41.5],
                      crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       facecolor='white',
                       edgecolor='black',
                       linewidth=0.3,
                       zorder=3)

        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=0.6,
                       edgecolor='black',
                       zorder=4)

        xticks = np.arange(117, 127.1, 2)
        yticks = np.arange(35.5, 41.6, 1)


        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels([f"{x}°E" for x in xticks])
        ax.set_yticklabels([f"{y}°N" for y in yticks])

        ax.tick_params(axis='both', labelsize=16)

        ax.set_title(title, fontsize=18,
                     fontweight='bold', pad=10)

        pos = ax.get_position()

        cax = fig.add_axes([
            pos.x1 + 0.015,
            pos.y0,
            0.02,
            pos.height
        ])

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('RMSE(℃) ', fontsize=16)
        cbar.ax.tick_params(labelsize=13)

        buf = BytesIO()
        plt.savefig(buf, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.005)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img_data = np.array(img)

        mask = np.any(img_data < 250, axis=2)
        coords = np.argwhere(mask)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        margin = 60
        y0 = max(y0 - margin, 0)
        x0 = max(x0 - margin, 0)
        y1 = min(y1 + margin, img.height)
        x1 = min(x1 + margin, img.width)

        cropped = img.crop((x0, y0, x1, y1))
        cropped.save(final_path)

        print(f"{title} saved: {final_path}")

    draw_and_crop(rmse_fvcom, 'FVCOM RMSE ', final_path_fvcom)
    draw_and_crop(rmse_model, 'SE-Unet-FVCOM RMSE ', final_path_model)


def plot_mean_temperature_map_comparison(
        model, test_dataset,
        lon, lat,
        means_X, stds_X,
        mean_Y, std_Y,
        device,
        save_path_sat='SE-Unet-FVCOM-SST_satellite_mean.png',
        save_path_fvcom='SE-Unet-FVCOM-SST_fvcom_mean.png',
        save_path_model='SE-Unet-FVCOM-SST_model_mean.png'):

    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fvcom_all, pred_all, sst_all = [], [], []
    fvcom_channel = 5

    with torch.no_grad():
        for x, y, mask in loader:

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)

            pred_np = pred.cpu().numpy()[0, 0]
            y_np = y.cpu().numpy()[0]
            x_np = x.cpu().numpy()[0]
            mask_np = mask.cpu().numpy()[0]

            pred_denorm = pred_np * std_Y + mean_Y
            sst_denorm = y_np * std_Y + mean_Y
            fvcom_denorm = x_np[fvcom_channel] * stds_X[fvcom_channel] + means_X[fvcom_channel]

            pred_denorm = np.where(mask_np == 1, pred_denorm, np.nan)
            sst_denorm = np.where(mask_np == 1, sst_denorm, np.nan)
            fvcom_denorm = np.where(mask_np == 1, fvcom_denorm, np.nan)

            pred_all.append(pred_denorm)
            sst_all.append(sst_denorm)
            fvcom_all.append(fvcom_denorm)

    pred_mean = np.nanmean(np.stack(pred_all), axis=0)
    sst_mean = np.nanmean(np.stack(sst_all), axis=0)
    fvcom_mean = np.nanmean(np.stack(fvcom_all), axis=0)

    vmin = max(-2, min(np.nanmin(sst_mean),
                      np.nanmin(fvcom_mean),
                      np.nanmin(pred_mean))-2)

    vmax = min(30, max(np.nanmax(sst_mean),
                       np.nanmax(fvcom_mean),
                       np.nanmax(pred_mean))+2)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    def draw_single(data, title, save_path):

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.75],
                          projection=ccrs.PlateCarree())

        im = ax.pcolormesh(
            lon_grid, lat_grid, data.T,
            cmap='turbo',
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree(),
            zorder=1
        )

        ax.set_extent([lon.min(), lon.max(),
                       lat.min(), lat.max()],
                      crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       facecolor='white',
                       edgecolor='black',
                       linewidth=0.3,
                       zorder=3)

        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=0.5,
                       edgecolor='black',
                       zorder=5)

        xticks = np.arange(117, 127.1, 2)
        yticks = np.arange(35.5, 41.6, 1)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels([f"{x:.1f}°E" for x in xticks],
                           fontsize=16)

        ax.set_yticklabels([f"{y:.1f}°N" for y in yticks],
                           fontsize=16)

        ax.set_title(title,
                     fontsize=18,
                     fontweight='bold',
                     pad=10)

        pos = ax.get_position()

        cax = fig.add_axes([
            pos.x1 + 0.015,
            pos.y0,
            0.02,
            pos.height
        ])

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('SST (°C)', fontsize=16)
        cbar.ax.tick_params(labelsize=13)

        buf = BytesIO()
        plt.savefig(buf, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.005)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img_data = np.array(img)

        mask = np.any(img_data < 250, axis=2)
        coords = np.argwhere(mask)

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        margin = 60
        y0 = max(y0 - margin, 0)
        x0 = max(x0 - margin, 0)
        y1 = min(y1 + margin, img.height)
        x1 = min(x1 + margin, img.width)

        img.crop((x0, y0, x1, y1)).save(save_path)

        print(f"{title} saved: {save_path}")

    draw_single(sst_mean, 'Satellite SST', save_path_sat)
    draw_single(fvcom_mean, 'FVCOM SST', save_path_fvcom)
    draw_single(pred_mean, 'SE-Unet-FVCOM SST', save_path_model)

def plot_bias_maps(bias_model, bias_fvcom,
                   save_path_model,
                   save_path_fvcom):

    vmax = max(
        np.nanmax(np.abs(bias_model)),
        np.nanmax(np.abs(bias_fvcom))
    )
    vmin = -vmax

    lon = np.linspace(117, 127, bias_model.shape[0])
    lat = np.linspace(35.5, 41.5, bias_model.shape[1])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    def draw(data, title, save_path):

        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.75],
                          projection=ccrs.PlateCarree())

        im = ax.pcolormesh(
            lon_grid, lat_grid, data.T,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        ax.set_extent([117, 127, 35.5, 41.5])

        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       facecolor='white',
                       edgecolor='black',
                       linewidth=0.3)

        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=0.5)

        xticks = np.arange(117, 127.1, 2)
        yticks = np.arange(35.5, 41.6, 1)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels([f"{x}°E" for x in xticks], fontsize=16)
        ax.set_yticklabels([f"{y}°N" for y in yticks], fontsize=16)

        ax.set_title(title, fontsize=18, fontweight='bold')

        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height])

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Bias (°C)', fontsize=16)
        cbar.ax.tick_params(labelsize=13)

        plt.savefig(save_path, dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.005)
        plt.close()

        print(f"{title} saved: {save_path}")

    draw(bias_fvcom, "FVCOM Bias", save_path_fvcom)
    draw(bias_model, "SE-Unet-FVCOM Bias", save_path_model)


def main():
    set_seed(46)
    DATA_PATH = r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'
    output_dir = r'./SE-Unet-results_2010_2023'

    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")

    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data(DATA_PATH)
    train, val, test, test_idx = split_data(X, Y, mask)
    train_ds, val_ds, test_ds = OceanDataset(*train), OceanDataset(*val), OceanDataset(*test)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(in_channels=10, out_channels=1)

    best_model_path = r'D:\CNN\SE_Unet模型\results_UNet_SE_year_2010_2023\best_resunet_model_seunet.pth'

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    print("Loaded the best validation model for evaluation")

    results = evaluate_model(
        model,
        test_ds,
        means,
        stds,
        Y_mean,
        Y_std,
        device
    )
    results_df = pd.DataFrame([results])

    save_path = os.path.join(output_dir, "SE-Unet-evaluation_metrics.csv")
    results_df.to_csv(save_path, index=False)

    print(f"RMSE and MAE results saved to: {save_path}")

    print("\nStep 6: Computing spatial RMSE maps")
    rmse_model, rmse_fvcom, improvement, improvement_pct, bias_model, bias_fvcom = compute_spatial_rmse_maps(
        model,
        test_ds,
        Y_mean, Y_std,
        means, stds,
        device
    )

    np.save(os.path.join(output_dir, "SE-Unet-FVCOM_bias_model.npy"), bias_model)
    np.save(os.path.join(output_dir, "SE-Unet-FVCOM_bias_fvcom.npy"), bias_fvcom)

    np.save(os.path.join(output_dir, "SE-Unet-FVCOM_rmse_model.npy"), rmse_model)
    np.save(os.path.join(output_dir, "SE-Unet-FVCOM_rmse_fvcom.npy"), rmse_fvcom)

    np.save(os.path.join(output_dir, "SE-Unet-FVCOM_improvement.npy"), improvement)

    print("Spatial statistics saved (for post-processing plots)")

    plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model=os.path.join(output_dir, 'SE-Unet-FVCOM-model_rmse_final.png'),
                   final_path_fvcom=os.path.join(output_dir, 'SE-Unet-FVCOM-fvcom_rmse_final.png'))

    plot_bias_maps(
        bias_model,
        bias_fvcom,
        save_path_model=os.path.join(output_dir, 'SE-Unet-FVCOM-bias_model.png'),
        save_path_fvcom=os.path.join(output_dir, 'SE-Unet-FVCOM-bias_fvcom.png')
    )

    print("\nStep 8: Plotting spatial mean temperature comparison")
    plot_mean_temperature_map_comparison(
        model=model,
        test_dataset=test_ds,
        lon=lon,
        lat=lat,
        means_X=means,
        stds_X=stds,
        mean_Y=Y_mean,
        std_Y=Y_std,
        device=device,

        save_path_sat=os.path.join(output_dir, 'SE-Unet-FVCOM-SST_satellite_mean.png'),
        save_path_fvcom=os.path.join(output_dir, 'SE-Unet-FVCOM-SST_fvcom_mean.png'),
        save_path_model=os.path.join(output_dir, 'SE-Unet-FVCOM-SST_model_mean.png')
    )


if __name__ == "__main__":
    main()
