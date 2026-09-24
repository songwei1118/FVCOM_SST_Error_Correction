import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    return device


def load_ocean_data(file_path, normalize=True):
    print(f"Loading data: {file_path}")

    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]

    print(f"Original data shape: {data.shape}")

    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Transposed shape: {data.shape}")

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :]
    Y = data[:, 0, :, :]
    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]
    time = data[:, 12, 0, 0]

    mask = ~np.isnan(Y)
    valid_samples = np.any(mask, axis=(1, 2))
    X, Y, mask = X[valid_samples], Y[valid_samples], mask[valid_samples]
    Y = np.where(np.isnan(Y), 0, Y)
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

        Y_valid = Y[mask == 1]
        mean_Y = np.nanmean(Y_valid)
        std_Y = np.nanstd(Y_valid)
        Y = (Y - mean_Y) / (std_Y + 1e-8)

        print(f"Target normalization parameters: mean={mean_Y:.4f}, std={std_Y:.4f}")
    else:
        means_X, stds_X, mean_Y, std_Y = None, None, None, None

    print(f"Data loading completed:")
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Longitude range: {lon.min():.2f} - {lon.max():.2f}")
    print(f"   Latitude range: {lat.min():.2f} - {lat.max():.2f}")
    print(f"   Time length: {len(time)} days")

    return {
        'X': X, 'Y': Y, 'mask': mask,
        'lon': lon, 'lat': lat, 'time': time,
        'means_X': means_X, 'stds_X': stds_X,
        'mean_Y': mean_Y, 'std_Y': std_Y
    }


def split_data_sequential(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)

    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    print(f"Time series data split:")
    print(f"   Training set: first {len(train_idx)} time points ({train_idx[0]} - {train_idx[-1]})")
    print(f"   Validation set: middle {len(val_idx)} time points ({val_idx[0]} - {val_idx[-1]})")
    print(f"   Test set: last {len(test_idx)} time points ({test_idx[0]} - {test_idx[-1]})")

    return (X[train_idx], Y[train_idx], mask[train_idx]), \
        (X[val_idx], Y[val_idx], mask[val_idx]), \
        (X[test_idx], Y[test_idx], mask[test_idx]), \
        test_idx


class OceanSSTDataset(Dataset):

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
            torch.tensor(self.mask[idx], dtype=torch.float32)
        )

class MemoryEfficientOceanSSTModel(nn.Module):

    def __init__(self,
                 in_channels=10):
        super().__init__()

        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        self.residual_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        x_in = x

        x = self.input_conv(x)

        out = self.output_conv(x)

        residual = self.residual_conv(x_in)

        out = out + self.residual_alpha * residual

        return out.squeeze(1)

    def count_parameters(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total": total_params,
            "trainable": trainable_params
        }

class MaskedHuberSmoothLoss(nn.Module):
    def __init__(self, delta=1.0, smooth_weight=0.05):
        super().__init__()
        self.delta = delta
        self.smooth_weight = smooth_weight

    def huber(self, diff):
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(self.delta, device=diff.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic ** 2 + self.delta * linear

    def spatial_smoothness(self, pred, mask):
        mask_x = mask[:, :, 1:] * mask[:, :, :-1]
        mask_y = mask[:, 1:, :] * mask[:, :-1, :]

        dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1]) * mask_x
        dy = torch.abs(pred[:, 1:, :] - pred[:, :-1, :]) * mask_y

        return (dx.sum() + dy.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)

    def forward(self, pred, target, mask):
        diff = pred - target

        huber_loss = self.huber(diff)
        masked_huber = (huber_loss * mask).sum() / (mask.sum() + 1e-8)

        smooth_loss = self.spatial_smoothness(pred, mask)

        return masked_huber + self.smooth_weight * smooth_loss


class Trainer:

    def __init__(self, model, device, lr=1e-3,
                 early_stop_patience=10,
                 early_stop_min_delta=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        self.criterion = MaskedHuberSmoothLoss(
            delta=1.0,
            smooth_weight=0.03
        )

        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def train_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for x, y, mask in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = self.criterion(pred, y, mask)


            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)

        return avg_loss

    def validate(self, val_loader, epoch, num_epochs):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for x, y, mask in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(x)

                loss = self.criterion(pred, y, mask)


                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        self.history['val_loss'].append(avg_loss)

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=100,
              save_path='单尺度CNN-Transformer-绝对位置编码-动态学习率2010-2023-有气压_best_model.pth'):

        print("Starting training...")
        params_info = self.model.count_parameters()
        print(f"Model parameters:")
        print(f"   Total parameters: {params_info['total']:,}")
        print(f"   Trainable parameters: {params_info['trainable']:,}")
        print(f"Using dynamic learning rate scheduler: ReduceLROnPlateau")
        print(f"Early Stopping enabled: patience={self.early_stop_patience}, "
              f"min_delta={self.early_stop_min_delta}")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            val_loss = self.validate(val_loader, epoch, num_epochs)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.2e}")

            if val_loss < self.best_val_loss - self.early_stop_min_delta:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)

                print(f"Saving best model (Val Loss={val_loss:.6f})")

            else:
                self.early_stop_counter += 1
                print(f"EarlyStopping counter: "
                      f"{self.early_stop_counter}/{self.early_stop_patience}")

            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly Stopping triggered at epoch {epoch + 1}")
                break

        print(f"Training finished, best validation loss: {self.best_val_loss:.6f}")
        return self.history


def evaluate_model(model, test_loader, device,
                   mean_Y, std_Y,
                   means_X, stds_X):

    model.eval()

    fvcom_channel_in_X = 5

    sum_sq_err_fvcom = 0.0
    sum_sq_err_model = 0.0
    sum_abs_err_fvcom = 0.0
    sum_abs_err_model = 0.0
    n_valid = 0

    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)

            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            pred_denorm = pred_np * std_Y + mean_Y
            y_denorm = y_np * std_Y + mean_Y

            fvcom_mean = means_X[fvcom_channel_in_X]
            fvcom_std = stds_X[fvcom_channel_in_X]
            fvcom_denorm = x_np[:, fvcom_channel_in_X, :, :] * fvcom_std + fvcom_mean

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

    fvcom_rmse = np.sqrt(sum_sq_err_fvcom / n_valid)
    model_rmse = np.sqrt(sum_sq_err_model / n_valid)

    fvcom_mae = sum_abs_err_fvcom / n_valid
    model_mae = sum_abs_err_model / n_valid

    rmse_improvement = (fvcom_rmse - model_rmse) / fvcom_rmse * 100
    mae_improvement = (fvcom_mae - model_mae) / fvcom_mae * 100

    print("\nOverall test set evaluation results (all samples)")
    print(f"FVCOM RMSE : {fvcom_rmse:.4f} °C")
    print(f"Model  RMSE : {model_rmse:.4f} °C")
    print(f"RMSE improvement : {rmse_improvement:.2f} %")
    print(f"FVCOM MAE  : {fvcom_mae:.4f} °C")
    print(f"Model  MAE  : {model_mae:.4f} °C")
    print(f"MAE improvement  : {mae_improvement:.2f} %")

    return {
        "fvcom_rmse": fvcom_rmse,
        "model_rmse": model_rmse,
        "rmse_improvement": rmse_improvement,
        "fvcom_mae": fvcom_mae,
        "model_mae": model_mae,
        "mae_improvement": mae_improvement
    }


def compute_spatial_rmse_maps(model, test_dataset,
                              mean_Y, std_Y,
                              means_X, stds_X,
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

            pred = pred.cpu().numpy()[0]
            y = y.numpy()[0]
            mask = mask.numpy()[0]
            x_np = x.cpu().numpy()[0]

            pred = pred * std_Y + mean_Y
            obs = y * std_Y + mean_Y
            fvcom = x_np[fvcom_channel] * stds_X[fvcom_channel] + means_X[fvcom_channel]

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
                   final_path_model='消融实验-单尺度CNN-FVCOM-RMSE_model_crop.png',
                   final_path_fvcom='消融实验-单尺度CNN-FVCOM-RMSE_fvcom_crop.png'):

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
    draw_and_crop(rmse_model, 'CNN-FVCOM RMSE ', final_path_model)


def plot_mean_temperature_map_comparison(
        model, test_dataset,
        lon, lat,
        means_X, stds_X,
        mean_Y, std_Y,
        device,
        save_path_sat='消融实验-单尺度CNN-FVCOM-SST_satellite_mean.png',
        save_path_fvcom='消融实验-单尺度CNN-FVCOM-SST_fvcom_mean.png',
        save_path_model='消融实验-单尺度CNN-FVCOM-SST_model_mean.png'):

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

            pred_np = pred.cpu().numpy()[0]
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
    draw_single(pred_mean, 'CNN-FVCOM SST', save_path_model)


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
    draw(bias_model, "CNN-FVCOM Bias", save_path_model)


def main():

    output_dir = r'./消融实验-单尺度C-FVCOM-results_2010_2023'

    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")


    set_seed(46)

    device = get_device()

    print("\nStep 1: Loading data")

    data = load_ocean_data(
        r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat',
        normalize=True
    )

    print("\nStep 2: Splitting dataset")
    (X_train, Y_train, mask_train), \
        (X_val, Y_val, mask_val), \
        (X_test, Y_test, mask_test), test_idx = split_data_sequential(
        data['X'], data['Y'], data['mask']
    )

    print("\nStep 3: Creating datasets")
    train_dataset = OceanSSTDataset(X_train, Y_train, mask_train)
    val_dataset = OceanSSTDataset(X_val, Y_val, mask_val)
    test_dataset = OceanSSTDataset(X_test, Y_test, mask_test)

    batch_size = 4
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\nStep 4: Creating single-scale CNN model")

    model = MemoryEfficientOceanSSTModel(
        in_channels=10
    )

    print("\nModel structure:")
    print(model)

    model.to(device)

    best_model_path = '单尺度CNN-Residual_best_model.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Best model loaded")
    print(f"   Best epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown')}")

    print("\nStep X: Saving test set time series")

    print("\nStep 7: Evaluating model")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y'],
        means_X=data['means_X'],
        stds_X=data['stds_X']
    )

    results_df = pd.DataFrame([results])

    save_path = os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM-evaluation_metrics.csv")
    results_df.to_csv(save_path, index=False)

    print(f"RMSE and MAE results saved to: {save_path}")


    print("\nStep 6: Computing spatial RMSE distribution")
    rmse_model, rmse_fvcom, improvement, improvement_pct, bias_model, bias_fvcom = compute_spatial_rmse_maps(
        model,
        test_dataset,
        data['mean_Y'], data['std_Y'],
        data['means_X'], data['stds_X'],
        device
    )


    np.save(os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM_bias_model.npy"), bias_model)
    np.save(os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM_bias_fvcom.npy"), bias_fvcom)

    np.save(os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM_rmse_model.npy"), rmse_model)
    np.save(os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM_rmse_fvcom.npy"), rmse_fvcom)

    np.save(os.path.join(output_dir, "消融实验-单尺度CNN-FVCOM_improvement.npy"), improvement)

    print("Spatial statistics saved (for post-processing plots)")

    plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-model_rmse_final.png'),
                   final_path_fvcom=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-fvcom_rmse_final.png'))

    print("\nStep 7: Plotting Bias Maps")

    plot_bias_maps(
        bias_model,
        bias_fvcom,
        save_path_model=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-bias_model.png'),
        save_path_fvcom=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-bias_fvcom.png')
    )


    print("\nStep 8: Plotting spatial mean temperature comparison")
    plot_mean_temperature_map_comparison(
        model=model,
        test_dataset=test_dataset,
        lon=data['lon'],
        lat=data['lat'],
        means_X=data['means_X'],
        stds_X=data['stds_X'],
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y'],
        device=device,

        save_path_sat=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-SST_satellite_mean.png'),
        save_path_fvcom=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-SST_fvcom_mean.png'),
        save_path_model=os.path.join(output_dir, '消融实验-单尺度CNN-FVCOM-SST_model_mean.png')
    )


if __name__ == '__main__':
    main()
