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
import torch.optim as optim
from tqdm import tqdm


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
    print(f"Random seed set: {seed}")


class OceanDataset_winter(Dataset):
    def __init__(self, X, Y, mask):
        self.X = X
        self.Y = Y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32),
                torch.tensor(self.mask[idx], dtype=torch.float32))


def load_data_winter(file_path=r'D:\CNN\冬季数据处理\output_2010_2023_winter\data_with_mask_2010_2023_winter.mat'):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    print(f"Original data shape: {data.shape}")
    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :]
    Y = data[:, 0, :, :]
    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]
    time = data[:, 12, 0, 0]

    mask = ~np.isnan(Y)
    valid_samples = np.any(mask, axis=(1, 2))
    X, Y, mask = X[valid_samples], Y[valid_samples], mask[valid_samples]

    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    Y = np.where(np.isnan(Y), 0, Y)
    Y_normalized = (Y - Y_mean) / (Y_std + 1e-8)

    mask = mask.astype(np.float32)

    means = np.zeros(X.shape[1])
    stds = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        xi = X[:, i]
        valid = ~np.isnan(xi)
        means[i] = np.nanmean(xi)
        stds[i] = np.nanstd(xi)
        xi[~valid] = means[i]
        X[:, i] = xi

    X = (X - means.reshape(1, -1, 1, 1)) / (stds.reshape(1, -1, 1, 1) + 1e-8)

    return X, Y_normalized, mask, lon, lat, time, means, stds, Y_mean, Y_std


def split_data_sequential_winter(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)

    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    print("Time-series data split:")
    print(f"   Train set: first {len(train_idx)} time points (index: {train_idx[0]} - {train_idx[-1]})")
    print(f"   Validation set: middle {len(val_idx)} time points (index: {val_idx[0]} - {val_idx[-1]})")
    print(f"   Test set: last {len(test_idx)} time points (index: {test_idx[0]} - {test_idx[-1]})")

    X_train, Y_train, mask_train = X[train_idx], Y[train_idx], mask[train_idx]
    X_val, Y_val, mask_val = X[val_idx], Y[val_idx], mask[val_idx]
    X_test, Y_test, mask_test = X[test_idx], Y[test_idx], mask[test_idx]

    return (X_train, Y_train, mask_train), \
        (X_val, Y_val, mask_val), \
        (X_test, Y_test, mask_test), \
        test_idx


class ResBlock_winter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class CNNResLSTM_winter(nn.Module):
    def __init__(self, in_channels=10, input_height=200, input_width=120,
                 lstm_hidden=256, downscale=0.25):
        super().__init__()
        self.downscale = downscale
        self.orig_height = input_height
        self.orig_width = input_width
        self.height = int(input_height * downscale)
        self.width = int(input_width * downscale)
        self.lstm_hidden = lstm_hidden

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_winter(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_winter(128)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=False)
        self.lstm_fc = nn.Linear(lstm_hidden, 128)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_winter(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(self.height, self.width),
                                      mode='bilinear', align_corners=False)

        B = x.size(0)
        encoded = self.encoder(x)

        spatial_seq = encoded.view(B, 128, self.height * self.width).permute(0, 2, 1)

        lstm_out, (hn, cn) = self.lstm(spatial_seq)

        lstm_processed = self.lstm_fc(lstm_out)
        lstm_features = lstm_processed.permute(0, 2, 1).view(B, 128, self.height, self.width)

        decoded = self.decoder(lstm_features)

        output = nn.functional.interpolate(decoded, size=(self.orig_height, self.orig_width),
                                           mode='bilinear', align_corners=False)

        return output.squeeze(1)


def train_model_winter(
        model,
        train_dataset,
        val_dataset,
        device,
        num_epochs=100,
        batch_size=8,
        lr=1e-3,
        save_dir='.',
        early_stop_patience=10,
        early_stop_min_delta=1e-4
):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_epoch = -1
    no_improve_epochs = 0

    best_model_path = os.path.join(
        save_dir, 'best_CNNLSTM_model_winter_2010_2023.pth'
    )
    last_model_path = os.path.join(
        save_dir, 'last_CNNLSTM_model_winter_2010_2023.pth'
    )

    print("\nStarting model training (Early Stopping enabled)")
    print(f"Early Stop Patience = {early_stop_patience}, "
          f"Min Delta = {early_stop_min_delta}")

    for epoch in range(1, num_epochs + 1):

        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            preds = model(batch_x)
            diff = preds - batch_y
            loss = (diff ** 2 * batch_mask).sum() / (batch_mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)

                preds = model(batch_x)
                diff = preds - batch_y
                batch_loss = (diff ** 2 * batch_mask).sum() / (batch_mask.sum() + 1e-8)
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        improved = val_loss < (best_val_loss - early_stop_min_delta)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            tag = "BEST"
        else:
            no_improve_epochs += 1
            tag = ""

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {epoch_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"No Improve: {no_improve_epochs}/{early_stop_patience} "
                f"{tag}"
            )

        if no_improve_epochs >= early_stop_patience:
            print("\nEarly Stopping triggered")
            print(f"Validation loss has not improved significantly for {early_stop_patience} consecutive epochs")
            print(f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}")
            break

    torch.save(model.state_dict(), last_model_path)

    print("\nTraining finished")
    print(f"Best model epoch: {best_epoch}")
    print(f"Best model path: {best_model_path}")
    print(f"Final model path: {last_model_path}")

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    if best_epoch > 0:
        plt.axvline(best_epoch - 1, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ConvLSTM 训练 / 验证损失曲线（Early Stopping）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_CNNLSTM_winter_2010_2023.png', dpi=300)
    plt.show()

    return train_losses, val_losses, best_model_path


def evaluate_full_testset_metrics_winter(
        model,
        test_ds,
        means,
        stds,
        Y_mean,
        Y_std,
        device,
        verbose=True
):

    model.eval()
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    sum_sq_err_fvcom = 0.0
    sum_sq_err_pred = 0.0
    sum_abs_err_fvcom = 0.0
    sum_abs_err_pred = 0.0

    total_valid_pixels = 0
    processed_samples = 0

    if verbose:
        print("Starting full test set evaluation ...")

    with torch.no_grad():
        for x, y, m in tqdm(loader, desc="Evaluating test set", disable=not verbose):
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x).squeeze().cpu().numpy()

            pred_denorm = pred * Y_std + Y_mean
            y_denorm = y.squeeze().cpu().numpy() * Y_std + Y_mean
            m = m.squeeze().cpu().numpy()

            fvcom = x[0, 5].cpu().numpy() * stds[5] + means[5]

            valid = (m == 1)
            n_valid = np.sum(valid)

            if n_valid > 0:
                sat_valid = y_denorm[valid]
                fvcom_valid = fvcom[valid]
                pred_valid = pred_denorm[valid]

                sum_sq_err_fvcom += np.sum((fvcom_valid - sat_valid) ** 2)
                sum_sq_err_pred += np.sum((pred_valid - sat_valid) ** 2)

                sum_abs_err_fvcom += np.sum(np.abs(fvcom_valid - sat_valid))
                sum_abs_err_pred += np.sum(np.abs(pred_valid - sat_valid))

                total_valid_pixels += n_valid

            processed_samples += 1

    if total_valid_pixels == 0:
        print("No valid pixels, evaluation failed")
        return None

    fvcom_rmse = np.sqrt(sum_sq_err_fvcom / total_valid_pixels)
    model_rmse = np.sqrt(sum_sq_err_pred / total_valid_pixels)

    fvcom_mae = sum_abs_err_fvcom / total_valid_pixels
    model_mae = sum_abs_err_pred / total_valid_pixels

    rmse_improvement = (fvcom_rmse - model_rmse) / fvcom_rmse * 100
    mae_improvement = (fvcom_mae - model_mae) / fvcom_mae * 100

    if verbose:
        print(f"\nEvaluation completed: {processed_samples} samples, "
              f"{total_valid_pixels:,} valid pixels")

        print("\nFull test set evaluation results:")
        print(f"FVCOM RMSE : {fvcom_rmse:.4f} °C")
        print(f"Model RMSE : {model_rmse:.4f} deg C")
        print(f"RMSE improvement: {rmse_improvement:.2f}%")

        print(f"FVCOM MAE  : {fvcom_mae:.4f} °C")
        print(f"Model MAE  : {model_mae:.4f} deg C")
        print(f"MAE improvement : {mae_improvement:.2f}%")

    return {
        "fvcom_rmse": fvcom_rmse,
        "model_rmse": model_rmse,
        "fvcom_mae": fvcom_mae,
        "model_mae": model_mae,
        "rmse_improvement": rmse_improvement,
        "mae_improvement": mae_improvement,
        "total_pixels": total_valid_pixels,
        "n_samples": processed_samples
    }


def compute_spatial_rmse_maps(model, test_dataset,
                              mean_Y, std_Y,
                              means_X, stds_X,
                              device,
                              fvcom_channel=5):

    model.eval()

    pred_list = []
    obs_list = []
    fvcom_list = []

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

            pred = np.where(mask == 1, pred, np.nan)
            obs = np.where(mask == 1, obs, np.nan)
            fvcom = np.where(mask == 1, fvcom, np.nan)

            pred_list.append(pred)
            obs_list.append(obs)
            fvcom_list.append(fvcom)

    pred = np.stack(pred_list)
    obs = np.stack(obs_list)
    fvcom = np.stack(fvcom_list)

    rmse_model = np.sqrt(np.nanmean((pred - obs) ** 2, axis=0))
    rmse_fvcom = np.sqrt(np.nanmean((fvcom - obs) ** 2, axis=0))

    improvement = rmse_fvcom - rmse_model

    return rmse_model, rmse_fvcom, improvement


def plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model='Winter-CNN-LSTM-FVCOM-RMSE_model_crop.png',
                   final_path_fvcom='Winter-CNN-LSTM-FVCOM-RMSE_fvcom_crop.png'):

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

    draw_and_crop(rmse_fvcom, 'Winter FVCOM RMSE ', final_path_fvcom)
    draw_and_crop(rmse_model, 'Winter CNN-LSTM-FVCOM RMSE ', final_path_model)


def plot_mean_temperature_map_comparison(
        model, test_dataset,
        lon, lat,
        means_X, stds_X,
        mean_Y, std_Y,
        device,
        save_path_sat='Winter-CNN-LSTM-FVCOM-SST_satellite_mean.png',
        save_path_fvcom='Winter-CNN-LSTM-FVCOM-SST_fvcom_mean.png',
        save_path_model='Winter-CNN-LSTM-FVCOM-SST_model_mean.png'):

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

    vmin = max(0, min(np.nanmin(sst_mean),
                      np.nanmin(fvcom_mean),
                      np.nanmin(pred_mean)) - 2)

    vmax = min(30, max(np.nanmax(sst_mean),
                       np.nanmax(fvcom_mean),
                       np.nanmax(pred_mean)) + 2)

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

    draw_single(sst_mean, 'Winter Satellite SST', save_path_sat)
    draw_single(fvcom_mean, 'Winter FVCOM SST', save_path_fvcom)
    draw_single(pred_mean, 'Winter CNN-LSTM-FVCOM SST', save_path_model)


def main_winter():
    output_dir = r'./冬季-CNN-LSTM-results_2010_2023'

    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")

    set_seed(46)
    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data_winter(
        r'D:\CNN\冬季数据处理\output_2010_2023_winter\data_with_mask_2010_2023_winter.mat')
    train, val, test, test_idx = split_data_sequential_winter(X, Y, mask)
    train_ds = OceanDataset_winter(*train)
    val_ds = OceanDataset_winter(*val)
    test_ds = OceanDataset_winter(*test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = CNNResLSTM_winter(in_channels=10, input_height=X.shape[2], input_width=X.shape[3])

    model_path = 'best_CNNLSTM_model_winter_2010_2023.pth'

    print(f"\nLoading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    print("\n" + "=" * 60)
    print("Full Test Set Evaluation")
    print("=" * 60)
    eval_results = evaluate_full_testset_metrics_winter(model, test_ds, means, stds, Y_mean, Y_std, device)

    results_df = pd.DataFrame([eval_results])

    csv_path = "CNNLSTM_冬季_test_metrics_2010_2023.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\nEvaluation metrics saved to: {csv_path}")
    print(results_df)

    print("\nStep 6: Computing spatial RMSE distribution")
    rmse_model, rmse_fvcom, improvement = compute_spatial_rmse_maps(
        model,
        test_ds,
        Y_mean, Y_std,
        means, stds,
        device
    )

    plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model=os.path.join(output_dir, 'Winter-CNN-LSTM-FVCOM-model_rmse_final.png'),
                   final_path_fvcom=os.path.join(output_dir, 'Winter-CNN-LSTM-FVCOM-fvcom_rmse_final.png'))

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

        save_path_sat=os.path.join(output_dir, 'Winter-CNN-LSTM-FVCOM-SST_satellite_mean.png'),
        save_path_fvcom=os.path.join(output_dir, 'Winter-CNN-LSTM-FVCOM-SST_fvcom_mean.png'),
        save_path_model=os.path.join(output_dir, 'Winter-CNN-LSTM-FVCOM-SST_model_mean.png')
    )


if __name__ == '__main__':
    main_winter()
