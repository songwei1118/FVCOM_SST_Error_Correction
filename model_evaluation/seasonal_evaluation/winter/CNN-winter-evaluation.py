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
    print(f"Random seed has been set to {seed}")


class OceanDataset(Dataset):
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


def load_data(file_path=r'D:\CNN\冬季数据处理\output_2010_2023_winter\data_with_mask_2010_2023_winter.mat'):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    print(f"Raw data shape: {data.shape}")

    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    var_names = ['sat_SST', 'U10', 'V10', 'short_wave', 'long_wave', 'air_pressure',  'FVCOM_temp',
                 'FVCOM_km', 'FVCOM_u', 'FVCOM_v', 'lon', 'lat', 'time', 'air_temperature']

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :].astype(np.float32)
    Y = data[:, 0, :, :].astype(np.float32)
    lon = data[0, 10, :, 0].astype(np.float32)
    lat = data[0, 11, 0, :].astype(np.float32)
    time = data[:, 12, 0, 0].astype(np.float32)

    mask = ~np.isnan(Y)
    valid_samples = np.any(mask, axis=(1, 2))
    X, Y, mask = X[valid_samples], Y[valid_samples], mask[valid_samples]
    Y = np.where(np.isnan(Y), 0.0, Y)
    mask = mask.astype(np.float32)

    C = X.shape[1]
    means_X = np.zeros(C)
    stds_X = np.zeros(C)
    for i in range(C):
        xi = X[:, i, :, :]
        valid = ~np.isnan(xi)
        means_X[i] = np.nanmean(xi)
        stds_X[i] = np.nanstd(xi)
        xi[~valid] = means_X[i]
        X[:, i, :, :] = xi
    X = (X - means_X.reshape(1, C, 1, 1)) / (stds_X.reshape(1, C, 1, 1) + 1e-8)

    Y_mean = np.nanmean(Y)
    Y_std = np.nanstd(Y)
    Y = (Y - Y_mean) / (Y_std + 1e-8)

    return X, Y, mask, lon, lat, time, means_X, stds_X, Y_mean, Y_std


def split_data(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    shuffled_train_idx = np.copy(train_idx)
    np.random.shuffle(shuffled_train_idx)

    X_train, Y_train, mask_train = X[shuffled_train_idx], Y[shuffled_train_idx], mask[shuffled_train_idx]
    X_val, Y_val, mask_val = X[val_idx], Y[val_idx], mask[val_idx]
    X_test, Y_test, mask_test = X[test_idx], Y[test_idx], mask[test_idx]
    return (X_train, Y_train, mask_train), (X_val, Y_val, mask_val), (X_test, Y_test, mask_test), test_idx


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return nn.ReLU(inplace=True)(out + x)


class CNNResNetSE(nn.Module):
    def __init__(self, in_channels=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(64, 2)
        self.layer3 = self._make_layer(64, 2)
        self.out = nn.Conv2d(64, 1, 1)

    def _make_layer(self, channels, num_blocks):
        return nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.out(x)


def train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        num_epochs=100,
        batch_size=4,
        lr=1e-3,
        save_dir='.',
        early_stop_patience=10,
        early_stop_min_delta=1e-4
):
    """
    训练模型（带 Early Stopping）
    - 基于验证集 loss 保存最优模型
    - 若验证集 loss 连续 patience 个 epoch 未显著下降，则提前停止
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_epoch = -1
    no_improve_epochs = 0

    best_model_path = os.path.join(save_dir, 'best_cnn_resnet_se_model_2010_2023_冬季.pth')


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

            preds = model(batch_x).squeeze(1)
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

                preds = model(batch_x).squeeze(1)
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

        if epoch % 5 == 0 or epoch == 1 or improved:
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


    print("\nTraining finished")
    print(f"Best model epoch: {best_epoch}")
    print(f"Best model path: {best_model_path}")


    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.axvline(best_epoch - 1, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练 / 验证损失曲线（Early Stopping）')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_cnn_2010_2023_冬季.png', dpi=300)
    plt.show()

    return best_model_path


def evaluate_model(model, test_loader, device,
                   mean_Y, std_Y,
                   means_X, stds_X):
    """
    在【整个测试集】上计算 RMSE 和 MAE
    - FVCOM vs 真实值
    - 模型预测 vs 真实值
    """

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

            pred = model(x).squeeze(1)

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
    print(f"RMSE Improvement : {rmse_improvement:.2f} %")
    print(f"FVCOM MAE  : {fvcom_mae:.4f} °C")
    print(f"Model  MAE  : {model_mae:.4f} °C")
    print(f"MAE Improvement  : {mae_improvement:.2f} %")

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

    pred_list = []
    obs_list = []
    fvcom_list = []

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for x, y, mask in loader:

            x = x.to(device)
            pred = model(x).squeeze(1)

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
                   final_path_model='Winter CNN-FVCOM-RMSE_model_crop.png',
                   final_path_fvcom='Winter CNN-FVCOM-RMSE_fvcom_crop.png'):

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
    draw_and_crop(rmse_model, 'Winter CNN-FVCOM RMSE ', final_path_model)


def plot_mean_temperature_map_comparison(
        model, test_dataset,
        lon, lat,
        means_X, stds_X,
        mean_Y, std_Y,
        device,
        save_path_sat='Winter CNN-FVCOM-SST_satellite_mean.png',
        save_path_fvcom='Winter CNN-FVCOM-SST_fvcom_mean.png',
        save_path_model='Winter CNN-FVCOM-SST_model_mean.png'):

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

            pred_np = pred.cpu().numpy().squeeze()
            y_np = y.cpu().numpy().squeeze()
            x_np = x.cpu().numpy().squeeze()
            mask_np = mask.cpu().numpy().squeeze()

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

    vmin = max(-5, min(np.nanmin(sst_mean),
                      np.nanmin(fvcom_mean),
                      np.nanmin(pred_mean))-2)

    vmax = min(35, max(np.nanmax(sst_mean),
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

    draw_single(sst_mean, 'Winter Satellite SST', save_path_sat)
    draw_single(fvcom_mean, 'Winter FVCOM SST', save_path_fvcom)
    draw_single(pred_mean, 'Winter CNN-FVCOM SST', save_path_model)

def main():
    output_dir = r'./冬季-CNN-FVCOM-results_2010_2023'

    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")

    set_seed(46)

    X, Y, mask, lon, lat, time, means_X, stds_X, Y_mean, Y_std = load_data(
        r'D:\CNN\冬季数据处理\output_2010_2023_winter\data_with_mask_2010_2023_winter.mat'
    )

    train, val, test, test_idx = split_data(X, Y, mask)

    train_ds = OceanDataset(*train)
    val_ds = OceanDataset(*val)
    test_ds = OceanDataset(*test)

    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNResNetSE(in_channels=10).to(device)

    best_model_path = r'D:\CNN\cnn模型\best_cnn_resnet_se_model_2010_2023_冬季.pth'


    print("\nLoading the best validation model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    metrics = evaluate_model(
        model,
        test_loader,
        device,
        Y_mean, Y_std,
        means_X, stds_X
    )

    df = pd.DataFrame([metrics])

    csv_path = 'evaluation_metrics_CNN_2010_2023_冬季.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"Evaluation results saved: {csv_path}")


    print("\nStep 6: Computing spatial RMSE distribution")
    rmse_model, rmse_fvcom, improvement = compute_spatial_rmse_maps(
        model,
        test_ds,
        Y_mean, Y_std,
        means_X, stds_X,
        device
    )

    plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model=os.path.join(output_dir, 'Winter-CNN-FVCOM-model_rmse_final.png'),
                   final_path_fvcom=os.path.join(output_dir, 'Winter-CNN-FVCOM-fvcom_rmse_final.png'))


    print("\nStep 8: Plotting spatial mean temperature comparison")
    plot_mean_temperature_map_comparison(
        model=model,
        test_dataset=test_ds,
        lon=lon,
        lat=lat,
        means_X=means_X,
        stds_X=stds_X,
        mean_Y=Y_mean,
        std_Y=Y_std,
        device=device,

        save_path_sat=os.path.join(output_dir, 'Winter-CNN-FVCOM-SST_satellite_mean.png'),
        save_path_fvcom=os.path.join(output_dir, 'Winter-CNN-FVCOM-SST_fvcom_mean.png'),
        save_path_model=os.path.join(output_dir, 'Winter-CNN-FVCOM-SST_model_mean.png')
    )


if __name__ == '__main__':
    main()
