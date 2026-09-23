import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16


def set_seed(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set: {seed}")


class OceanDataset_summer(Dataset):
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


def load_data_summer(file_path=r'D:\CNN\夏季数据处理\output_2010_2023_summer\data_with_mask_2010_2023_summer.mat'):
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


def split_data_sequential_summer(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
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


class ResBlock_summer(nn.Module):
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


class CNNResLSTM_summer(nn.Module):
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
            ResBlock_summer(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_summer(128)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=False)
        self.lstm_fc = nn.Linear(lstm_hidden, 128)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_summer(64),
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


def train_model_summer(
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
        save_dir, 'best_CNNLSTM_model_summer_2010_2023.pth'
    )
    last_model_path = os.path.join(
        save_dir, 'last_CNNLSTM_model_summer_2010_2023.pth'
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
    plt.savefig('loss_curve_CNNLSTM_summer_2010_2023.png', dpi=300)
    plt.show()

    return train_losses, val_losses, best_model_path


def evaluate_full_testset_metrics_summer(
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


def plot_temperature_comparison_simple_summer(model, test_ds, test_idx, means, stds, Y_mean, Y_std, device, last_n=60):
    model.eval()
    loader = DataLoader(test_ds, batch_size=1)

    sst_temps = []
    fvcom_temps = []
    adjusted_temps = []

    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x).squeeze().cpu().numpy()

            pred_denorm = pred * Y_std + Y_mean

            fvcom = x[0, 5].cpu().numpy() * stds[5] + means[5]
            sst = y.squeeze().cpu().numpy() * Y_std + Y_mean
            m = m.squeeze().cpu().numpy()

            valid_mask = (m == 1)
            if np.any(valid_mask):
                sst_temps.append(np.nanmean(sst[valid_mask]))
                fvcom_temps.append(np.nanmean(fvcom[valid_mask]))
                adjusted_temps.append(np.nanmean(pred_denorm[valid_mask]))
            else:
                sst_temps.append(np.nan)
                fvcom_temps.append(np.nan)
                adjusted_temps.append(np.nan)

    sst_temps = np.array(sst_temps)
    fvcom_temps = np.array(fvcom_temps)
    adjusted_temps = np.array(adjusted_temps)

    if len(sst_temps) > last_n:
        sst_temps = sst_temps[-last_n:]
        fvcom_temps = fvcom_temps[-last_n:]
        adjusted_temps = adjusted_temps[-last_n:]
        test_idx_plot = test_idx[-last_n:]
    else:
        test_idx_plot = test_idx
        last_n = len(sst_temps)

    valid = ~(np.isnan(sst_temps) | np.isnan(fvcom_temps) | np.isnan(adjusted_temps))
    sst_temps = sst_temps[valid]
    fvcom_temps = fvcom_temps[valid]
    adjusted_temps = adjusted_temps[valid]

    time_steps = np.arange(1, len(sst_temps) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, sst_temps, color='g', linewidth=2.5, label='Satellite SST (Obs)')
    plt.plot(time_steps, fvcom_temps, color='b', linewidth=2.5, label='FVCOM (Raw)')
    plt.plot(time_steps, adjusted_temps, color='r', linewidth=2.5, label='CNNLSTM Adjustment')

    plt.xlabel('测试集时间步 (最后60个样本)', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.title('SST、FVCOM和CNNLSTM订正温度对比 (2010-2023年)', fontsize=15)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('temperature_comparison_CNNLSTM_summer_2010_2023.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Three-curve temperature comparison figure saved: temperature_comparison_CNNLSTM_summer_2010_2023.png (last {len(sst_temps)} samples shown)")


def plot_mean_temperature_comparison_summer(model, test_ds, lon, lat, means, stds, Y_mean, Y_std, device,
                                     output_path='mean_temperature_comparison_CNNLSTM_summer_2010_2023.png'):

    model.eval()
    loader = DataLoader(test_ds, batch_size=1)

    fvcom_all = []
    pred_all = []
    sst_all = []

    print("Computing mean temperature fields of all test samples...")

    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            x, y, m = x.to(device), y.to(device), m.to(device)

            pred = model(x).squeeze().cpu().numpy()
            pred_denorm = pred * Y_std + Y_mean

            fvcom = x[0, 5].cpu().numpy() * stds[5] + means[5]

            sst = y.squeeze().cpu().numpy() * Y_std + Y_mean
            m = m.squeeze().cpu().numpy()

            fvcom_masked = np.where(m == 1, fvcom, np.nan)
            pred_masked = np.where(m == 1, pred_denorm, np.nan)
            sst_masked = np.where(m == 1, sst, np.nan)

            fvcom_all.append(fvcom_masked)
            pred_all.append(pred_masked)
            sst_all.append(sst_masked)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} samples")

    fvcom_mean = np.nanmean(np.stack(fvcom_all), axis=0)
    pred_mean = np.nanmean(np.stack(pred_all), axis=0)
    sst_mean = np.nanmean(np.stack(sst_all), axis=0)

    diff_mean = pred_mean - fvcom_mean

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    temp_min = max(0, min(np.nanmin(fvcom_mean), np.nanmin(pred_mean), np.nanmin(sst_mean)) - 2)
    temp_max = min(30, max(np.nanmax(fvcom_mean), np.nanmax(pred_mean), np.nanmax(sst_mean)) + 2)

    diff_abs_max = max(abs(np.nanmin(diff_mean)), abs(np.nanmax(diff_mean)))
    diff_abs_max = min(diff_abs_max, 10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    titles = [
        '订正前平均温度 (°C)\n(FVCOM原始模拟)',
        '订正后平均温度 (°C)\n(CNNLSTM校正结果)',
        '温度差值 (°C)\n(订正后 - 订正前)'
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

    plt.suptitle('2020-2023年海表温度订正效果对比 - 区域平均',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Mean temperature comparison figure saved: {output_path}")

    print("\nStatistics (2020-2023):")
    print(f"Satellite observed mean temperature: {np.nanmean(sst_mean):.3f} +/- {np.nanstd(sst_mean):.3f} deg C")
    print(f"Mean temperature before correction: {np.nanmean(fvcom_mean):.3f} +/- {np.nanstd(fvcom_mean):.3f} deg C")
    print(f"Mean temperature after correction: {np.nanmean(pred_mean):.3f} +/- {np.nanstd(pred_mean):.3f} deg C")
    print(f"Mean temperature change: {np.nanmean(diff_mean):.3f} deg C")


def main_summer():
    set_seed(46)
    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data_summer(
        r'D:\CNN\夏季数据处理\output_2010_2023_summer\data_with_mask_2010_2023_summer.mat')
    train, val, test, test_idx = split_data_sequential_summer(X, Y, mask)
    train_ds = OceanDataset_summer(*train)
    val_ds = OceanDataset_summer(*val)
    test_ds = OceanDataset_summer(*test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = CNNResLSTM_summer(in_channels=10, input_height=X.shape[2], input_width=X.shape[3])

    model_path = 'best_CNNLSTM_model_summer_2010_2023.pth'

    print(f"\nLoading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("\n" + "=" * 60)
    print("Full Test Set Evaluation")
    print("=" * 60)
    eval_results = evaluate_full_testset_metrics_summer(model, test_ds, means, stds, Y_mean, Y_std, device)

    results_df = pd.DataFrame([eval_results])

    csv_path = "CNNLSTM_夏季_test_metrics_2010_2023.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\nEvaluation metrics saved to: {csv_path}")
    print(results_df)

    plot_temperature_comparison_simple_summer(model, test_ds, test_idx, means, stds, Y_mean, Y_std, device, last_n=60)

    plot_mean_temperature_comparison_summer(model, test_ds, lon, lat, means, stds, Y_mean, Y_std, device,
                                     output_path='mean_temperature_comparison_CNNLSTM_summer_2010_2023.png')


if __name__ == '__main__':
    main_summer()
