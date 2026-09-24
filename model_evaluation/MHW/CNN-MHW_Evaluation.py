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
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32),
                torch.tensor(self.mask[idx], dtype=torch.float32))


def load_data(file_path='D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    print(f"Original data shape: {data.shape}")

    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    var_names = ['sat_SST', 'U10', 'V10', 'short_wave', 'long_wave', 'air_pressure',  'FVCOM_temp',
                 'FVCOM_km', 'FVCOM_u', 'FVCOM_v', 'lon', 'lat', 'time', 'air_temperature']

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :].astype(np.float32)
    Y = data[:, 0, :, :].astype(np.float32)
    lon = data[0, 10, :, 0].astype(np.float32)
    lat = data[0, 11, 0, :].astype(np.float32)
    time = data[:, 12, 0, 0].astype(np.float32)

    full_time = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
    time = full_time[~((full_time.month == 1) & (full_time.day == 1))]

    assert len(time) == data.shape[0], \
        f"Time length ({len(time)}) does not match data length ({data.shape[0]})!"

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_epoch = -1
    no_improve_epochs = 0

    best_model_path = os.path.join(save_dir, 'best_cnn_resnet_se_model_2010_2023.pth')

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
            print(f"Validation loss has not improved significantly for "
                  f"{early_stop_patience} consecutive epochs")
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
    plt.title('Training / Validation Loss Curve (Early Stopping)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_cnn_2010_2023.png', dpi=300)
    plt.show()

    return best_model_path


def evaluate_mhw_model(
        model,
        test_loader,
        time_test,
        mhw_dates,
        device,
        mean_Y,
        std_Y,
        means_X,
        stds_X):

    model.eval()

    fvcom_channel = 5

    sum_sq_err_fvcom = 0.0
    sum_sq_err_model = 0.0
    sum_abs_err_fvcom = 0.0
    sum_abs_err_model = 0.0

    n_valid = 0

    n_mhw_days = 0

    with torch.no_grad():

        for i, (x, y, mask) in enumerate(test_loader):

            current_date = pd.Timestamp(
                time_test[i]
            ).normalize()

            if current_date not in mhw_dates:
                continue

            n_mhw_days += 1

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)

            pred_np = pred.squeeze(1).cpu().numpy()
            y_np = y.cpu().numpy()
            x_np = x.cpu().numpy()
            mask_np = mask.cpu().numpy()

            pred_denorm = pred_np * std_Y + mean_Y
            y_denorm = y_np * std_Y + mean_Y

            fvcom_denorm = (
                x_np[:, fvcom_channel, :, :] *
                stds_X[fvcom_channel] +
                means_X[fvcom_channel]
            )

            valid_mask = mask_np == 1

            if not np.any(valid_mask):
                continue

            fvcom_vals = fvcom_denorm[valid_mask]
            model_vals = pred_denorm[valid_mask]
            obs_vals = y_denorm[valid_mask]

            sum_sq_err_fvcom += np.sum(
                (fvcom_vals - obs_vals) ** 2
            )

            sum_sq_err_model += np.sum(
                (model_vals - obs_vals) ** 2
            )

            sum_abs_err_fvcom += np.sum(
                np.abs(fvcom_vals - obs_vals)
            )

            sum_abs_err_model += np.sum(
                np.abs(model_vals - obs_vals)
            )

            n_valid += fvcom_vals.size

    if n_mhw_days == 0 or n_valid == 0:
        print("\nNo MHW dates matched in the test set!")
        return None

    fvcom_rmse = np.sqrt(
        sum_sq_err_fvcom / n_valid
    )

    model_rmse = np.sqrt(
        sum_sq_err_model / n_valid
    )

    fvcom_mae = (
        sum_abs_err_fvcom / n_valid
    )

    model_mae = (
        sum_abs_err_model / n_valid
    )

    rmse_improvement = (
        (fvcom_rmse - model_rmse)
        / fvcom_rmse * 100
    )

    mae_improvement = (
        (fvcom_mae - model_mae)
        / fvcom_mae * 100
    )

    print("\n" + "=" * 65)
    print("SST Error Evaluation Over the Full Study Domain During MHW")
    print("=" * 65)

    print(f"Number of MHW days            : {n_mhw_days}")
    print(f"Total valid ocean grid points : {n_valid}")

    print("-" * 65)

    print(f"FVCOM RMSE           : {fvcom_rmse:.4f} °C")
    print(f"CNN-FVCOM RMSE   : {model_rmse:.4f} °C")
    print(f"RMSE Improvement     : {rmse_improvement:.2f} %")

    print("-" * 65)

    print(f"FVCOM MAE            : {fvcom_mae:.4f} °C")
    print(f"CNN-FVCOM MAE    : {model_mae:.4f} °C")
    print(f"MAE Improvement      : {mae_improvement:.2f} %")

    print("=" * 65)

    result_df = pd.DataFrame({
        "Dataset": ["MHW_period"],
        "MHW_days": [n_mhw_days],
        "Valid_grid_points": [n_valid],

        "FVCOM_RMSE_C": [fvcom_rmse],
        "CNN_FVCOM_RMSE_C": [model_rmse],
        "RMSE_Improvement_percent": [rmse_improvement],

        "FVCOM_MAE_C": [fvcom_mae],
        "CNN_FVCOM_MAE_C": [model_mae],
        "MAE_Improvement_percent": [mae_improvement]
    })

    save_path = (
        r"D:\CNN\cnn模型"
        r"\CNN_FVCOM_MHW_RMSE_MAE.csv"
    )

    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nMHW RMSE/MAE saved:")
    print(save_path)

    return result_df


def main():

    output_dir = r'./CNN-FVCOM-results_2010_2023'
    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")

    set_seed(46)

    print("\nLoading data")

    X, Y, mask, lon, lat, time, \
        means_X, stds_X, Y_mean, Y_std = load_data(
            r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'
        )

    print("\nSplitting dataset")

    train, val, test, test_idx = split_data(
        X, Y, mask
    )

    X_train, Y_train, mask_train = train
    X_val, Y_val, mask_val = val
    X_test, Y_test, mask_test = test

    time_test = time[test_idx]

    print(
        f"Test set dates: "
        f"{time_test[0]} -> {time_test[-1]}"
    )

    train_ds = OceanDataset(
        X_train, Y_train, mask_train
    )

    val_ds = OceanDataset(
        X_val, Y_val, mask_val
    )

    test_ds = OceanDataset(
        X_test, Y_test, mask_test
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"Using device: {device}")

    model = CNNResNetSE(
        in_channels=10
    ).to(device)

    print("\nLoading the best model from validation")

    best_model_path = (
        r'D:\CNN\cnn模型'
        r'\best_cnn_resnet_se_model_2010_2023.pth'
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model.eval()

    print("Best model loaded")

    mhw_file = (
        r"D:\CNN\春季数据处理\极端事件"
        r"\YellowBohai_MHW_days_2022_2023.csv"
    )

    mhw_df = pd.read_csv(
        mhw_file,
        header=None
    )

    mhw_dates = set(
        pd.to_datetime(
            mhw_df.iloc[:, 0]
        ).dt.normalize()
    )

    print("\nMHW date information")
    print(f"Total MHW dates : {len(mhw_dates)}")
    print(f"MHW start date  : {min(mhw_dates)}")
    print(f"MHW end date    : {max(mhw_dates)}")

    test_dates_set = set(
        pd.to_datetime(time_test).normalize()
    )

    matched_mhw_dates = (
        mhw_dates.intersection(test_dates_set)
    )

    print(
        f"Number of MHW dates in test set : "
        f"{len(matched_mhw_dates)}"
    )

    if len(matched_mhw_dates) == 0:
        print("No overlap between MHW dates and the test set!")
        return

    print(
        f"Test set MHW start date : "
        f"{min(matched_mhw_dates)}"
    )

    print(
        f"Test set MHW end date   : "
        f"{max(matched_mhw_dates)}"
    )

    print(
        "\nComputing RMSE / MAE during MHW"
    )

    result = evaluate_mhw_model(
        model=model,
        test_loader=test_loader,
        time_test=time_test,
        mhw_dates=matched_mhw_dates,
        device=device,
        mean_Y=Y_mean,
        std_Y=Y_std,
        means_X=means_X,
        stds_X=stds_X
    )

    print("\nMHW evaluation completed")

    return result


if __name__ == '__main__':
    main()
