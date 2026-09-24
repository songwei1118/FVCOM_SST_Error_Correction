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
    print(f"Random seed set: {seed}")


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


def load_data(file_path=r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'):
    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]
    print(f"Raw data shape: {data.shape}")
    data = np.transpose(data, (3, 2, 1, 0))
    print(f"Data shape: {data.shape}")

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :]
    Y = data[:, 0, :, :]
    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]
    time = data[:, 12, 0, 0]

    full_time = pd.date_range(
        start='2010-01-01',
        end='2023-12-31',
        freq='D'
    )

    time = full_time[~((full_time.month == 1) & (full_time.day == 1))]

    assert len(time) == data.shape[0], \
        f"时间长度({len(time)}) 与数据长度({data.shape[0]})不一致！"

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


def split_data_sequential(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
    N = len(X)

    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    print("Sequential data split:")
    print(f"   Training: first {len(train_idx)} time points (index: {train_idx[0]} - {train_idx[-1]})")
    print(f"   Validation: middle {len(val_idx)} time points (index: {val_idx[0]} - {val_idx[-1]})")
    print(f"   Testing: last {len(test_idx)} time points (index: {test_idx[0]} - {test_idx[-1]})")

    X_train, Y_train, mask_train = X[train_idx], Y[train_idx], mask[train_idx]
    X_val, Y_val, mask_val = X[val_idx], Y[val_idx], mask[val_idx]
    X_test, Y_test, mask_test = X[test_idx], Y[test_idx], mask[test_idx]

    return (X_train, Y_train, mask_train), \
        (X_val, Y_val, mask_val), \
        (X_test, Y_test, mask_test), \
        test_idx


class ResBlock(nn.Module):
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


class CNNResLSTM(nn.Module):
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
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock(128)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=False)
        self.lstm_fc = nn.Linear(lstm_hidden, 128)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock(64),
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


def train_model(
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
        save_dir, 'best_CNNLSTM_model_2010_2023.pth'
    )
    last_model_path = os.path.join(
        save_dir, 'last_CNNLSTM_model_2010_2023.pth'
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
            print(f"Validation loss did not improve significantly for {early_stop_patience} consecutive epochs")
            print(f"Best epoch: {best_epoch}, best Val Loss: {best_val_loss:.6f}")
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
    plt.savefig('loss_curve_CNNLSTM_2010_2023.png', dpi=300)
    plt.show()

    return train_losses, val_losses, best_model_path


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
    print("SST error evaluation over the full study region during MHW period")
    print("=" * 65)

    print(f"Number of MHW days            : {n_mhw_days}")
    print(f"Total valid ocean grid points : {n_valid}")

    print("-" * 65)

    print(f"FVCOM RMSE           : {fvcom_rmse:.4f} °C")
    print(f"CNN-LSTM-FVCOM RMSE   : {model_rmse:.4f} °C")
    print(f"RMSE Improvement     : {rmse_improvement:.2f} %")

    print("-" * 65)

    print(f"FVCOM MAE            : {fvcom_mae:.4f} °C")
    print(f"CNN-LSTM-FVCOM MAE    : {model_mae:.4f} °C")
    print(f"MAE Improvement      : {mae_improvement:.2f} %")

    print("=" * 65)

    result_df = pd.DataFrame({
        "Dataset": ["MHW_period"],
        "MHW_days": [n_mhw_days],
        "Valid_grid_points": [n_valid],

        "FVCOM_RMSE_C": [fvcom_rmse],
        "CNN-LSTM_FVCOM_RMSE_C": [model_rmse],
        "RMSE_Improvement_percent": [rmse_improvement],

        "FVCOM_MAE_C": [fvcom_mae],
        "CNN-LSTM_FVCOM_MAE_C": [model_mae],
        "MAE_Improvement_percent": [mae_improvement]
    })

    save_path = (
        r"D:\CNN\convLSTM模型"
        r"\CNN-LSTM_FVCOM_MHW_RMSE_MAE.csv"
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
    output_dir = r'./CNN-LSTM-results_2010_2023'

    os.makedirs(output_dir, exist_ok=True)

    print(f"All results will be saved to: {output_dir}")

    set_seed(46)
    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data(
        r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat')

    print("\nSplitting dataset")

    train, val, test, test_idx = split_data_sequential(
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Loaded best model")
    model = CNNResLSTM(in_channels=10, input_height=X.shape[2], input_width=X.shape[3])

    model.load_state_dict(torch.load('best_CNNLSTM_model_2010_2023.pth', map_location=device))

    model.to(device)

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
    print(f"Total MHW days : {len(mhw_dates)}")
    print(f"MHW start date : {min(mhw_dates)}")
    print(f"MHW end date   : {max(mhw_dates)}")

    test_dates_set = set(
        pd.to_datetime(time_test).normalize()
    )

    matched_mhw_dates = (
        mhw_dates.intersection(test_dates_set)
    )

    print(
        f"Number of MHW days in test set : "
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
        "\nCalculating RMSE / MAE during MHW period"
    )

    result = evaluate_mhw_model(
        model=model,
        test_loader=test_loader,
        time_test=time_test,
        mhw_dates=matched_mhw_dates,
        device=device,
        mean_Y=Y_mean,
        std_Y=Y_std,
        means_X=means,
        stds_X=stds
    )

    print("\nMHW evaluation completed")

    return result


if __name__ == '__main__':
    main()
