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
from sklearn.metrics import mean_squared_error, r2_score


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


def load_data(file_path=r'D:\CNN\夏季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'):
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

    best_model_path = os.path.join(save_dir, 'best_cnn_resnet_se_model_2010_2023_秋季.pth')
    last_model_path = os.path.join(save_dir, 'last_cnn_resnet_se_model_2010_2023_秋季.pth')

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

    torch.save(model.state_dict(), last_model_path)

    print("\nTraining finished")
    print(f"Best model epoch: {best_epoch}")
    print(f"Best model path: {best_model_path}")
    print(f"Final model path: {last_model_path}")

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
    plt.savefig('loss_curve_cnn_2010_2023_秋季.png', dpi=300)
    plt.show()

    return best_model_path



def main():
    set_seed(46)

    X, Y, mask, lon, lat, time, means_X, stds_X, Y_mean, Y_std = load_data(
        r'D:\CNN\秋季数据处理\output_2010_2023_autumn\data_with_mask_2010_2023_autumn.mat'
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

    best_model_path = train_model(
        model,
        train_ds,
        val_ds,
        device,
        num_epochs=100,
        batch_size=4,
        lr=1e-3
    )


    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()


if __name__ == '__main__':
    main()
