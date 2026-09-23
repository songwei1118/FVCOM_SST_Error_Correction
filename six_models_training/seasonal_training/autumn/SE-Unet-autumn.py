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


def load_data(file_path=r'D:\CNN\秋季数据处理\output_2010_2023_autumn\data_with_mask_2010_2023_autumn.mat'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            raise KeyError("HDF5 文件中未找到 'data' 数据集")
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
    print(f"Mean of Y: {Y_mean:.4f}, Std of Y: {Y_std:.4f}")

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


def train_model(model, train_dataset, val_dataset, device, num_epochs=100, batch_size=16, lr=5e-4,
                save_dir="./results_UNet_SE_autumn_2010_2023", patience=10):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_resunet_model_seunet_秋季.pth")
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
    plt.title("Training & Validation Loss (2010-2023 Autumn)")
    plt.savefig(os.path.join(save_dir, "loss_curve_seunet_autumn_2010_2023.png"), dpi=150)
    plt.close()

    print(f"Best model saved: {best_model_path}")
    print(f"Total epochs trained: {epoch}, Best validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, best_model_path



def main():
    set_seed(46)
    DATA_PATH = r'D:\CNN\秋季数据处理\output_2010_2023_autumn\data_with_mask_2010_2023_autumn.mat'
    SAVE_DIR = "./results_UNet_SE_autumn_2010_2023"

    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data(DATA_PATH)
    train, val, test, test_idx = split_data(X, Y, mask)
    train_ds, val_ds, test_ds = OceanDataset(*train), OceanDataset(*val), OceanDataset(*test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(in_channels=10, out_channels=1)

    train_losses, val_losses, best_model_path = train_model(model, train_ds, val_ds, device, save_dir=SAVE_DIR)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)


if __name__ == "__main__":
    main()
