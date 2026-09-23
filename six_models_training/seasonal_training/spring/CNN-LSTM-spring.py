import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader


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


class OceanDataset_spring(Dataset):
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


def load_data_spring(file_path=r'D:\CNN\春季数据处理\output_2010_2023_spring\data_with_mask_2010_2023_spring.mat'):
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


def split_data_sequential_spring(X, Y, mask, train_ratio=0.7, val_ratio=0.15):
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


class ResBlock_spring(nn.Module):
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


class CNNResLSTM_spring(nn.Module):
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
            ResBlock_spring(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_spring(128)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=False)
        self.lstm_fc = nn.Linear(lstm_hidden, 128)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            ResBlock_spring(64),
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


def train_model_spring(
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
        save_dir, 'best_CNNLSTM_model_spring_2010_2023.pth'
    )
    last_model_path = os.path.join(
        save_dir, 'last_CNNLSTM_model_spring_2010_2023.pth'
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
    plt.savefig('loss_curve_CNNLSTM_spring_2010_2023.png', dpi=300)
    plt.show()

    return train_losses, val_losses, best_model_path



def main_spring():
    set_seed(46)
    X, Y, mask, lon, lat, time, means, stds, Y_mean, Y_std = load_data_spring(
        r'D:\CNN\春季数据处理\output_2010_2023_spring\data_with_mask_2010_2023_spring.mat')
    train, val, test, test_idx = split_data_sequential_spring(X, Y, mask)
    train_ds = OceanDataset_spring(*train)
    val_ds = OceanDataset_spring(*val)
    test_ds = OceanDataset_spring(*test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = CNNResLSTM_spring(in_channels=10, input_height=X.shape[2], input_width=X.shape[3])

    train_losses, val_losses, best_model_path = train_model_spring(model, train_ds, val_ds, device, num_epochs=100, batch_size=8)

    model.load_state_dict(torch.load('best_CNNLSTM_model_spring_2010_2023.pth'))



if __name__ == '__main__':
    main_spring()
