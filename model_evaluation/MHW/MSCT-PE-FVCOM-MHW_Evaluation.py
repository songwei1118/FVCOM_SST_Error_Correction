import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import pandas as pd


warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20


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
    print(f"Transposed data shape: {data.shape}")

    X = data[:, [1,2,3,4,5,6,7,8,9,13], :, :]
    Y = data[:, 0, :, :]

    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]

    full_time = pd.date_range('2010-01-01', '2023-12-31', freq='D')
    time = full_time[~((full_time.month == 1) & (full_time.day == 1))]

    assert len(time) == data.shape[0]

    mask = ~np.isnan(Y)
    Y = np.where(mask, Y, 0)
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

        X = (X - means_X.reshape(1,-1,1,1)) / (stds_X.reshape(1,-1,1,1)+1e-8)

        mean_Y = np.nanmean(Y[mask==1])
        std_Y = np.nanstd(Y[mask==1])
        Y = (Y - mean_Y) / (std_Y + 1e-8)

    else:
        means_X = stds_X = mean_Y = std_Y = None

    print("Data loading completed")

    return {
        'X': X, 'Y': Y, 'mask': mask,
        'lon': lon, 'lat': lat,
        'time': time,
        'means_X': means_X,
        'stds_X': stds_X,
        'mean_Y': mean_Y,
        'std_Y': std_Y
    }



def split_data_sequential(X, Y, mask, time, train_ratio=0.7, val_ratio=0.15):
    N = len(X)

    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, N)

    print("Time series data split:")
    print(f"   Training set: first {len(train_idx)} time points ({train_idx[0]} - {train_idx[-1]})")
    print(f"   Validation set: middle {len(val_idx)} time points ({val_idx[0]} - {val_idx[-1]})")
    print(f"   Test set: last {len(test_idx)} time points ({test_idx[0]} - {test_idx[-1]})")

    print("\nCorresponding real date ranges:")

    print("\nTraining set:")
    print("Start date:", time[train_idx[0]])
    print("End date:", time[train_idx[-1]])

    print("\nValidation set:")
    print("Start date:", time[val_idx[0]])
    print("End date:", time[val_idx[-1]])

    print("\nTest set:")
    print("Start date:", time[test_idx[0]])
    print("End date:", time[test_idx[-1]])

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


class LearnableSpatialPositionEncoding(nn.Module):

    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model

        self.pos_embed = nn.Parameter(
            torch.randn(1, d_model, height, width) * 0.02
        )

    def forward(self, x):
        return x + self.pos_embed


class MultiScaleCNNEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 2, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x3 = self.branch_3x3(x)
        x5 = self.branch_5x5(x)

        x = torch.cat([x3, x5], dim=1)

        x = self.fusion(x)
        return x


class MemoryEfficientOceanSSTModel(nn.Module):

    def __init__(self,
                 in_channels=10,
                 spatial_height=200,
                 spatial_width=120,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 downscale_factor=2):
        super().__init__()
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

        self.orig_height = spatial_height
        self.orig_width = spatial_width
        self.height = spatial_height // downscale_factor
        self.width = spatial_width // downscale_factor
        self.d_model = d_model

        self.input_conv = MultiScaleCNNEncoder(
            in_channels=in_channels,
            out_channels=128
        )

        self.scale_enhance = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(128, d_model,
                      kernel_size=downscale_factor,
                      stride=downscale_factor),
            nn.ReLU(),
            nn.BatchNorm2d(d_model)
        )

        self.pos_encoding = LearnableSpatialPositionEncoding(
            d_model=d_model,
            height=self.height,
            width=self.width
        )

        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                d_model, 64,
                kernel_size=downscale_factor,
                stride=downscale_factor),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        x_in = x
        x = self.input_conv(x)
        x = self.scale_enhance(x)
        x = self.downsample(x)

        x = self.pos_encoding(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        for layer in self.transformer:
            x = layer(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.upsample(x)
        out = self.output_conv(x)

        residual = self.residual_conv(x_in)
        out = out + self.residual_alpha * residual

        return out.squeeze(1)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
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

    def train(self, train_loader, val_loader, num_epochs=50,
              save_path='CNN-Transformer-多尺度35绝对位置编码-动态学习率2010-2023-有气压_best_model.pth'):

        print("Starting training...")
        params_info = self.model.count_parameters()
        print("Model parameters:")
        print(f"   Total parameters: {params_info['total']:,}")
        print(f"   Trainable parameters: {params_info['trainable']:,}")
        print("Using dynamic learning rate scheduler: ReduceLROnPlateau")
        print(f"Early stopping enabled: patience={self.early_stop_patience}, "
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

                print(f"Best model saved (Val Loss={val_loss:.6f})")

            else:
                self.early_stop_counter += 1
                print(f"EarlyStopping counter: "
                      f"{self.early_stop_counter}/{self.early_stop_patience}")

            if self.early_stop_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        print(f"Training finished, best validation loss: {self.best_val_loss:.6f}")
        return self.history


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

            pred_np = pred.cpu().numpy()
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
    print("SST Error Evaluation over the Entire Study Area During MHW")
    print("=" * 65)

    print(f"Number of MHW days      : {n_mhw_days}")
    print(f"Total valid ocean grids : {n_valid}")

    print("-" * 65)

    print(f"FVCOM RMSE           : {fvcom_rmse:.4f} °C")
    print(f"MSCT-PE-FVCOM RMSE   : {model_rmse:.4f} °C")
    print(f"RMSE Improvement     : {rmse_improvement:.2f} %")

    print("-" * 65)

    print(f"FVCOM MAE            : {fvcom_mae:.4f} °C")
    print(f"MSCT-PE-FVCOM MAE    : {model_mae:.4f} °C")
    print(f"MAE Improvement      : {mae_improvement:.2f} %")

    print("=" * 65)

    result_df = pd.DataFrame({
        "Dataset": ["MHW_period"],
        "MHW_days": [n_mhw_days],
        "Valid_grid_points": [n_valid],

        "FVCOM_RMSE_C": [fvcom_rmse],
        "MSCT_PE_FVCOM_RMSE_C": [model_rmse],
        "RMSE_Improvement_percent": [rmse_improvement],

        "FVCOM_MAE_C": [fvcom_mae],
        "MSCT_PE_FVCOM_MAE_C": [model_mae],
        "MAE_Improvement_percent": [mae_improvement]
    })

    save_path = (
        r"D:\CNN\convtransformer模型"
        r"\MSCT_PE_FVCOM_MHW_RMSE_MAE.csv"
    )

    result_df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nMHW RMSE/MAE results saved:")
    print(save_path)

    return result_df


def main():

    print("=" * 60)
    print("Ocean SST Prediction Model - Loading Best Model Directly (2010-2023)")
    print("=" * 60)

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
        data['X'], data['Y'], data['mask'], data['time']
    )
    time_test = data['time'][test_idx]

    print("\nStep 3: Creating datasets")
    train_dataset = OceanSSTDataset(X_train, Y_train, mask_train)
    val_dataset = OceanSSTDataset(X_val, Y_val, mask_val)
    test_dataset = OceanSSTDataset(X_test, Y_test, mask_test)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\nStep 4: Creating memory-efficient model")
    model = MemoryEfficientOceanSSTModel(
        in_channels=10,
        spatial_height=200,
        spatial_width=120,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        downscale_factor=2
    )
    model.to(device)

    best_model_path = 'CNN-Transformer-多尺度35绝对位置编码-动态学习率2010-2023-有气压_best_model.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Best model loaded")
    print(f"   Best epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown')}")

    print("\nStep X: Saving test set time series")

    mhw_file = r"D:\CNN\春季数据处理\极端事件\YellowBohai_MHW_days_2022_2023.csv"

    mhw_df = pd.read_csv(
        mhw_file,
        header=None
    )

    mhw_dates = pd.to_datetime(
        mhw_df.iloc[:, 0]
    ).dt.normalize()

    mhw_dates = set(mhw_dates)

    print("\nMHW date information")
    print(f"Total MHW dates : {len(mhw_dates)}")
    print(f"MHW start date  : {min(mhw_dates)}")
    print(f"MHW end date    : {max(mhw_dates)}")

    test_dates_set = set(
        pd.to_datetime(time_test).normalize()
    )

    matched_mhw_dates = mhw_dates.intersection(
        test_dates_set
    )

    print(f"Number of MHW dates in test set : {len(matched_mhw_dates)}")

    if len(matched_mhw_dates) == 0:
        print("No overlap between MHW dates and the test set!")
        return

    print(
        f"Test set MHW start date : "
        f"{min(matched_mhw_dates)}"
    )

    print(
        f"Test set MHW end date : "
        f"{max(matched_mhw_dates)}"
    )

    print("\nStarting RMSE / MAE calculation during MHW")

    evaluate_mhw_model(
        model=model,
        test_loader=test_loader,
        time_test=time_test,
        mhw_dates=matched_mhw_dates,
        device=device,
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y'],
        means_X=data['means_X'],
        stds_X=data['stds_X']
    )


if __name__ == '__main__':
    main()
