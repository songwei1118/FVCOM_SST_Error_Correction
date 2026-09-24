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
from scipy.io import loadmat

warnings.filterwarnings('ignore')

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
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


    X = data[:, [1, 2, 3, 4, 6, 7, 8, 9, 13], :, :]


    Y = data[:, 0, :, :]


    FVCOM_SST = data[:, 6, :, :]

    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]

    time_raw = data[:, 12, 0, 0]

    full_time = pd.date_range(
        start='2010-01-01',
        end='2023-12-31',
        freq='D'
    )

    time = full_time[
        ~((full_time.month == 1) & (full_time.day == 1))
    ]

    assert len(time) == data.shape[0], \
        f"时间长度({len(time)}) 与数据长度({data.shape[0]})不一致！"


    mask = ~np.isnan(Y)

    Y = np.where(mask, Y, 0)

    mask = mask.astype(np.float32)


    means_X = np.zeros(X.shape[1])
    stds_X = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        xi = X[:, i]

        means_X[i] = np.nanmean(xi)
        stds_X[i] = np.nanstd(xi)

        xi = np.where(
            np.isnan(xi),
            means_X[i],
            xi
        )

        X[:, i] = xi


    if normalize:

        X = (
                    X
                    - means_X.reshape(1, -1, 1, 1)
            ) / (
                    stds_X.reshape(1, -1, 1, 1) + 1e-8
            )


        Y_valid = Y[mask == 1]

        mean_Y = np.mean(Y_valid)
        std_Y = np.std(Y_valid)

        Y = (
                    Y - mean_Y
            ) / (
                    std_Y + 1e-8
            )

        print(
            f"OSTIA normalization parameters: "
            f"mean={mean_Y:.4f}, std={std_Y:.4f}"
        )

    else:

        mean_Y = None
        std_Y = None

    print("\nData loading completed")

    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    print(f"   FVCOM SST shape: {FVCOM_SST.shape}")

    print("\nCurrent model input variables:")


    return {
        'X': X,
        'Y': Y,
        'FVCOM_SST': FVCOM_SST,
        'mask': mask,

        'lon': lon,
        'lat': lat,
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

    print(f"Time series data split:")
    print(f"   Training set: first {len(train_idx)} time points ({train_idx[0]} - {train_idx[-1]})")
    print(f"   Validation set: middle {len(val_idx)} time points ({val_idx[0]} - {val_idx[-1]})")
    print(f"   Test set: last {len(test_idx)} time points ({test_idx[0]} - {test_idx[-1]})")

    print("\nCorresponding actual date range:")

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


class EarthSpecificPositionEncoding(nn.Module):

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
                 in_channels=9,
                 spatial_height=200,
                 spatial_width=120,
                 d_model=128,
                 nhead=8,
                 num_layers=2,
                 dim_feedforward=512,
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

        self.pos_encoding = EarthSpecificPositionEncoding(
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
              save_path='ERA5-only-MSCT-PE-2010-2023-best-model.pth'):

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


def evaluate_model(
        model,
        test_loader,
        test_idx,
        FVCOM_SST,
        device,
        mean_Y,
        std_Y):

    model.eval()


    sum_sq_err_fvcom = 0.0
    sum_sq_err_model = 0.0

    sum_abs_err_fvcom = 0.0
    sum_abs_err_model = 0.0

    n_valid = 0


    with torch.no_grad():

        for batch_id, (
                x,
                y,
                mask) in enumerate(test_loader):

            x = x.to(device)

            y = y.to(device)

            mask = mask.to(device)


            pred = model(x)


            pred_np = \
                pred.cpu().numpy()[0]

            y_np = \
                y.cpu().numpy()[0]

            mask_np = \
                mask.cpu().numpy()[0]


            pred_denorm = (
                pred_np
                * std_Y
                + mean_Y
            )

            y_denorm = (
                y_np
                * std_Y
                + mean_Y
            )


            global_idx = \
                test_idx[batch_id]

            fvcom_denorm = \
                FVCOM_SST[global_idx]


            valid_mask = (
                mask_np == 1
            )


            if not np.any(valid_mask):

                continue


            fvcom_vals = \
                fvcom_denorm[
                    valid_mask
                ]

            model_vals = \
                pred_denorm[
                    valid_mask
                ]

            obs_vals = \
                y_denorm[
                    valid_mask
                ]


            sum_sq_err_fvcom += np.sum(
                (
                    fvcom_vals
                    - obs_vals
                ) ** 2
            )


            sum_sq_err_model += np.sum(
                (
                    model_vals
                    - obs_vals
                ) ** 2
            )


            sum_abs_err_fvcom += np.sum(
                np.abs(
                    fvcom_vals
                    - obs_vals
                )
            )


            sum_abs_err_model += np.sum(
                np.abs(
                    model_vals
                    - obs_vals
                )
            )


            n_valid += \
                fvcom_vals.size


    fvcom_rmse = np.sqrt(
        sum_sq_err_fvcom
        / n_valid
    )

    model_rmse = np.sqrt(
        sum_sq_err_model
        / n_valid
    )


    fvcom_mae = (
        sum_abs_err_fvcom
        / n_valid
    )

    model_mae = (
        sum_abs_err_model
        / n_valid
    )


    rmse_improvement = (

        (
            fvcom_rmse
            - model_rmse
        )
        / fvcom_rmse
        * 100
    )


    mae_improvement = (

        (
            fvcom_mae
            - model_mae
        )
        / fvcom_mae
        * 100
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "MSCT-PE "
        "No-FVCOM-SST "
        "Test results"
    )

    print(
        "=" * 60
    )


    print(
        f"FVCOM RMSE : "
        f"{fvcom_rmse:.4f} °C"
    )

    print(
        f"Model RMSE : "
        f"{model_rmse:.4f} °C"
    )

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:.2f} %"
    )


    print(
        f"FVCOM MAE : "
        f"{fvcom_mae:.4f} °C"
    )

    print(
        f"Model MAE : "
        f"{model_mae:.4f} °C"
    )

    print(
        f"MAE improvement : "
        f"{mae_improvement:.2f} %"
    )


    print("=" * 60)


    return {

        "FVCOM_RMSE":
            fvcom_rmse,

        "MSCT_PE_No_FVCOM_SST_RMSE":
            model_rmse,

        "RMSE_Improvement_percent":
            rmse_improvement,

        "FVCOM_MAE":
            fvcom_mae,

        "MSCT_PE_No_FVCOM_SST_MAE":
            model_mae,

        "MAE_Improvement_percent":
            mae_improvement
    }


def compute_spatial_rmse_maps(
        model,
        test_dataset,
        test_idx,
        FVCOM_SST,
        mean_Y,
        std_Y,
        device):
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    sum_sq_model = None
    sum_sq_fvcom = None

    sum_err_model = None
    sum_err_fvcom = None

    count = None

    with torch.no_grad():

        for local_idx, (x, y, mask) in enumerate(loader):

            x = x.to(device)

            pred = model(x)

            pred = pred.cpu().numpy()[0]

            y = y.numpy()[0]

            mask = mask.numpy()[0]


            pred = (
                    pred * std_Y + mean_Y
            )

            obs = (
                    y * std_Y + mean_Y
            )


            global_idx = test_idx[local_idx]

            fvcom = FVCOM_SST[global_idx]

            valid = mask == 1


            if sum_sq_model is None:
                sum_sq_model = np.zeros_like(pred)

                sum_sq_fvcom = np.zeros_like(pred)

                sum_err_model = np.zeros_like(pred)

                sum_err_fvcom = np.zeros_like(pred)

                count = np.zeros_like(pred)


            diff_model = pred - obs

            diff_fvcom = fvcom - obs

            sum_sq_model[valid] += (
                    diff_model[valid] ** 2
            )

            sum_sq_fvcom[valid] += (
                    diff_fvcom[valid] ** 2
            )

            sum_err_model[valid] += (
                diff_model[valid]
            )

            sum_err_fvcom[valid] += (
                diff_fvcom[valid]
            )

            count[valid] += 1

    count_safe = count + 1e-8

    rmse_model = np.sqrt(
        sum_sq_model / count_safe
    )

    rmse_fvcom = np.sqrt(
        sum_sq_fvcom / count_safe
    )

    bias_model = (
            sum_err_model / count_safe
    )

    bias_fvcom = (
            sum_err_fvcom / count_safe
    )

    improvement = (
            rmse_fvcom - rmse_model
    )

    improvement_pct = (
            improvement
            / (rmse_fvcom + 1e-8)
            * 100
    )

    return (
        rmse_model,
        rmse_fvcom,
        improvement,
        improvement_pct,
        bias_model,
        bias_fvcom
    )


def plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model='MSCT-PE-NO-SST-RMSE_model_crop.png',
                   final_path_fvcom='MSCT-PE-NO-SST-RMSE_fvcom_crop.png'):
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

        ax.set_xticklabels([f"{x:.1f}°E" for x in xticks])
        ax.set_yticklabels([f"{y:.1f}°N" for y in yticks])

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
    draw_and_crop(rmse_model, 'MSCT-PE-NO-SST RMSE ', final_path_model)


def plot_mean_temperature_map_comparison(
        model,
        test_dataset,
        test_idx,
        FVCOM_SST,
        lon,
        lat,
        mean_Y,
        std_Y,
        device,
        save_path_sat='MSCT-PE-NO-SST-SST_satellite_mean.png',
        save_path_fvcom='MSCT-PE-NO-SST-SST_fvcom_mean.png',
        save_path_model='MSCT-PE-NO-SST-SST_model_mean.png'):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fvcom_all, pred_all, sst_all = [], [], []

    with torch.no_grad():
        for local_idx, (x, y, mask) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)

            pred_np = pred.cpu().numpy()[0]
            y_np = y.cpu().numpy()[0]
            mask_np = mask.cpu().numpy()[0]

            pred_denorm = pred_np * std_Y + mean_Y
            sst_denorm = y_np * std_Y + mean_Y

            global_idx = test_idx[local_idx]

            fvcom_denorm = FVCOM_SST[global_idx]

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

    draw_single(sst_mean, 'Satellite SST', save_path_sat)
    draw_single(fvcom_mean, 'FVCOM SST', save_path_fvcom)
    draw_single(pred_mean, 'MSCT-PE-NO-SST SST', save_path_model)


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

        ax.set_xticklabels([f"{x:.1f}°E" for x in xticks], fontsize=16)
        ax.set_yticklabels([f"{y:.1f}°N" for y in yticks], fontsize=16)

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
    draw(bias_model, "MSCT-PE-NO-SST Bias", save_path_model)


def main():

    print("=" * 60)
    print(
        "MSCT-PE "
        "No-FVCOM-SST Evaluation "
        "(2010-2023)"
    )
    print("=" * 60)

    output_dir = r'./MSCT-PE-NoFVCOMSST-results_2010_2023'

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
        data['X'], data['Y'], data['mask'], data['time']
    )

    print("\nStep 3: Creating datasets")
    train_dataset = OceanSSTDataset(X_train, Y_train, mask_train)
    val_dataset = OceanSSTDataset(X_val, Y_val, mask_val)
    test_dataset = OceanSSTDataset(X_test, Y_test, mask_test)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\nStep 4: Creating memory-efficient model")
    model = MemoryEfficientOceanSSTModel(
        in_channels=9,
        spatial_height=200,
        spatial_width=120,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        downscale_factor=2
    )
    model.to(device)

    best_model_path = r"D:\CNN\convtransformer模型\No-FVCOM-SST-MSCT-PE-results_2010_2023\No-FVCOM-SST-MSCT-PE-FVCOM-2010-2023-best-model.pth"
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Best model loaded")
    print(f"   Best epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown')}")

    print("\nStep X: Saving test set time series")

    print("\nStep 5: Evaluating model")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        test_idx=test_idx,
        FVCOM_SST=data['FVCOM_SST'],
        device=device,
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y']
    )

    results_df = pd.DataFrame([results])

    save_path = os.path.join(output_dir, "MSCT-PE-NO-SST-evaluation_metrics.csv")
    results_df.to_csv(save_path, index=False)

    print(f"RMSE and MAE results saved to: {save_path}")

    print("\nStep 6: Computing spatial RMSE distribution")
    rmse_model, rmse_fvcom, improvement, improvement_pct, \
        bias_model, bias_fvcom = compute_spatial_rmse_maps(
        model=model,
        test_dataset=test_dataset,
        test_idx=test_idx,
        FVCOM_SST=data['FVCOM_SST'],
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y'],
        device=device
    )


    np.save(os.path.join(output_dir, "MSCT-PE-NO-SST-bias_model.npy"), bias_model)
    np.save(os.path.join(output_dir, "MSCT-PE-NO-SST-bias_fvcom.npy"), bias_fvcom)

    np.save(os.path.join(output_dir, "MSCT-PE-NO-SST-rmse_model.npy"), rmse_model)
    np.save(os.path.join(output_dir, "MSCT-PE-NO-SST-rmse_fvcom.npy"), rmse_fvcom)

    np.save(os.path.join(output_dir, "MSCT-PE-NO-SST-improvement.npy"), improvement)

    print("Spatial statistics saved (for post-processing plots)")

    plot_rmse_maps(rmse_model, rmse_fvcom,
                   final_path_model=os.path.join(output_dir, 'MSCT-PE-NO-SST-model_rmse_final.png'),
                   final_path_fvcom=os.path.join(output_dir, 'MSCT-PE-NO-SST-fvcom_rmse_final.png'))

    print("\nStep 7: Plotting Bias Maps")

    plot_bias_maps(
        bias_model,
        bias_fvcom,
        save_path_model=os.path.join(output_dir, 'MSCT-PE-NO-SST-bias_model.png'),
        save_path_fvcom=os.path.join(output_dir, 'MSCT-PE-NO-SST-bias_fvcom.png')
    )

    print("\nStep 8: Plotting spatial mean temperature comparison")
    plot_mean_temperature_map_comparison(
        model=model,
        test_dataset=test_dataset,
        test_idx=test_idx,
        FVCOM_SST=data['FVCOM_SST'],
        lon=data['lon'],
        lat=data['lat'],
        mean_Y=data['mean_Y'],
        std_Y=data['std_Y'],
        device=device,

        save_path_sat=os.path.join(
            output_dir,
            'MSCT-PE-NO-SST-SST_OSTIA_mean.png'
        ),

        save_path_fvcom=os.path.join(
            output_dir,
            'MSCT-PE-NO-SST-SST_FVCOM_mean.png'
        ),

        save_path_model=os.path.join(
            output_dir,
            'MSCT-PE-NO-SST-SST_model_mean.png'
        )
    )


if __name__ == '__main__':
    main()
