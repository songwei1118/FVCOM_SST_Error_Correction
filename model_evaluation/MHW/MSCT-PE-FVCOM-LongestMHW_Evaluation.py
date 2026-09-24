import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import warnings
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from PIL import Image
from matplotlib.colors import BoundaryNorm


warnings.filterwarnings('ignore')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"]
})

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
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU model: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
        )

    return device


def load_ocean_data(file_path, normalize=True):
    print(f"Loading data: {file_path}")

    with h5py.File(file_path, 'r') as f:
        data = f['data'][:]

    print(f"Original data shape: {data.shape}")

    data = np.transpose(
        data,
        (3, 2, 1, 0)
    )

    print(
        f"Transposed data shape: "
        f"{data.shape}"
    )

    X = data[
        :,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 13],
        :,
        :
    ]

    Y = data[:, 0, :, :]

    lon = data[0, 10, :, 0]
    lat = data[0, 11, 0, :]

    time_raw = data[:, 12, 0, 0]

    full_time = pd.date_range(
        start='2010-01-01',
        end='2023-12-31',
        freq='D'
    )

    time = full_time[
        ~(
            (full_time.month == 1) &
            (full_time.day == 1)
        )
    ]

    assert len(time) == data.shape[0], (
        f"Time length ({len(time)}) does not match "
        f"data length ({data.shape[0]})!"
    )

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

        X = (
            X -
            means_X.reshape(1, -1, 1, 1)
        ) / (
            stds_X.reshape(1, -1, 1, 1)
            + 1e-8
        )

        Y_valid = Y[mask == 1]

        mean_Y = np.nanmean(Y_valid)
        std_Y = np.nanstd(Y_valid)

        Y = (
            Y - mean_Y
        ) / (
            std_Y + 1e-8
        )

        print(
            f"Target normalization parameters: "
            f"mean={mean_Y:.4f}, "
            f"std={std_Y:.4f}"
        )

    else:

        means_X = None
        stds_X = None
        mean_Y = None
        std_Y = None

    print("\nData loading completed:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Mask shape: {mask.shape}")

    print(
        f"Longitude range: "
        f"{lon.min():.2f} - {lon.max():.2f}"
    )

    print(
        f"Latitude range: "
        f"{lat.min():.2f} - {lat.max():.2f}"
    )

    print(f"Number of days: {len(time)}")

    print(
        f"Time range: "
        f"{time[0]} -> {time[-1]}"
    )

    print("\n========== Training normalization parameters ==========")

    print("X means:")
    for i, v in enumerate(means_X):
        print(f"Channel {i}: {v:.10f}")

    print("\nX stds:")
    for i, v in enumerate(stds_X):
        print(f"Channel {i}: {v:.10f}")

    print(f"\nY mean: {mean_Y:.10f}")
    print(f"Y std : {std_Y:.10f}")

    return {
        'X': X,
        'Y': Y,
        'mask': mask,
        'lon': lon,
        'lat': lat,
        'time': time,
        'means_X': means_X,
        'stds_X': stds_X,
        'mean_Y': mean_Y,
        'std_Y': std_Y
    }


def split_data_sequential(
        X,
        Y,
        mask,
        time,
        train_ratio=0.7,
        val_ratio=0.15):

    N = len(X)

    train_end = int(
        N * train_ratio
    )

    val_end = (
        train_end +
        int(N * val_ratio)
    )

    train_idx = np.arange(
        0,
        train_end
    )

    val_idx = np.arange(
        train_end,
        val_end
    )

    test_idx = np.arange(
        val_end,
        N
    )

    print("\nSequential data split:")

    print(
        f"Training samples: "
        f"{len(train_idx)} "
        f"({train_idx[0]} - {train_idx[-1]})"
    )

    print(
        f"Validation samples: "
        f"{len(val_idx)} "
        f"({val_idx[0]} - {val_idx[-1]})"
    )

    print(
        f"Testing samples: "
        f"{len(test_idx)} "
        f"({test_idx[0]} - {test_idx[-1]})"
    )

    print("\nActual date ranges:")

    print(
        f"Training: "
        f"{time[train_idx[0]]} -> "
        f"{time[train_idx[-1]]}"
    )

    print(
        f"Validation: "
        f"{time[val_idx[0]]} -> "
        f"{time[val_idx[-1]]}"
    )

    print(
        f"Testing: "
        f"{time[test_idx[0]]} -> "
        f"{time[test_idx[-1]]}"
    )

    return (
        X[train_idx],
        Y[train_idx],
        mask[train_idx]
    ), (
        X[val_idx],
        Y[val_idx],
        mask[val_idx]
    ), (
        X[test_idx],
        Y[test_idx],
        mask[test_idx]
    ), test_idx


class OceanSSTDataset(Dataset):

    def __init__(
            self,
            X,
            Y,
            mask):

        self.X = X
        self.Y = Y
        self.mask = mask

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            torch.tensor(
                self.X[idx],
                dtype=torch.float32
            ),

            torch.tensor(
                self.Y[idx],
                dtype=torch.float32
            ),

            torch.tensor(
                self.mask[idx],
                dtype=torch.float32
            )
        )


class EarthSpecificPositionEncoding(nn.Module):

    def __init__(
            self,
            d_model,
            height,
            width):

        super().__init__()

        self.height = height
        self.width = width
        self.d_model = d_model

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                d_model,
                height,
                width
            ) * 0.02
        )

    def forward(self, x):

        return x + self.pos_embed


class MultiScaleCNNEncoder(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels):

        super().__init__()

        self.branch_3x3 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64)
        )

        self.branch_5x5 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64)
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                64 * 2,
                out_channels,
                kernel_size=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(
                out_channels
            )
        )

    def forward(self, x):

        x3 = self.branch_3x3(x)

        x5 = self.branch_5x5(x)

        x = torch.cat(
            [x3, x5],
            dim=1
        )

        x = self.fusion(x)

        return x


class MemoryEfficientOceanSSTModel(nn.Module):

    def __init__(
            self,
            in_channels=10,
            spatial_height=200,
            spatial_width=120,
            d_model=128,
            nhead=8,
            num_layers=2,
            dim_feedforward=512,
            downscale_factor=2):

        super().__init__()

        self.residual_alpha = nn.Parameter(
            torch.tensor(0.1)
        )

        self.orig_height = spatial_height
        self.orig_width = spatial_width

        self.height = (
            spatial_height //
            downscale_factor
        )

        self.width = (
            spatial_width //
            downscale_factor
        )

        self.d_model = d_model

        self.input_conv = MultiScaleCNNEncoder(
            in_channels=in_channels,
            out_channels=128
        )

        self.scale_enhance = nn.Sequential(

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(128)
        )

        self.downsample = nn.Sequential(

            nn.Conv2d(
                128,
                d_model,
                kernel_size=downscale_factor,
                stride=downscale_factor
            ),

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
                d_model,
                64,
                kernel_size=downscale_factor,
                stride=downscale_factor
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64)
        )

        self.output_conv = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
            )
        )

        self.residual_conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        x_in = x

        x = self.input_conv(x)

        x = self.scale_enhance(x)

        x = self.downsample(x)

        x = self.pos_encoding(x)

        B, C, H, W = x.shape

        x = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        for layer in self.transformer:

            x = layer(x)

        x = x.transpose(
            1,
            2
        ).reshape(
            B,
            C,
            H,
            W
        )

        x = self.upsample(x)

        out = self.output_conv(x)

        residual = self.residual_conv(x_in)

        out = (
            out +
            self.residual_alpha * residual
        )

        return out.squeeze(1)

    def count_parameters(self):

        total_params = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total": total_params,
            "trainable": trainable_params
        }


class MaskedHuberSmoothLoss(nn.Module):

    def __init__(
            self,
            delta=1.0,
            smooth_weight=0.05):

        super().__init__()

        self.delta = delta
        self.smooth_weight = smooth_weight

    def huber(self, diff):

        abs_diff = torch.abs(diff)

        quadratic = torch.minimum(
            abs_diff,
            torch.tensor(
                self.delta,
                device=diff.device
            )
        )

        linear = (
            abs_diff -
            quadratic
        )

        return (
            0.5 * quadratic ** 2 +
            self.delta * linear
        )

    def spatial_smoothness(
            self,
            pred,
            mask):

        mask_x = (
            mask[:, :, 1:] *
            mask[:, :, :-1]
        )

        mask_y = (
            mask[:, 1:, :] *
            mask[:, :-1, :]
        )

        dx = (
            torch.abs(
                pred[:, :, 1:] -
                pred[:, :, :-1]
            ) *
            mask_x
        )

        dy = (
            torch.abs(
                pred[:, 1:, :] -
                pred[:, :-1, :]
            ) *
            mask_y
        )

        return (
            dx.sum() +
            dy.sum()
        ) / (
            mask_x.sum() +
            mask_y.sum() +
            1e-8
        )

    def forward(
            self,
            pred,
            target,
            mask):

        diff = pred - target

        huber_loss = self.huber(diff)

        masked_huber = (
            huber_loss * mask
        ).sum() / (
            mask.sum() + 1e-8
        )

        smooth_loss = self.spatial_smoothness(
            pred,
            mask
        )

        return (
            masked_huber +
            self.smooth_weight *
            smooth_loss
        )


def evaluate_model(
        model,
        test_loader,
        device,
        mean_Y,
        std_Y,
        means_X,
        stds_X):

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

            pred = model(x)

            pred_np = (
                pred.detach()
                .cpu()
                .numpy()
            )

            y_np = (
                y.detach()
                .cpu()
                .numpy()
            )

            x_np = (
                x.detach()
                .cpu()
                .numpy()
            )

            mask_np = (
                mask.detach()
                .cpu()
                .numpy()
            )

            pred_denorm = (
                pred_np * std_Y +
                mean_Y
            )

            y_denorm = (
                y_np * std_Y +
                mean_Y
            )

            fvcom_denorm = (
                x_np[
                    :,
                    fvcom_channel_in_X,
                    :,
                    :
                ] *
                stds_X[fvcom_channel_in_X] +
                means_X[fvcom_channel_in_X]
            )

            valid_mask = (
                mask_np == 1
            )

            if not np.any(valid_mask):

                continue

            fvcom_vals = (
                fvcom_denorm[
                    valid_mask
                ]
            )

            model_vals = (
                pred_denorm[
                    valid_mask
                ]
            )

            obs_vals = (
                y_denorm[
                    valid_mask
                ]
            )

            sum_sq_err_fvcom += np.sum(
                (fvcom_vals - obs_vals) ** 2
            )

            sum_sq_err_model += np.sum(
                (model_vals - obs_vals) ** 2
            )

            sum_abs_err_fvcom += np.sum(
                np.abs(
                    fvcom_vals - obs_vals
                )
            )

            sum_abs_err_model += np.sum(
                np.abs(
                    model_vals - obs_vals
                )
            )

            n_valid += fvcom_vals.size

    fvcom_rmse = np.sqrt(
        sum_sq_err_fvcom /
        n_valid
    )

    model_rmse = np.sqrt(
        sum_sq_err_model /
        n_valid
    )

    fvcom_mae = (
        sum_abs_err_fvcom /
        n_valid
    )

    model_mae = (
        sum_abs_err_model /
        n_valid
    )

    rmse_improvement = (
        (fvcom_rmse - model_rmse) /
        fvcom_rmse *
        100
    )

    mae_improvement = (
        (fvcom_mae - model_mae) /
        fvcom_mae *
        100
    )

    print("\nFull test set evaluation:")

    print(
        f"FVCOM RMSE: "
        f"{fvcom_rmse:.4f} °C"
    )

    print(
        f"Model RMSE: "
        f"{model_rmse:.4f} °C"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"FVCOM MAE: "
        f"{fvcom_mae:.4f} °C"
    )

    print(
        f"Model MAE: "
        f"{model_mae:.4f} °C"
    )

    print(
        f"MAE improvement: "
        f"{mae_improvement:.2f}%"
    )

    return {
        "fvcom_rmse": fvcom_rmse,
        "model_rmse": model_rmse,
        "rmse_improvement": rmse_improvement,
        "fvcom_mae": fvcom_mae,
        "model_mae": model_mae,
        "mae_improvement": mae_improvement
    }


def compute_spatial_rmse_maps(
        model,
        test_dataset,
        mean_Y,
        std_Y,
        means_X,
        stds_X,
        device,
        mhw_mask=None,
        fvcom_channel=5):

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

    selected_days = 0

    with torch.no_grad():

        for day_idx, (
                x,
                y,
                mask
        ) in enumerate(loader):

            if mhw_mask is not None:

                if not mhw_mask[day_idx]:

                    continue

            selected_days += 1

            x = x.to(device)

            pred = model(x)

            pred = (
                pred
                .cpu()
                .numpy()[0]
            )

            y = (
                y.numpy()[0]
            )

            mask = (
                mask.numpy()[0]
            )

            x_np = (
                x.cpu()
                .numpy()[0]
            )

            pred = (
                pred * std_Y +
                mean_Y
            )

            obs = (
                y * std_Y +
                mean_Y
            )

            fvcom = (
                x_np[fvcom_channel] *
                stds_X[fvcom_channel] +
                means_X[fvcom_channel]
            )

            valid = (
                mask == 1
            )

            if sum_sq_model is None:

                sum_sq_model = np.zeros_like(
                    pred,
                    dtype=np.float64
                )

                sum_sq_fvcom = np.zeros_like(
                    pred,
                    dtype=np.float64
                )

                sum_err_model = np.zeros_like(
                    pred,
                    dtype=np.float64
                )

                sum_err_fvcom = np.zeros_like(
                    pred,
                    dtype=np.float64
                )

                count = np.zeros_like(
                    pred,
                    dtype=np.float64
                )

            diff_model = (
                pred - obs
            )

            diff_fvcom = (
                fvcom - obs
            )

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

    print(
        f"\nNumber of selected MHW days: "
        f"{selected_days}"
    )

    if selected_days == 0:

        raise ValueError(
            "No selected MHW days were found "
            "in the test set."
        )

    count_safe = (
        count + 1e-8
    )

    rmse_model = np.sqrt(
        sum_sq_model /
        count_safe
    )

    rmse_fvcom = np.sqrt(
        sum_sq_fvcom /
        count_safe
    )

    bias_model = (
        sum_err_model /
        count_safe
    )

    bias_fvcom = (
        sum_err_fvcom /
        count_safe
    )

    valid_grid = count > 0

    model_bias_valid = bias_model[valid_grid]
    fvcom_bias_valid = bias_fvcom[valid_grid]

    print("\n" + "=" * 60)
    print("Spatial Bias statistics during the selected MHW")
    print("=" * 60)

    print("\nFVCOM Bias:")
    print(
        f"  Minimum Bias : {np.nanmin(fvcom_bias_valid):.4f} °C"
    )
    print(
        f"  Maximum Bias : {np.nanmax(fvcom_bias_valid):.4f} °C"
    )
    print(
        f"  Mean Bias    : {np.nanmean(fvcom_bias_valid):.4f} °C"
    )
    print(
        f"  Absolute Bias Mean : "
        f"{np.nanmean(np.abs(fvcom_bias_valid)):.4f} °C"
    )

    print("\nMSCT-PE-FVCOM Bias:")
    print(
        f"  Minimum Bias : {np.nanmin(model_bias_valid):.4f} °C"
    )
    print(
        f"  Maximum Bias : {np.nanmax(model_bias_valid):.4f} °C"
    )
    print(
        f"  Mean Bias    : {np.nanmean(model_bias_valid):.4f} °C"
    )
    print(
        f"  Absolute Bias Mean : "
        f"{np.nanmean(np.abs(model_bias_valid)):.4f} °C"
    )

    print("\nBias range:")
    print(
        f"  FVCOM           : "
        f"{np.nanmin(fvcom_bias_valid):.4f} "
        f"to "
        f"{np.nanmax(fvcom_bias_valid):.4f} °C"
    )

    print(
        f"  MSCT-PE-FVCOM   : "
        f"{np.nanmin(model_bias_valid):.4f} "
        f"to "
        f"{np.nanmax(model_bias_valid):.4f} °C"
    )

    print("=" * 60)

    improvement = (
        rmse_fvcom -
        rmse_model
    )

    improvement_pct = (
        improvement /
        (rmse_fvcom + 1e-8) *
        100
    )

    return (
        rmse_model,
        rmse_fvcom,
        improvement,
        improvement_pct,
        bias_model,
        bias_fvcom
    )


def plot_rmse_maps(
        rmse_model,
        rmse_fvcom,
        final_path_model,
        final_path_fvcom):

    global_vmin = 0

    global_vmax = max(
        np.nanmax(rmse_model),
        np.nanmax(rmse_fvcom)
    )

    levels = np.linspace(
        global_vmin,
        global_vmax,
        15
    )

    norm = BoundaryNorm(
        levels,
        ncolors=256,
        clip=True
    )

    lon = np.linspace(
        117,
        127,
        rmse_model.shape[0]
    )

    lat = np.linspace(
        35.5,
        41.5,
        rmse_model.shape[1]
    )

    lon_grid, lat_grid = np.meshgrid(
        lon,
        lat
    )

    def draw_and_crop(
            data,
            title,
            final_path):

        fig = plt.figure(
            figsize=(7, 5)
        )

        ax = fig.add_axes(
            [
                0.08,
                0.15,
                0.75,
                0.75
            ],
            projection=ccrs.PlateCarree()
        )

        im = ax.pcolormesh(
            lon_grid,
            lat_grid,
            data.T,
            cmap='turbo',
            norm=norm,
            shading='nearest',
            transform=ccrs.PlateCarree(),
            zorder=1
        )

        ax.set_extent(
            [
                117,
                127,
                35.5,
                41.5
            ],
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(
            cfeature.LAND.with_scale('10m'),
            facecolor='white',
            edgecolor='black',
            linewidth=0.3,
            zorder=3
        )

        ax.add_feature(
            cfeature.COASTLINE.with_scale('10m'),
            linewidth=0.6,
            edgecolor='black',
            zorder=4
        )

        xticks = np.arange(
            117,
            127.1,
            2
        )

        yticks = np.arange(
            35.5,
            41.6,
            1
        )

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels(
            [
                f"{x:g}°E"
                for x in xticks
            ],
            fontsize=16
        )

        ax.set_yticklabels(
            [
                f"{y:g}°N"
                for y in yticks
            ],
            fontsize=16
        )

        ax.set_title(
            title,
            fontsize=18,
            fontweight='bold',
            pad=10
        )

        pos = ax.get_position()

        cax = fig.add_axes(
            [
                pos.x1 + 0.015,
                pos.y0,
                0.02,
                pos.height
            ]
        )

        cbar = fig.colorbar(
            im,
            cax=cax,
            boundaries=levels,
            ticks=levels[::2],
            spacing='proportional'
        )

        cbar.set_label(
            'RMSE (°C)',
            fontsize=16
        )

        cbar.ax.tick_params(
            labelsize=13
        )

        buf = BytesIO()

        plt.savefig(
            buf,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.005
        )

        plt.close(fig)

        buf.seek(0)

        img = Image.open(
            buf
        ).convert("RGB")

        img_data = np.array(img)

        crop_mask = np.any(
            img_data < 250,
            axis=2
        )

        coords = np.argwhere(
            crop_mask
        )

        y0, x0 = coords.min(
            axis=0
        )

        y1, x1 = coords.max(
            axis=0
        ) + 1

        margin = 60

        y0 = max(
            y0 - margin,
            0
        )

        x0 = max(
            x0 - margin,
            0
        )

        y1 = min(
            y1 + margin,
            img.height
        )

        x1 = min(
            x1 + margin,
            img.width
        )

        cropped = img.crop(
            (
                x0,
                y0,
                x1,
                y1
            )
        )

        cropped.save(
            final_path
        )

        print(
            f"Saved: {final_path}"
        )

    draw_and_crop(
        rmse_fvcom,
        'FVCOM RMSE',
        final_path_fvcom
    )

    draw_and_crop(
        rmse_model,
        'MSCT-PE-FVCOM RMSE',
        final_path_model
    )


def plot_single_bias_map(
        bias,
        title,
        save_path,
        lon,
        lat,
        vmin=-6.0,
        vmax=6.0):

    levels = np.arange(
        vmin,
        vmax + 0.5,
        0.5
    )

    norm = BoundaryNorm(
        levels,
        ncolors=256,
        clip=True
    )

    lon_grid, lat_grid = np.meshgrid(
        lon,
        lat
    )

    fig = plt.figure(
        figsize=(8.0, 6.5)
    )

    ax = fig.add_axes(
        [
            0.10,
            0.18,
            0.78,
            0.68
        ],
        projection=ccrs.PlateCarree()
    )

    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        bias.T,
        cmap='RdBu_r',
        norm=norm,
        shading='nearest',
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    ax.set_extent(
        [
            117,
            127,
            35.5,
            41.5
        ],
        crs=ccrs.PlateCarree()
    )

    ax.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='white',
        edgecolor='black',
        linewidth=0.6,
        zorder=3
    )

    ax.add_feature(
        cfeature.COASTLINE.with_scale('10m'),
        linewidth=0.8,
        edgecolor='black',
        zorder=4
    )

    xticks = np.arange(
        117,
        127.1,
        2.5
    )

    yticks = np.arange(
        35.5,
        41.6,
        2
    )

    ax.set_xticks(
        xticks
    )

    ax.set_yticks(
        yticks
    )

    ax.set_xticklabels(
        [
            f"{x:g}°E"
            for x in xticks
        ],
        fontsize=18
    )

    ax.set_yticklabels(
        [
            f"{y:g}°N"
            for y in yticks
        ],
        fontsize=18
    )

    ax.set_title(
        title,
        fontsize=20,
        fontweight='bold',
        pad=12
    )

    border_rect = patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor='black',
        linewidth=0.8,
        zorder=1000
    )

    ax.add_patch(
        border_rect
    )

    pos = ax.get_position()

    cax = fig.add_axes(
        [
            pos.x0,
            0.075,
            pos.width,
            0.025
        ]
    )

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation='horizontal',
        boundaries=levels,
        ticks=levels,
        extend='both'
    )

    cbar.set_label(
        'SST Bias (°C)',
        fontsize=18
    )

    cbar.ax.tick_params(
        labelsize=14
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.05
    )

    plt.close(fig)

    print(
        f"Saved bias map: {save_path}"
    )


def plot_bias_maps(
        bias_model,
        bias_fvcom,
        save_dir,
        lon,
        lat,
        vmin=-6.0,
        vmax=6.0):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    fvcom_path = os.path.join(
        save_dir,
        'longest_MHW_20230511_20230827_FVCOM_Bias.png'
    )

    plot_single_bias_map(
        bias=bias_fvcom,
        title='FVCOM Bias',
        save_path=fvcom_path,
        lon=lon,
        lat=lat,
        vmin=vmin,
        vmax=vmax
    )

    model_path = os.path.join(
        save_dir,
        'longest_MHW_20230511_20230827_MSCT-PE-FVCOM_Bias.png'
    )

    plot_single_bias_map(
        bias=bias_model,
        title='MSCT-PE-FVCOM Bias',
        save_path=model_path,
        lon=lon,
        lat=lat,
        vmin=vmin,
        vmax=vmax
    )

    print("\nIndependent bias figures generated:")
    print(f"FVCOM: {fvcom_path}")
    print(f"MSCT-PE-FVCOM: {model_path}")

    return (
        fvcom_path,
        model_path
    )


def main():

    print("=" * 70)

    print(
        "MSCT-PE-FVCOM Evaluation "
        "During the Longest MHW Event"
    )

    print("=" * 70)

    output_dir = (
        './MSCT-PE-FVCOM-longest-MHW-2023'
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    set_seed(46)

    device = get_device()

    print(
        "\nStep 1: Loading ocean data"
    )

    data = load_ocean_data(
        r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat',
        normalize=True
    )

    print(
        "\nStep 2: Splitting data"
    )

    (
        X_train,
        Y_train,
        mask_train
    ), (
        X_val,
        Y_val,
        mask_val
    ), (
        X_test,
        Y_test,
        mask_test
    ), test_idx = split_data_sequential(
        data['X'],
        data['Y'],
        data['mask'],
        data['time']
    )

    print(
        "\nStep 3: Identifying the longest MHW event"
    )

    longest_mhw_start = pd.Timestamp(
        '2023-05-11'
    )

    longest_mhw_end = pd.Timestamp(
        '2023-08-27'
    )

    test_time = data['time'][test_idx]

    mhw_mask = (
        (test_time >= longest_mhw_start) &
        (test_time <= longest_mhw_end)
    )

    selected_dates = test_time[
        mhw_mask
    ]

    print(
        f"Longest MHW start: "
        f"{longest_mhw_start.date()}"
    )

    print(
        f"Longest MHW end: "
        f"{longest_mhw_end.date()}"
    )

    print(
        f"Number of selected MHW days: "
        f"{mhw_mask.sum()}"
    )

    if mhw_mask.sum() != 109:

        raise ValueError(
            f"Expected 109 MHW days, "
            f"but found {mhw_mask.sum()} days "
            f"in the test dataset."
        )

    print(
        f"Actual selected date range: "
        f"{selected_dates[0]} -> "
        f"{selected_dates[-1]}"
    )

    print(
        "\nStep 4: Creating datasets"
    )

    train_dataset = OceanSSTDataset(
        X_train,
        Y_train,
        mask_train
    )

    val_dataset = OceanSSTDataset(
        X_val,
        Y_val,
        mask_val
    )

    test_dataset = OceanSSTDataset(
        X_test,
        Y_test,
        mask_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(
        "\nStep 5: Creating MSCT-PE-FVCOM model"
    )

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

    params_info = (
        model.count_parameters()
    )

    print(
        f"Total parameters: "
        f"{params_info['total']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{params_info['trainable']:,}"
    )

    print(
        "\nStep 6: Loading the best model"
    )

    best_model_path = (
        'CNN-Transformer-多尺度35绝对位置编码-动态学习率2010-2023-有气压_best_model.pth'
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            'model_state_dict'
        ]
    )

    print(
        "Best model loaded successfully."
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('val_loss', 'Unknown')}"
    )

    print(
        "\nStep 7: Evaluating the full test set"
    )

    results = evaluate_model(

        model=model,

        test_loader=test_loader,

        device=device,

        mean_Y=data['mean_Y'],

        std_Y=data['std_Y'],

        means_X=data['means_X'],

        stds_X=data['stds_X']
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_path = os.path.join(
        output_dir,
        'full_test_set_evaluation_metrics.csv'
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"Full test set metrics saved: "
        f"{results_path}"
    )

    print(
        "\nStep 8: Calculating spatial statistics "
        "during the longest MHW"
    )

    (
        rmse_model,
        rmse_fvcom,
        improvement,
        improvement_pct,
        bias_model,
        bias_fvcom
    ) = compute_spatial_rmse_maps(

        model=model,

        test_dataset=test_dataset,

        mean_Y=data['mean_Y'],

        std_Y=data['std_Y'],

        means_X=data['means_X'],

        stds_X=data['stds_X'],

        device=device,

        mhw_mask=mhw_mask,

        fvcom_channel=5
    )

    print(
        "\nStep 9: Saving spatial statistics"
    )

    np.save(
        os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_bias_model.npy'
        ),
        bias_model
    )

    np.save(
        os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_bias_fvcom.npy'
        ),
        bias_fvcom
    )

    np.save(
        os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_rmse_model.npy'
        ),
        rmse_model
    )

    np.save(
        os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_rmse_fvcom.npy'
        ),
        rmse_fvcom
    )

    np.save(
        os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_improvement.npy'
        ),
        improvement
    )

    print(
        "Spatial statistics saved successfully."
    )

    print(
        "\nStep 10: Plotting spatial RMSE maps"
    )

    plot_rmse_maps(

        rmse_model=rmse_model,

        rmse_fvcom=rmse_fvcom,

        final_path_model=os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_MSCT-PE-FVCOM_RMSE.png'
        ),

        final_path_fvcom=os.path.join(
            output_dir,
            'longest_MHW_20230511_20230827_FVCOM_RMSE.png'
        )
    )

    print(
        "\nStep 11: Plotting spatial SST Bias maps"
    )

    fvcom_bias_path, model_bias_path = plot_bias_maps(

        bias_model=bias_model,

        bias_fvcom=bias_fvcom,

        save_dir=output_dir,

        lon=data['lon'],

        lat=data['lat'],

        vmin=-6.0,

        vmax=6.0
    )

    print("\n" + "=" * 70)

    print(
        "Longest MHW spatial Bias analysis completed."
    )

    print("=" * 70)

    print(
        f"MHW period: "
        f"{longest_mhw_start.date()} "
        f"-> "
        f"{longest_mhw_end.date()}"
    )

    print(
        f"MHW duration: "
        f"{mhw_mask.sum()} days"
    )

    print(
        "\nBias definition:"
    )

    print(
        "FVCOM Bias = FVCOM SST - Satellite SST"
    )

    print(
        "MSCT-PE-FVCOM Bias = "
        "MSCT-PE-FVCOM SST - Satellite SST"
    )

    print(
        "\nSpatial domain:"
    )

    print(
        "117–127°E, 35.5–41.5°N"
    )

    print(
        "\nBias figures:"
    )

    print(
        f"FVCOM Bias: "
        f"{fvcom_bias_path}"
    )

    print(
        f"MSCT-PE-FVCOM Bias: "
        f"{model_bias_path}"
    )

    print(
        "\nAll analysis completed successfully."
    )


if __name__ == '__main__':

    main()
