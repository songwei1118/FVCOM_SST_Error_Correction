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

warnings.filterwarnings("ignore")


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
    "font.size": 16
})


def set_seed(seed=46):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


class OceanDataset(Dataset):

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


def load_data(
        file_path=
        r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'
):

    print(f"\nLoading data: {file_path}")

    with h5py.File(file_path, 'r') as f:

        data = f['data'][:]

    print(
        f"Original data shape: "
        f"{data.shape}"
    )

    data = np.transpose(
        data,
        (3, 2, 1, 0)
    )

    print(
        f"Converted data shape: "
        f"{data.shape}"
    )

    X = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 13], :, :].astype(np.float32)
    Y = data[:, 0, :, :].astype(np.float32)
    lon = data[0, 10, :, 0].astype(np.float32)
    lat = data[0, 11, 0, :].astype(np.float32)
    time = data[:, 12, 0, 0].astype(np.float32)

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
        f"Time length {len(time)} "
        f"does not match data length "
        f"{data.shape[0]}"
    )

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

    print("\n" + "=" * 60)
    print("Sequential data split")
    print("=" * 60)

    print(
        f"Training samples: "
        f"{len(train_idx)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_idx)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_idx)}"
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
        (
            X[train_idx],
            Y[train_idx],
            mask[train_idx]
        ),
        (
            X[val_idx],
            Y[val_idx],
            mask[val_idx]
        ),
        (
            X[test_idx],
            Y[test_idx],
            mask[test_idx]
        ),
        test_idx
    )


class SEBlock(nn.Module):

    def __init__(
            self,
            channels,
            reduction=16):

        super().__init__()

        self.fc = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels // reduction,
                1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels // reduction,
                channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.fc(x)

        return x * w


class ResidualBlock(nn.Module):

    def __init__(
            self,
            channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            )
        )

        self.se = SEBlock(
            channels
        )

    def forward(self, x):

        out = self.block(x)

        out = self.se(out)

        return nn.ReLU(
            inplace=True
        )(out + x)


class CNNResNetSE(nn.Module):

    def __init__(
            self,
            in_channels=10):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.layer1 = self._make_layer(
            64,
            2
        )

        self.layer2 = self._make_layer(
            64,
            2
        )

        self.layer3 = self._make_layer(
            64,
            2
        )

        self.out = nn.Conv2d(
            64,
            1,
            1
        )

    def _make_layer(
            self,
            channels,
            num_blocks):

        return nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        return self.out(x)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params
        }


def evaluate_model(
        model,
        test_loader,
        device,
        Y_mean,
        Y_std,
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

            pred = pred.squeeze(1)

            pred_np = (
                pred.cpu()
                .numpy()
            )

            y_np = (
                y.cpu()
                .numpy()
            )

            x_np = (
                x.cpu()
                .numpy()
            )

            mask_np = (
                mask.cpu()
                .numpy()
            )

            pred_denorm = (
                pred_np *
                Y_std +
                Y_mean
            )

            y_denorm = (
                y_np *
                Y_std +
                Y_mean
            )

            fvcom_denorm = (
                x_np[
                    :,
                    fvcom_channel_in_X,
                    :,
                    :
                ] *
                stds_X[
                    fvcom_channel_in_X
                ] +
                means_X[
                    fvcom_channel_in_X
                ]
            )

            valid_mask = (
                mask_np == 1
            )

            if not np.any(
                valid_mask
            ):

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
                (
                    fvcom_vals -
                    obs_vals
                ) ** 2
            )

            sum_sq_err_model += np.sum(
                (
                    model_vals -
                    obs_vals
                ) ** 2
            )

            sum_abs_err_fvcom += np.sum(
                np.abs(
                    fvcom_vals -
                    obs_vals
                )
            )

            sum_abs_err_model += np.sum(
                np.abs(
                    model_vals -
                    obs_vals
                )
            )

            n_valid += (
                fvcom_vals.size
            )

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
        (
            fvcom_rmse -
            model_rmse
        ) /
        fvcom_rmse *
        100
    )

    mae_improvement = (
        (
            fvcom_mae -
            model_mae
        ) /
        fvcom_mae *
        100
    )

    print("\n" + "=" * 70)
    print("FULL TEST SET EVALUATION")
    print("=" * 70)

    print(
        f"FVCOM RMSE       : "
        f"{fvcom_rmse:.4f} °C"
    )

    print(
        f"CNN RMSE : "
        f"{model_rmse:.4f} °C"
    )

    print(
        f"RMSE improvement : "
        f"{rmse_improvement:.2f}%"
    )

    print()

    print(
        f"FVCOM MAE        : "
        f"{fvcom_mae:.4f} °C"
    )

    print(
        f"CNN MAE  : "
        f"{model_mae:.4f} °C"
    )

    print(
        f"MAE improvement  : "
        f"{mae_improvement:.2f}%"
    )

    print("=" * 70)

    return {

        "fvcom_rmse":
            fvcom_rmse,

        "model_rmse":
            model_rmse,

        "rmse_improvement":
            rmse_improvement,

        "fvcom_mae":
            fvcom_mae,

        "model_mae":
            model_mae,

        "mae_improvement":
            mae_improvement
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
                .numpy()[0,0]
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

    print("\nCNN Bias:")
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
        f"  CNN-FVCOM   : "
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
        'CNN-FVCOM RMSE',
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
        'longest_MHW_20230511_20230827_CNN-FVCOM_Bias.png'
    )

    plot_single_bias_map(
        bias=bias_model,
        title='CNN-FVCOM Bias',
        save_path=model_path,
        lon=lon,
        lat=lat,
        vmin=vmin,
        vmax=vmax
    )

    print("\nIndependent bias figures generated:")
    print(f"FVCOM: {fvcom_path}")
    print(f"CNN-FVCOM: {model_path}")

    return (
        fvcom_path,
        model_path
    )


def main():

    print("=" * 70)

    print(
        "CNNResNetSE Evaluation "
        "During the Longest MHW Event"
    )

    print("=" * 70)

    output_dir = (
        './CNNResNetSE-longest-MHW-2023'
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    set_seed(46)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        f"Using device: {device}"
    )

    print("\nStep 1: Loading data")

    (
        X,
        Y,
        mask,
        lon,
        lat,
        time,
        means_X,
        stds_X,
        Y_mean,
        Y_std
    ) = load_data(
        r'D:\CNN\春季数据处理\output_2010_2023_full_year_strict\data_with_mask_2010_2023_full_year.mat'
    )
    print("Data loaded successfully.")

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
        X,
        Y,
        mask,
        time
    )

    train_dataset = OceanDataset(
        X_train,
        Y_train,
        mask_train
    )

    val_dataset = OceanDataset(
        X_val,
        Y_val,
        mask_val
    )

    test_dataset = OceanDataset(
        X_test,
        Y_test,
        mask_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Train dataset: {len(train_dataset)}"
    )

    print(
        f"Validation dataset: {len(val_dataset)}"
    )

    print(
        f"Test dataset: {len(test_dataset)}"
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

    test_time = time[test_idx]

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
        "\nStep 3: Creating CNNResNetSE"
    )

    model = CNNResNetSE(
        in_channels=10
    ).to(device)

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
        "\nStep 4: Loading best model"
    )

    best_model_path = (
        r'D:\CNN\cnn模型\best_cnn_resnet_se_model_2010_2023.pth'
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if 'model_state_dict' in checkpoint:

            model.load_state_dict(
                checkpoint['model_state_dict']
            )

        elif 'state_dict' in checkpoint:

            model.load_state_dict(
                checkpoint['state_dict']
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    print(
        "Best model weights loaded successfully."
    )

    if isinstance(checkpoint, dict):
        print(
            f"Best epoch: "
            f"{checkpoint.get('epoch', 'Unknown')}"
        )

        print(
            f"Validation loss: "
            f"{checkpoint.get('val_loss', 'Unknown')}"
        )

    print(
        "\nStep 5: Full test set evaluation"
    )

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        Y_mean=Y_mean,
        Y_std=Y_std,
        means_X=means_X,
        stds_X=stds_X
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_path = os.path.join(
        output_dir,
        'CNN_full_test_set_evaluation_metrics.csv'
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

        mean_Y=Y_mean,
        std_Y=Y_std,
        means_X=means_X,
        stds_X=stds_X,

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
            'CNN-longest_MHW_20230511_20230827_bias_model.npy'
        ),
        bias_model
    )

    np.save(
        os.path.join(
            output_dir,
            'CNN-longest_MHW_20230511_20230827_bias_fvcom.npy'
        ),
        bias_fvcom
    )

    np.save(
        os.path.join(
            output_dir,
            'CNN-longest_MHW_20230511_20230827_rmse_model.npy'
        ),
        rmse_model
    )

    np.save(
        os.path.join(
            output_dir,
            'CNN-longest_MHW_20230511_20230827_rmse_fvcom.npy'
        ),
        rmse_fvcom
    )

    np.save(
        os.path.join(
            output_dir,
            'CNN-longest_MHW_20230511_20230827_improvement.npy'
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
            'longest_MHW_20230511_20230827_CNN-FVCOM_RMSE.png'
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

        lon=lon,
        lat=lat,

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
        "CNN-FVCOM Bias = "
        "CNN-FVCOM SST - Satellite SST"
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
        f"CNN-FVCOM Bias: "
        f"{model_bias_path}"
    )

    print(
        "\nAll analysis completed successfully."
    )


if __name__ == '__main__':
    main()
