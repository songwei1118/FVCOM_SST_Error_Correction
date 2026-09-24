import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({"font.family": "serif","font.serif": ["Times New Roman"]})
plt.rcParams['axes.unicode_minus'] = False


model_names = [
    "CNN-FVCOM",
    "CNN-LSTM-FVCOM",
    "SE-Unet-FVCOM",
    "SA-Unet-FVCOM",
    "CBAM-UNet-FVCOM",
    "MSCT-PE-FVCOM"
]

bias_fvcom = np.load(r"D:\CNN\convtransformer模型\MSCT-PE-FVCOM-longest-MHW-2023\longest_MHW_20230511_20230827_bias_fvcom.npy")
rmse_fvcom = np.load(r"D:\CNN\convtransformer模型\MSCT-PE-FVCOM-longest-MHW-2023\longest_MHW_20230511_20230827_rmse_fvcom.npy")

bias_models = [
    np.load(r"D:\CNN\cnn模型\CNNResNetSE-longest-MHW-2023\CNN-longest_MHW_20230511_20230827_bias_model.npy"),
    np.load(r"D:\CNN\convLSTM模型\CNN-LSTM-longest-MHW-2023\CNN-LSTM-longest_MHW_20230511_20230827_bias_model.npy"),
    np.load(r"D:\CNN\SE_Unet模型\SE-Unet-longest-MHW-2023\SE-Unet-longest_MHW_20230511_20230827_bias_model.npy"),
    np.load(r"D:\CNN\SA_Unet模型\SA-Unet-longest-MHW-2023\SA-Unet-longest_MHW_20230511_20230827_bias_model.npy"),
    np.load(r"D:\CNN\CBAM_Unet模型\CBAM-Unet-longest-MHW-2023\CBAM-Unet-longest_MHW_20230511_20230827_bias_model.npy"),
    np.load(r"D:\CNN\convtransformer模型\MSCT-PE-FVCOM-longest-MHW-2023\longest_MHW_20230511_20230827_bias_model.npy")
]

rmse_models = [
    np.load(r"D:\CNN\cnn模型\CNNResNetSE-longest-MHW-2023\CNN-longest_MHW_20230511_20230827_rmse_model.npy"),
    np.load(r"D:\CNN\convLSTM模型\CNN-LSTM-longest-MHW-2023\CNN-LSTM-longest_MHW_20230511_20230827_rmse_model.npy"),
    np.load(r"D:\CNN\SE_Unet模型\SE-Unet-longest-MHW-2023\SE-Unet-longest_MHW_20230511_20230827_rmse_model.npy"),
    np.load(r"D:\CNN\SA_Unet模型\SA-Unet-longest-MHW-2023\SA-Unet-longest_MHW_20230511_20230827_rmse_model.npy"),
    np.load(r"D:\CNN\CBAM_Unet模型\CBAM-Unet-longest-MHW-2023\CBAM-Unet-longest_MHW_20230511_20230827_rmse_model.npy"),
    np.load(r"D:\CNN\convtransformer模型\MSCT-PE-FVCOM-longest-MHW-2023\longest_MHW_20230511_20230827_rmse_model.npy")
]

def plot_bias_all(bias_fvcom, bias_models, model_names, save_path):

    all_data = [bias_fvcom] + bias_models
    titles = ["FVCOM"] + model_names

    vmax = np.ceil(max(np.nanmax(np.abs(d)) for d in all_data))
    vmin = -vmax

    levels = np.arange(vmin, vmax + 0.4, 0.4)

    norm = BoundaryNorm(
        levels,
        ncolors=256,
        clip=True
    )

    lon = np.linspace(117, 127, bias_fvcom.shape[0])
    lat = np.linspace(35.5, 41.5, bias_fvcom.shape[1])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(
        2, 4,
        figsize=(20, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    fig.subplots_adjust(
        wspace=0.22,
        hspace=0.26
    )

    axes = axes.flatten()

    plot_positions = [0,1,2,3,5,6,7]

    mappable = None

    for idx, (data, title) in enumerate(zip(all_data, titles)):

        ax = axes[plot_positions[idx]]

        im = ax.pcolormesh(
            lon_grid, lat_grid, data.T,
            cmap='RdBu_r',
            norm=norm,
            shading='nearest',
            transform=ccrs.PlateCarree(),
            zorder=1,
            edgecolors='none',
            linewidth=0
        )

        ax.set_extent([117, 127, 35.5, 41.5])

        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       facecolor='white',
                       edgecolor='black',
                       linewidth=0.3,
                       zorder=10
                       )

        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=0.6,
                       edgecolor = 'black',
                       zorder = 20
        )

        xticks = np.arange(117, 127.1, 2.5)
        yticks = np.arange(35.5, 41.6, 2)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

        ax.minorticks_off()

        ax.set_xticklabels(
            [f"{x:g}°E" for x in xticks]
        )

        if idx in [0, 4]:
            ax.set_yticklabels(
                [f"{y:g}°N" for y in yticks]
            )
        else:
            ax.set_yticklabels([])

        if idx in [0, 4]:
            ax.tick_params(
                axis='y',
                left=True,
                labelleft=True
            )
        else:
            ax.tick_params(
                axis='y',
                left=False,
                labelleft=False
            )

        ax.tick_params(
            axis='both',
            which='major',
            direction='out',
            length=4,
            width=1.0,
            labelsize=16,
            top=False,
            right=False,
            bottom=True
        )
        ax.set_title(title, fontsize=18, fontweight='bold')

        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=0.8,
            edgecolor='black',
            zorder=1000
        )

        ax.add_patch(rect)

    fig.delaxes(axes[4])

    left = axes[0].get_position().x0
    right = axes[3].get_position().x1
    bottom = axes[5].get_position().y0 - 0.08

    cax = fig.add_axes([left, bottom, right-left, 0.025])

    mappable = ScalarMappable(
        norm=norm,
        cmap='RdBu_r'
    )

    mappable.set_array([])

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation='horizontal',
        extend='both',
        extendfrac=0.05,
        boundaries=levels,
        ticks=levels,
        spacing='proportional'
    )

    cbar.set_label('Bias (°C)', fontsize=18)

    cbar.ax.xaxis.set_major_formatter(
        FormatStrFormatter('%.1f')
    )

    cbar.ax.tick_params(labelsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rmse_all(rmse_fvcom, rmse_models, model_names, save_path):

    all_data = [rmse_fvcom] + rmse_models
    titles = ["FVCOM"] + model_names

    vmin = 0

    vmax = np.ceil(np.nanmax(rmse_fvcom) / 0.2) * 0.2

    levels = np.arange(vmin, vmax + 0.1, 0.1)
    print("FVCOM min/max:", np.nanmin(rmse_fvcom), np.nanmax(rmse_fvcom))

    norm = BoundaryNorm(
        levels,
        ncolors=256,
        clip=True
    )

    lon = np.linspace(117, 127, rmse_fvcom.shape[0])
    lat = np.linspace(35.5, 41.5, rmse_fvcom.shape[1])
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(
        2, 4,
        figsize=(20, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    fig.subplots_adjust(
        wspace=0.22,
        hspace=0.26
    )
    axes = axes.flatten()

    plot_positions = [0,1,2,3,5,6,7]

    for idx, (data, title) in enumerate(zip(all_data, titles)):

        ax = axes[plot_positions[idx]]

        im = ax.pcolormesh(
            lon_grid, lat_grid, data.T,
            cmap='turbo',
            norm=norm,
            shading='nearest',
            transform=ccrs.PlateCarree(),
            zorder=1,
            edgecolors='none',
            linewidth=0
        )

        ax.set_extent([117, 127, 35.5, 41.5])

        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       facecolor='white',
                       edgecolor='black',
                       linewidth=0.3,
                       zorder=10
                       )

        ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                       linewidth=0.6,
                       edgecolor='black',
                       zorder=20
                       )

        xticks = np.arange(117, 127.1, 2.5)
        yticks = np.arange(35.5, 41.6, 2)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

        ax.minorticks_off()

        ax.set_xticklabels(
            [f"{x:g}°E" for x in xticks]
        )

        if idx in [0, 4]:
            ax.set_yticklabels(
                [f"{y:g}°N" for y in yticks]
            )
        else:
            ax.set_yticklabels([])

        if idx in [0, 4]:
            ax.tick_params(
                axis='y',
                left=True,
                labelleft=True
            )
        else:
            ax.tick_params(
                axis='y',
                left=False,
                labelleft=False
            )

        ax.tick_params(
            axis='both',
            which='major',
            direction='out',
            length=4,
            width=1.0,
            labelsize=16,
            top=False,
            right=False,
            bottom=True
        )
        ax.set_title(title, fontsize=18, fontweight='bold')

        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=0.8,
            edgecolor='black',
            zorder=1000
        )

        ax.add_patch(rect)

    fig.delaxes(axes[4])

    left = axes[0].get_position().x0
    right = axes[3].get_position().x1
    bottom = axes[5].get_position().y0 - 0.08

    cax = fig.add_axes([left, bottom, right-left, 0.025])

    mappable = ScalarMappable(
        norm=norm,
        cmap='turbo'
    )

    mappable.set_array([])

    ticks = np.arange(vmin, vmax + 0.4, 0.4)

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation='horizontal',
        extend='max',
        extendfrac=0.05,
        boundaries=levels,
        ticks=ticks,
        spacing='proportional'
    )

    cbar.set_label('RMSE (°C)', fontsize=18)

    cbar.ax.xaxis.set_major_formatter(
        FormatStrFormatter('%.1f')
    )

    cbar.ax.tick_params(labelsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"RMSE plot completed: {save_path}")


plot_bias_all(bias_fvcom, bias_models, model_names, "longestMHW-Bias_all.png")
plot_rmse_all(rmse_fvcom, rmse_models, model_names, "longestMHW-RMSE_all.png")
