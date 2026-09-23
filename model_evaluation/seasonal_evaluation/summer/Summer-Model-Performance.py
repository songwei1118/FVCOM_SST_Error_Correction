import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({"font.family": "serif","font.serif": ["Times New Roman"]})
plt.rcParams['axes.unicode_minus'] = False


def read_csv_file(file_path):

    df = pd.read_csv(file_path)
    cols = [c.lower().strip() for c in df.columns]

    fvcom_rmse = df[df.columns[cols.index('fvcom_rmse')]].values[0]
    model_rmse = df[df.columns[cols.index('model_rmse')]].values[0]
    fvcom_mae  = df[df.columns[cols.index('fvcom_mae')]].values[0]
    model_mae  = df[df.columns[cols.index('model_mae')]].values[0]

    return fvcom_rmse, model_rmse, fvcom_mae, model_mae


csv_files = [
    r"D:\CNN\cnn模型\evaluation_metrics_CNN_2010_2023_夏季.csv",
    r"D:\CNN\convLSTM模型\CNNLSTM_夏季_test_metrics_2010_2023.csv",
    r"D:\CNN\convtransformer模型\夏季-多尺度CNN-Transformer-evaluation_metrics_2010_2023.csv",
    r"D:\CNN\SA_Unet模型\results_UNet_SA_year_2010_2023_summer\evaluation_results_saunet_year_2010_2023_summer.csv",
    r"D:\CNN\SE_Unet模型\results_UNet_SE_summer_2010_2023\evaluation_results_seunet_year_2010_2023_summer.csv",
    r"D:\CNN\CBAM_Unet模型\results_UNet_CBAM_year_2010_2023_夏季\evaluation_results_CBAMunet_year_2010_2023_summer.csv"
]

model_names = ["CNN-FVCOM", "CNN-LSTM-FVCOM", "MSCT-PE-FVCOM", "SA-Unet-FVCOM", "SE-Unet-FVCOM", "CBAM-Unet-FVCOM"]


fvcom_rmse_list = []
model_rmse_list = []
fvcom_mae_list = []
model_mae_list = []

for file in csv_files:
    fvcom_rmse, model_rmse, fvcom_mae, model_mae = read_csv_file(file)
    fvcom_rmse_list.append(fvcom_rmse)
    model_rmse_list.append(model_rmse)
    fvcom_mae_list.append(fvcom_mae)
    model_mae_list.append(model_mae)


fvcom_rmse = fvcom_rmse_list[0]
fvcom_mae  = fvcom_mae_list[0]

rmse_values = [fvcom_rmse] + model_rmse_list
mae_values  = [fvcom_mae]  + model_mae_list
labels = ["FVCOM"] + model_names


colors = [
    "#F26C6C",  # FVCOM
    "#7DA0C4",  # CNN
    "#C7C79E",  # CNN-LSTM
    "#A8D0A6",  # Transformer
    "#B4A7D6",  # SA-Unet
    "#9EC1CF",  # SE-Unet
    "#D8C3A5"   # CBAM-Unet
]


sorted_data = sorted(
    zip(rmse_values, mae_values, labels, colors),
    key=lambda x: x[0],
    reverse=True
)

rmse_values, mae_values, labels, colors = zip(*sorted_data)

rmse_values = list(rmse_values)
mae_values = list(mae_values)
labels = list(labels)
colors = list(colors)


gap = 1.3
x = np.arange(len(labels)) * gap

fig, ax1 = plt.subplots(figsize=(15,7))


bars = ax1.bar(
    x,
    rmse_values,
    width=0.55,
    color=colors,
    edgecolor='black',
    linewidth=1
)
ax1.set_ylabel("RMSE (°C)", fontsize=22)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=24)
ax1.set_title("Summer", fontsize=28, fontweight='bold')
ax1.tick_params(axis='y', labelsize=22)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')


ax2 = ax1.twinx()
ax2.plot(
    x,
    mae_values,
    linestyle='-',
    linewidth=2,
    marker='o',
    markersize=6,
    color="#444444"
)
ax2.set_ylabel("MAE (°C)", fontsize=22)
ax2.tick_params(axis='y', labelsize=22)


ymin = min(min(rmse_values), min(mae_values)) * 0.95
ymax = max(max(rmse_values), max(mae_values)) * 1.05
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)



ax1.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()


plt.savefig("Summer_RMSE_MAE_Comparison.png", dpi=300, bbox_inches='tight')


plt.savefig("Summer_RMSE_MAE_Comparison.pdf", dpi=300, bbox_inches='tight')


plt.show()



