'''
This code can be used to reproduce Fig. 6 and Supplementary Fig. 4
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from metrics_record import *

data = SIC_MAE_diff_para
figurename = 'figure/SIC_MAE_External_impact.png'

df = pd.DataFrame(data)

hisunet = df["HISUnet"]
percent_change = pd.DataFrame()
for col in df.columns:
    if col != "HISUnet":
        percent_change[col] = ((df[col] - hisunet) / hisunet * 100).round(2)

percent_change.index = [f"Sample_{i+1}" for i in range(len(percent_change))]

colors = ['#00008B', '#4169E1', '#87CEFA', '#FFFFE0', '#FFA500', '#FF6347', '#8B0000']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 18))

im = ax.imshow(percent_change.T, cmap=cmap, aspect='auto', vmin=-30, vmax=30)

ax.set_xticks(range(len(percent_change.index)))
ax.set_xticklabels(percent_change.index, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(percent_change.columns)))
ax.set_yticklabels(percent_change.columns, fontsize=9)

for i in range(len(percent_change.columns)):
    for j in range(len(percent_change.index)):
        val = percent_change.iloc[j, i]
        text_color = 'white' if abs(val) > 15 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

ax.set_title('SIC MAE Metric Change (%)', fontsize=14, pad=20)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, orientation='vertical')
cbar.set_label('-30', rotation=0, labelpad=10, y=0.02, fontsize=10)
cbar.set_label('30', rotation=0, labelpad=10, y=0.98, fontsize=10)
cbar.ax.tick_params(labelsize=0) 

plt.tight_layout()
plt.savefig(figurename, dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
