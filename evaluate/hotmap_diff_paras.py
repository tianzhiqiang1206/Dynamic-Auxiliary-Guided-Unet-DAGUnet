'''
This code can be used to reproduce the supplementary Fig.4
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from metrics_record import *

# --- 1. 数据提取与重构 ---

# 识别所有模型的名称
# model_names = ['HISUnet', 'MCPN_sit', 'MCPN_ist11', 'MCPN_ist12', 'MCPN_ist13', 
#                'MCPN_ist14', 'MCPN_u10', 'MCPN_v10', 'MCPN_u100', 'MCPN_v100']

# 提取图片中的数据（按列读取，每列对应一个模型）
# 数据的排列顺序是 7天 * 5年 = 35行
para_name = ['SIV_V_MAE']
outpath = rf'F:/DAGUnet_code/figures/{para_name[0]}_Hotmap.png'
raw_data = SIV_V_MAE_diff_para
# 创建 DataFrame
df = pd.DataFrame(raw_data)

# 添加 Year 和 Day_Index 列
years = [2007] * 7 + [2012] * 7 + [2019] * 7 + [2020] * 7 + [2023] * 7
day_indices = [1, 2, 3, 4, 5, 6, 7] * 5

df['Year'] = years
df['Day_Index'] = day_indices

# 创建一个组合键 (Year_Day) 作为新的索引
df['Year_Day'] = df['Year'].astype(str) + '_Day' + df['Day_Index'].astype(str)
df = df.set_index('Year_Day')

# 删除原始的 Year 和 Day_Index 列
df = df.drop(columns=['Year', 'Day_Index'])

# 将 HISUnet 列作为基准
hisunet_baseline = df['HISUnet']

# 计算百分比变化（基准模型 HISUnet 除外）
df_percentage_change = df.copy()

for col in df.columns:
    if col != 'HISUnet':
        # 计算百分比变化
        df_percentage_change[col] = ((df[col] - hisunet_baseline) / hisunet_baseline) * 100

# 移除 HISUnet 列，因为它的变化总是 0
df_heatmap = df_percentage_change.drop(columns=['HISUnet'])

# 将 Day_Index 作为行索引，模型作为列索引
df_heatmap_pivot = df_heatmap.reset_index()
# 将 Year_Day 拆分为 Year 和 Day
df_heatmap_pivot[['Year', 'Day']] = df_heatmap_pivot['Year_Day'].str.split('_Day', expand=True)

# --- 3. 绘制热力图 ---

# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 如果 SimHei 不可用，可以尝试 'Microsoft YaHei' 或其他中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

plt.figure(figsize=(7, 8))

# 确定颜色映射的中心点（0% 变化）和范围
# vmin 和 vmax 应该对称，以确保 0 在颜色条中心
v_max = df_heatmap.abs().max().max()
v_min = -v_max

# 绘制热力图
sns.heatmap(
    df_heatmap,
    cmap='coolwarm',  # 'coolwarm' 颜色映射，适合显示正负变化
    annot=True,       # 显示数值
    fmt=".2f",        # 保留一位小数
    linewidths=.5,    # 网格线
    cbar_kws={'shrink': 0.8, 'aspect': 20, 'fraction': 0.03},
    # cbar_kws={'label': '相对于 HISUnet 的指标变化 (%)'},
    vmin=v_min,
    vmax=v_max
)

plt.title(rf'{para_name[0]} Metric Change (%)', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(outpath,dpi = 1200)
plt.show()

# ----------------------------------------------------
# 简要解读（根据您提供的数据）
# ----------------------------------------------------
print("\n--- 热力图解读要点 ---")
print("颜色越红 (正值)，表示该模型在该 Year/Day Index 下的指标值比 HISUnet 高。")
print("颜色越蓝 (负值)，表示该模型在该 Year/Day Index 下的指标值比 HISUnet 低。")
print("如果该指标是误差 (如 MAE/RMSE)，蓝色（负变化）表示模型性能更好。")
print("如果该指标是相关性 (如 CORR)，蓝色（负变化）表示模型性能更差。")