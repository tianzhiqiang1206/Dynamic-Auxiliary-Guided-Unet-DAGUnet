# '''
# 计算不同区域的指标，生成表格,
# '''
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import netCDF4 as nc
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from scipy.stats import pearsonr
# import pandas as pd
# import os
# from read_data import crop_center

# def denormalize(data, min_val, max_val):
#     """将归一化后的数据反归一化回原始范围"""
#     return data * (max_val - min_val) + min_val

# def calculate_metrics(pred, true, mask=None):
#     """
#     计算相关系数(R)、均方根误差(RMSE)和平均绝对误差(MAE)
#     mask: 区域掩码，只计算掩码中为True(或非0)区域的数据
#           - 输入mask为二维(256,256)，函数内部自动扩展为和pred/true匹配的三维
#     """
#     if mask is not None:
#         # 将二维掩码扩展为三维：[时间步, 256, 256]（每个时间步用相同的掩码）
#         mask_3d = np.expand_dims(mask, axis=0)
#         mask_3d = np.repeat(mask_3d, pred.shape[0], axis=0)
        
#         mask_bool = mask_3d.astype(bool)
#         pred_flat = pred.flatten()[mask_bool.flatten()]
#         true_flat = true.flatten()[mask_bool.flatten()]
#     else:
#         pred_flat = pred.flatten()
#         true_flat = true.flatten()
    
#     # 处理空数据情况
#     if len(pred_flat) == 0 or len(true_flat) == 0:
#         return 0, 0, 0
    
#     # 计算相关系数R
#     numerator = np.sum((pred_flat - np.mean(pred_flat)) * (true_flat - np.mean(true_flat)))
#     denominator = np.sqrt(np.sum((pred_flat - np.mean(pred_flat))**2) * np.sum((true_flat - np.mean(true_flat))** 2))
#     R = numerator / denominator if denominator != 0 else 0
    
#     # 计算均方根误差RMSE
#     RMSE = np.sqrt(np.mean((pred_flat - true_flat)**2))
    
#     # 计算平均绝对误差MAE
#     MAE = np.mean(np.abs(pred_flat - true_flat))
    
#     return R, RMSE, MAE

# def save_metrics_to_excel(all_metrics_data, save_path):
#     """
#     将所有指标数据保存为Excel表格
#     all_metrics_data: 存储所有模型-时间段-区域指标的列表
#     save_path: Excel保存路径（如'./arctic_metrics_results.xlsx'）
#     """
#     # 初始化数据列表，用于构建DataFrame
#     metrics_records = []
    
#     # 遍历所有模型数据
#     for model_data in all_metrics_data:
#         for period_data in model_data:
#             model_name = period_data['model']
#             time_period = period_data['time_period']
            
#             # 处理SIC指标
#             if 'sic' in period_data:
#                 sic_metrics = period_data['sic']
#                 for area, metrics in sic_metrics.items():
#                     # 计算平均值（若有多天数据）
#                     mae_mean = round(np.mean(metrics['MAE']), 4)
#                     rmse_mean = round(np.mean(metrics['RMSE']), 4)
#                     corr_mean = round(np.mean(metrics['CORR']), 4)
                    
#                     metrics_records.append({
#                         '模型名称': model_name,
#                         '时间段': time_period,
#                         '区域': area,
#                         '指标类型': 'SIC',
#                         'MAE（平均值）': mae_mean,
#                         'RMSE（平均值）': rmse_mean,
#                         'CORR（平均值）': corr_mean,
#                         'MAE（每日值）': str(metrics['MAE']),
#                         'RMSE（每日值）': str(metrics['RMSE']),
#                         'CORR（每日值）': str(metrics['CORR'])
#                     })
            
#             # 处理SIV U分量指标
#             if 'siv_u' in period_data:
#                 siv_u_metrics = period_data['siv_u']
#                 for area, metrics in siv_u_metrics.items():
#                     mae_mean = round(np.mean(metrics['MAE']), 4)
#                     rmse_mean = round(np.mean(metrics['RMSE']), 4)
#                     corr_mean = round(np.mean(metrics['CORR']), 4)
                    
#                     metrics_records.append({
#                         '模型名称': model_name,
#                         '时间段': time_period,
#                         '区域': area,
#                         '指标类型': 'SIV_U',
#                         'MAE（平均值）': mae_mean,
#                         'RMSE（平均值）': rmse_mean,
#                         'CORR（平均值）': corr_mean,
#                         'MAE（每日值）': str(metrics['MAE']),
#                         'RMSE（每日值）': str(metrics['RMSE']),
#                         'CORR（每日值）': str(metrics['CORR'])
#                     })
            
#             # 处理SIV V分量指标
#             if 'siv_v' in period_data:
#                 siv_v_metrics = period_data['siv_v']
#                 for area, metrics in siv_v_metrics.items():
#                     mae_mean = round(np.mean(metrics['MAE']), 4)
#                     rmse_mean = round(np.mean(metrics['RMSE']), 4)
#                     corr_mean = round(np.mean(metrics['CORR']), 4)
                    
#                     metrics_records.append({
#                         '模型名称': model_name,
#                         '时间段': time_period,
#                         '区域': area,
#                         '指标类型': 'SIV_V',
#                         'MAE（平均值）': mae_mean,
#                         'RMSE（平均值）': rmse_mean,
#                         'CORR（平均值）': corr_mean,
#                         'MAE（每日值）': str(metrics['MAE']),
#                         'RMSE（每日值）': str(metrics['RMSE']),
#                         'CORR（每日值）': str(metrics['CORR'])
#                     })
    
#     # 构建DataFrame
#     df = pd.DataFrame(metrics_records)
    
#     # 创建Excel写入器，支持多sheet
#     with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
#         # 总表：所有指标
#         df.to_excel(writer, sheet_name='所有指标汇总', index=False)
        
#         # 分表：按指标类型拆分
#         for metric_type in ['SIC', 'SIV_U', 'SIV_V']:
#             df_sub = df[df['指标类型'] == metric_type]
#             df_sub.to_excel(writer, sheet_name=f'{metric_type}指标', index=False)
    
#     print(f"✅ 指标数据已成功保存至：{save_path}")
#     return df

# # -------------------------- 主程序 --------------------------
# model_names = ['Unet_siv']  # 'Unet_sic', 'Unet_siv', 'HISUnet', 'DAGUnet'
# year_indexs = [[2000,2007], [2005,2012], [2012,2019], [2013,2020], [2000,2023]] # 
# area_indexs = ['Central Arctic','Beaufort Sea','Chukchi Sea','East Siberian Sea',
#                'Laptev Sea','Kara Sea','Barents Sea','East Greenland Sea',
#                'baffin and Labrador Seas','Canadian Archipelago']
# area_mask_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

# all_metrics_data = []  # 存储所有模型的指标数据
# SIC_FLAG = False
# SIV_FLAG = True
# num_days = 7

# # 读取掩码数据
# ds_mask = xr.open_dataset(rf'F:/DAGUnet_code/Data/mask_reprojected.nc')
# arctic_mask = crop_center(ds_mask['arctic_mask'].values)  # shape[256, 256]

# # 为每个区域创建对应的二维掩码
# area_masks = {
#     area: (arctic_mask == num) for area, num in zip(area_indexs, area_mask_num)
# }
# area_masks['Whole Arctic'] = (arctic_mask != 0)

# for model_name in model_names:
#     model_metrics = []  # 存储当前模型的所有时间段指标
#     for year_index in year_indexs:
#         start_year = year_index[0]
#         end_year = year_index[1]
#         time_period = f'{start_year}-{end_year}'
#         print(f"--- 正在处理模型: {model_name}，时间段: {time_period} ---")
        
#         # 读取预测结果
#         ds = xr.open_dataset(rf'E:/DAGUnet_code/newfolder/predict_{model_name}_pre7_{start_year}_{end_year}.nc')

#         # 初始化区域指标
#         sic_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}
#         siv_u_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}
#         siv_v_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}

#         # 按天处理数据
#         for day_idx in range(num_days):
#             day_number = day_idx + 1
#             print(f"---- 处理第 {day_number} 天 ----")

#             if SIC_FLAG:
#                 # 读取单日SIC数据（三维中的单天：[256,256]）
#                 SIC_pred = ds[f'SIC_pred_day{day_idx+1}'].values
#                 SIC_true = ds[f'SIC_true_day{day_idx+1}'].values
#                 SIC_pred = denormalize(SIC_pred, 0, 100)
#                 SIC_true = denormalize(SIC_true, 0, 100)

#                 # 计算每个区域的SIC指标
#                 for area, mask in area_masks.items():
#                     r, rmse, mae = calculate_metrics(SIC_pred, SIC_true, mask)
#                     sic_metrics[area]['MAE'].append(round(mae, 4))
#                     sic_metrics[area]['RMSE'].append(round(rmse, 4))
#                     sic_metrics[area]['CORR'].append(round(r, 4))

#             if SIV_FLAG:
#                 # 读取单日SIV数据
#                 SIV_u_pred = ds[f'SIV_u_pred_day{day_idx+1}'].values
#                 SIV_u_true = ds[f'SIV_u_true_day{day_idx+1}'].values
#                 SIV_v_pred = ds[f'SIV_v_pred_day{day_idx+1}'].values
#                 SIV_v_true = ds[f'SIV_v_true_day{day_idx+1}'].values
                
#                 # 反归一化+单位转换
#                 SIV_u_pred = denormalize(SIV_u_pred, -60, 54) * 0.864
#                 SIV_u_true = denormalize(SIV_u_true, -60, 54) * 0.864
#                 SIV_v_pred = denormalize(SIV_v_pred, -58, 54) * 0.864
#                 SIV_v_true = denormalize(SIV_v_true, -58, 54) * 0.864

#                 # 计算SIV U分量指标
#                 for area, mask in area_masks.items():
#                     r, rmse, mae = calculate_metrics(SIV_u_pred, SIV_u_true, mask)
#                     siv_u_metrics[area]['MAE'].append(round(mae, 4))
#                     siv_u_metrics[area]['RMSE'].append(round(rmse, 4))
#                     siv_u_metrics[area]['CORR'].append(round(r, 4))

#                 # 计算SIV V分量指标
#                 for area, mask in area_masks.items():
#                     r, rmse, mae = calculate_metrics(SIV_v_pred, SIV_v_true, mask)
#                     siv_v_metrics[area]['MAE'].append(round(mae, 4))
#                     siv_v_metrics[area]['RMSE'].append(round(rmse, 4))
#                     siv_v_metrics[area]['CORR'].append(round(r, 4))

#         # 打印结果
#         if SIC_FLAG:
#             print("\n=================== SIC 区域指标 =======================")
#             for area in area_masks.keys():
#                 print(f"\n----- {area} -----")
#                 print(f"MAE: {sic_metrics[area]['MAE']}，平均值: {round(np.mean(sic_metrics[area]['MAE']), 4)}")
#                 print(f"RMSE: {sic_metrics[area]['RMSE']}，平均值: {round(np.mean(sic_metrics[area]['RMSE']), 4)}")
#                 print(f"CORR: {sic_metrics[area]['CORR']}，平均值: {round(np.mean(sic_metrics[area]['CORR']), 4)}")

#         if SIV_FLAG:
#             print("\n=================== SIV U分量 区域指标 =======================")
#             for area in area_masks.keys():
#                 print(f"\n----- {area} -----")
#                 print(f"MAE: {siv_u_metrics[area]['MAE']}，平均值: {round(np.mean(siv_u_metrics[area]['MAE']), 4)}")
#                 print(f"RMSE: {siv_u_metrics[area]['RMSE']}，平均值: {round(np.mean(siv_u_metrics[area]['RMSE']), 4)}")
#                 print(f"CORR: {siv_u_metrics[area]['CORR']}，平均值: {round(np.mean(siv_u_metrics[area]['CORR']), 4)}")

#             print("\n=================== SIV V分量 区域指标 =======================")
#             for area in area_masks.keys():
#                 print(f"\n----- {area} -----")
#                 print(f"MAE: {siv_v_metrics[area]['MAE']}，平均值: {round(np.mean(siv_v_metrics[area]['MAE']), 4)}")
#                 print(f"RMSE: {siv_v_metrics[area]['RMSE']}，平均值: {round(np.mean(siv_v_metrics[area]['RMSE']), 4)}")
#                 print(f"CORR: {siv_v_metrics[area]['CORR']}，平均值: {round(np.mean(siv_v_metrics[area]['CORR']), 4)}")

#         # 保存当前时间段的指标数据
#         model_metrics.append({
#             'model': model_name,
#             'time_period': time_period,
#             'sic': sic_metrics,
#             'siv_u': siv_u_metrics,
#             'siv_v': siv_v_metrics
#         })
    
#     # 将当前模型的所有时间段数据加入总列表
#     all_metrics_data.append(model_metrics)

# # 保存指标数据到Excel
# save_path = r'F:/DAGUnet_code/newresult/differentarea/metrics_diff_area_Unet.xlsx'  # 可修改保存路径
# save_metrics_to_excel(all_metrics_data, save_path)

# print('成功保存为表格文件')

'''
用不同的点型表示不同区域的指标,绘制箱线图
'''
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # --------------------------
# # 1. 全局绘图风格配置
# # --------------------------
# def set_plot_style():
#     plt.rcParams['font.sans-serif'] = ['Arial', 'WenQuanYi Zen Hei']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.rcParams['axes.linewidth'] = 1.2
#     plt.rcParams['grid.linewidth'] = 0.8
#     plt.rcParams['grid.alpha'] = 0.3
#     plt.rcParams['lines.linewidth'] = 1.2
#     plt.rcParams['axes.titlesize'] = 18
#     plt.rcParams['axes.labelsize'] = 18
#     plt.rcParams['xtick.labelsize'] = 18
#     plt.rcParams['ytick.labelsize'] = 12
#     plt.rcParams['text.usetex'] = False  # 避免LaTeX渲染问题

# # --------------------------
# # 2. 数据读取与预处理
# # --------------------------
# def load_and_preprocess_data(file_path):
#     df = pd.read_csv(file_path, sep='\t')
#     print(f"✅ Data Shape: {df.shape} (rows×cols), Columns: {list(df.columns)}")
    
#     # 修复列名（若需）
#     if len(df.columns) == 1 and '\t' in df.columns[0]:
#         df = pd.read_csv(
#             file_path,
#             sep='\t',
#             names=['Model', 'Time_Period', 'Region', 'MAE(Mean)', 'RMSE(Mean)', 'CORR(Mean)'],
#             skiprows=1
#         )
    
#     # 转换数值列
#     numeric_cols = ['MAE(Mean)', 'RMSE(Mean)', 'CORR(Mean)']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     df = df.dropna(subset=numeric_cols)
#     print(f"📊 Models: {df['Model'].unique()}, Arctic Regions: {df['Region'].nunique()}")
#     print(f"✅ Cleaned Data Shape: {df.shape}")
#     return df

# # --------------------------
# # 3. 核心功能：为每个区域分配唯一标记（点型）
# # --------------------------
# def get_region_marker_map(regions):
#     """
#     为每个区域分配唯一的点型（marker），覆盖所有北极区域
#     regions: 数据集中的区域列表
#     return: 区域-点型映射字典
#     """
#     # 精选15种常用且易区分的点型（覆盖11个北极区域完全足够）
#     available_markers = [
#         'o', 's', '^', 'D', 'x', 'p', '*', 'h', '+', 'v', '>', 
#         '<', '1', '2', '3']
    
#     # 为每个区域分配唯一标记（按区域名排序，确保一致性）
#     sorted_regions = sorted(regions.unique())
#     region_marker = {
#         region: available_markers[i] 
#         for i, region in enumerate(sorted_regions)
#     }
    
#     # 打印区域-点型映射表（便于后续图表解读）
#     print("\n🌍 区域-点型（Marker）映射表:")
#     print("-" * 50)
#     for region, marker in region_marker.items():
#         print(f"{region:<25} → Marker: '{marker}'")
#     print("-" * 50)
    
#     return region_marker

# # --------------------------
# # 4. 核心箱线图绘制函数（新增区域点型区分）
# # --------------------------
# def plot_sic_boxplot(df, metric_col, group_by_col, save_path):
#     # 准备分组数据
#     groups = df[group_by_col].unique()
#     box_data = [df[df[group_by_col] == g][metric_col].values for g in groups]
    
#     # 创建画布（加宽以容纳文本框和图例）
#     fig, ax = plt.subplots(1, 1, figsize=(max(14, len(groups)*4), 8))
    
#     # 颜色方案（模型固定色）
#     colors = ['#2E86AB', '#A23B72', '#F18F01'] if group_by_col == 'Model' else plt.cm.Set3(np.linspace(0,1,len(groups)))
    
#     # 关键：获取区域-点型映射
#     region_marker_map = get_region_marker_map(df['Region'])
    
#     # 绘制箱线图（基础箱体）
#     bp = ax.boxplot(
#         box_data,
#         positions=np.arange(1, len(groups)+1),
#         widths=0.8,
#         patch_artist=True,
#         showfliers=False,  # 关闭默认异常值，后续手动按区域绘制散点
#         medianprops=dict(color='white', linewidth=3)  # 白色粗中位数线（醒目）
#     )
    
#     # 箱体上色
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
    
#     # 按“模型+区域”绘制散点（不同区域用不同点型）
#     for i, g in enumerate(groups):
#         # 提取当前模型的所有数据
#         model_data = df[df[group_by_col] == g]
#         # 按区域分组绘制散点
#         for region in model_data['Region'].unique():
#             # 提取当前区域的数据
#             region_data = model_data[model_data['Region'] == region]
#             x_jitter = np.random.normal(i+1, 0.06, size=len(region_data))  # 轻微抖动避免重叠
#             # 绘制散点（使用区域对应的点型）
#             ax.scatter(
#                 x_jitter, region_data[metric_col],
#                 marker=region_marker_map[region],  # 核心：按区域设置点型
#                 color=colors[i],                 # 按模型设置颜色
#                 s=60,                           # 点大小（适中，便于区分）
#                 edgecolor='black',               # 白色边缘，增强立体感
#                 linewidth=1.0,
#                 alpha=0.8,                       # 透明度，避免遮挡
#                 zorder=3,                        # 散点层级高于箱体
#                 label=f'{region}' if i == 0 else ""  # 仅在第一个模型处添加图例标签（避免重复）
#             )
    
#     # 计算并打印统计指标（保留原功能）
#     print(f"\n📈 {metric_col} 统计指标（按模型）:")
#     for i, g in enumerate(groups):
#         data = df[df[group_by_col] == g][metric_col].values
#         median_val = np.median(data)
#         q1_val = np.percentile(data, 25)
#         q3_val = np.percentile(data, 75)
#         std_val = np.std(data)
#         mean_val = np.mean(data)
#         print(f"\n{g}:")
#         print(f"  Median: {median_val:.4f}, Q1: {q1_val:.4f}, Q3: {q3_val:.4f}")
#         print(f"  Std: {std_val:.4f}, Mean: {mean_val:.4f}")
    
#     # 坐标轴配置（英文）
#     ax.set_xticks(np.arange(1, len(groups)+1))
#     ax.set_xticklabels([str(g) for g in groups], fontsize=12, fontweight='bold')
#     ax.set_xlabel(group_by_col, fontweight='bold', labelpad=15, fontsize=12)
#     # 指标标签映射
#     metric_label = {
#         'MAE(Mean)': 'MAE (Mean)',
#         'RMSE(Mean)': 'RMSE (Mean)',
#         'CORR(Mean)': 'CORR (Mean)'
#     }[metric_col]
#     ax.set_ylabel(metric_label, fontweight='bold', labelpad=15, fontsize=12)
    
#     # 优化y轴范围（按指标类型适配）
#     if metric_col == 'CORR(Mean)':
#         ax.set_ylim(0.85, 1.0)
#         ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))  # 每0.01一个刻度，便于对比
#     else:
#         data_min, data_max = df[metric_col].min(), df[metric_col].max()
#         ax.set_ylim(data_min*0.6, data_max*1.1)  # 留出足够空间显示散点和图例
#         ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    
#     # 英文标题（简洁明了）
#     title_map = {
#         'MAE(Mean)': 'MAE Distribution',
#         'RMSE(Mean)': 'RMSE Distribution',
#         'CORR(Mean)': 'CORR Distribution'
#     }
#     ax.set_title(title_map[metric_col], fontweight='bold', pad=20, fontsize=16)
    
#     # 网格配置
#     ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
#     # 关键：添加区域-点型图例（放在右侧，不遮挡数据）
#     ax.legend(
#         title='Arctic Regions',  # 图例标题
#         title_fontsize=12,
#         fontsize=12,
#         loc='center left',       # 图例位置（右侧中间）
#         bbox_to_anchor=(1, 0.5),
#         frameon=True,
#         fancybox=True,
#         shadow=True
#     )
    
#     # 调整布局（容纳右侧图例）
#     plt.tight_layout(rect=[0, 0.02, 0.85, 0.98])  # 左侧留85%空间给图表，15%给图例
    
#     # 保存高清图片（600dpi，适合论文/报告）
#     plt.savefig(
#         save_path,
#         dpi=600,
#         bbox_inches='tight',
#         facecolor='white',
#         edgecolor='none'
#     )
#     plt.close()
#     print(f"\n📥 Saved to: {save_path}")

# # --------------------------
# # 5. 主函数（一键运行）
# # --------------------------
# def main(file_path='SIV_V_DiffArea_xiangxian.csv'):
#     set_plot_style()
#     df = load_and_preprocess_data(file_path)
    
# #     绘制3个核心指标的箱线图（按模型分组，含区域点型区分）
#     print("\n🎨 正在绘制 MAE 箱线图（含区域点型区分）...")
#     plot_sic_boxplot(df, 'MAE(Mean)', group_by_col='Model', save_path='Different_Area/SIV_V_MAE_by_model.png')
    
#     print("\n🎨 正在绘制 RMSE 箱线图（含区域点型区分）...")
#     plot_sic_boxplot(df, 'RMSE(Mean)', group_by_col='Model', save_path='Different_Area/SIV_V_RMSE_by_model.png')
    
#     print("\n🎨 正在绘制 CORR 箱线图（含区域点型区分）...")
#     plot_sic_boxplot(df, 'CORR(Mean)', group_by_col='Model', save_path='Different_Area/SIV_V_CORR_by_model.png')
    
#     print("\n🎉 All plots generated successfully!")

# # 运行入口
# if __name__ == "__main__":
#     main()

'''
不同区域的平均值
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # SIC (海冰密集度变量) 的平均指标数据
# data = {
#     'Model': [
#         'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet',
#         'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet',
#         'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet'
#     ],
#     'Region': [
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago'
#     ],
#     'MAE': [
#         # DAGUnet (Averaged over 5 periods)
#         1.1782, 3.1970, 3.3283, 2.7686, 2.6681, 3.4447, 4.4168, 3.3908, 2.3768, 1.4878,
#         # HISUnet (Averaged over 5 periods)
#         1.2185, 3.2982, 3.4862, 2.8360, 2.7230, 3.5419, 4.5028, 3.4851, 2.4578, 1.5411,
#         # Unet (Averaged over 5 periods)
#         1.3411, 3.5358, 3.7548, 3.0975, 2.9781, 3.9141, 4.8876, 3.8267, 2.7118, 1.7061
#     ],
#     'RMSE': [
#         # DAGUnet (Averaged over 5 periods)
#         3.5419, 7.8209, 8.8752, 7.8443, 7.6416, 8.9416, 10.9822, 9.3897, 7.3719, 4.4372,
#         # HISUnet (Averaged over 5 periods)
#         3.5583, 7.9135, 9.1500, 7.9575, 7.7497, 9.1256, 11.0967, 9.5752, 7.5501, 4.5126,
#         # Unet (Averaged over 5 periods)
#         3.8340, 8.2439, 9.5445, 8.3582, 8.1672, 9.7712, 11.7589, 10.2238, 8.0435, 4.8711
#     ],
#     'CORR': [
#         # DAGUnet (Averaged over 5 periods)
#         0.9859, 0.9785, 0.9806, 0.9839, 0.9845, 0.9800, 0.9691, 0.9620, 0.9798, 0.9897,
#         # HISUnet (Averaged over 5 periods)
#         0.9852, 0.9774, 0.9793, 0.9832, 0.9839, 0.9792, 0.9678, 0.9602, 0.9782, 0.9890,
#         # Unet (Averaged over 5 periods)
#         0.9829, 0.9739, 0.9760, 0.9802, 0.9808, 0.9748, 0.9622, 0.9535, 0.9738, 0.9860
#     ]
# }

# # SIV_U (海冰速度 U 分量) 的平均指标数据
# data = {
#     'Model': [
#         'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet',
#         'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet',
#         'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet'
#     ],
#     'Region': [
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
#         'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
#         'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago'
#     ],
#     'MAE': [
#         # DAGUnet (Averaged over 5 periods)
#         1.8596, 2.8066, 2.7681, 2.4542, 2.4925, 3.2359, 3.6559, 3.5147, 2.4998, 1.5323,
#         # HISUnet (Averaged over 5 periods)
#         1.8845, 2.8530, 2.8105, 2.4939, 2.5292, 3.2755, 3.6934, 3.5606, 2.5332, 1.5348,
#         # Unet (Averaged over 5 periods)
#         1.9839, 3.0180, 2.9723, 2.6253, 2.6775, 3.4429, 3.8687, 3.7317, 2.6419, 1.4501
#     ],
#     'RMSE': [
#         # DAGUnet (Averaged over 5 periods)
#         4.0175, 6.3541, 6.5518, 5.6586, 5.7601, 7.3941, 8.4287, 8.1691, 5.8617, 3.9268,
#         # HISUnet (Averaged over 5 periods)
#         4.0592, 6.4172, 6.6111, 5.7262, 5.8197, 7.4646, 8.4900, 8.2435, 5.9189, 3.9238,
#         # Unet (Averaged over 5 periods)
#         4.2120, 6.6433, 6.8373, 5.9224, 6.0383, 7.7126, 8.7183, 8.4893, 6.0822, 3.5976
#     ],
#     'CORR': [
#         # DAGUnet (Averaged over 5 periods)
#         0.9575, 0.9482, 0.9507, 0.9547, 0.9542, 0.9500, 0.9333, 0.9351, 0.9490, 0.9452,
#         # HISUnet (Averaged over 5 periods)
#         0.9566, 0.9472, 0.9497, 0.9538, 0.9532, 0.9492, 0.9324, 0.9341, 0.9482, 0.9454,
#         # Unet (Averaged over 5 periods)
#         0.9526, 0.9416, 0.9442, 0.9489, 0.9479, 0.9443, 0.9265, 0.9290, 0.9429, 0.9508
#     ]
# }

# SIV_V (海冰速度 V 分量) 的平均指标数据
data = {
    'Model': [
        'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet', 'DAGUnet',
        'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet', 'HISUnet',
        'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet', 'Unet'
    ],
    'Region': [
        'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
        'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
        'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
        'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago',
        'Central Arctic', 'Beaufort Sea', 'Chukchi Sea', 'East Siberian Sea', 'Laptev Sea', 
        'Kara Sea', 'Barents Sea', 'East Greenland Sea', 'Baffin and Labrador Seas', 'Canadian Archipelago'
    ],
    'MAE': [
        # DAGUnet (Averaged over 5 periods)
        1.8519, 2.6806, 2.6934, 2.5034, 2.5921, 3.2359, 3.9110, 3.9515, 2.8715, 1.4878,
        # HISUnet (Averaged over 5 periods)
        1.8767, 2.7317, 2.7444, 2.5414, 2.6367, 3.2878, 3.9610, 3.9996, 2.9100, 1.4862,
        # Unet (Averaged over 5 periods)
        1.9688, 2.8839, 2.9138, 2.6865, 2.7937, 3.4735, 4.1481, 4.1953, 3.0336, 1.4055
    ],
    'RMSE': [
        # DAGUnet (Averaged over 5 periods)
        3.9934, 6.1306, 6.2731, 5.7600, 5.9540, 7.3516, 8.8756, 9.1772, 6.4674, 3.7508,
        # HISUnet (Averaged over 5 periods)
        4.0298, 6.2081, 6.3533, 5.8143, 6.0145, 7.4326, 8.9566, 9.2554, 6.5360, 3.7431,
        # Unet (Averaged over 5 periods)
        4.1738, 6.4410, 6.6190, 6.0270, 6.2570, 7.7279, 9.2319, 9.5583, 6.7410, 3.4542
    ],
    'CORR': [
        # DAGUnet (Averaged over 5 periods)
        0.9575, 0.9482, 0.9491, 0.9520, 0.9528, 0.9496, 0.9298, 0.9231, 0.9419, 0.9499,
        # HISUnet (Averaged over 5 periods)
        0.9567, 0.9472, 0.9481, 0.9511, 0.9518, 0.9487, 0.9288, 0.9222, 0.9412, 0.9502,
        # Unet (Averaged over 5 periods)
        0.9530, 0.9418, 0.9427, 0.9463, 0.9468, 0.9439, 0.9234, 0.9168, 0.9366, 0.9546
    ]
}

df = pd.DataFrame(data)

# ---------------------------------
# 绘图部分
# ---------------------------------

# 设置Matplotlib和Seaborn的风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['The NewRoman'] # 用于显示中文标签（如'时间段均值'）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 1. 绘制 MAE (MAE_时间段均值) 的折线图

plt.figure(figsize=(8, 6))

# 使用Seaborn的lineplot，它可以自动处理分组、颜色和点型
# style='Model' 会根据Model列自动分配不同的点型
# hue='Model' 会根据Model列自动分配不同的颜色
sns.lineplot(data=df, x='Region', y='MAE', hue='Model', style='Model', 
             markers=True, dashes=False, linewidth=2.5, markersize=8)

plt.title('MAE Changes in Different Areas', fontsize=8)
plt.xlabel('Region', fontsize=8)
plt.ylabel('MAE', fontsize=8)

# 旋转x轴标签，防止重叠
plt.xticks(rotation=45, ha='right')

# 调整图例位置
plt.legend(title='Model', loc='upper left')

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show()

# 2. 绘制 RMSE (RMSE_时间段均值) 的折线图
# 步骤与MAE类似

plt.figure(figsize=(8, 6))

sns.lineplot(data=df, x='Region', y='RMSE', hue='Model', style='Model', 
             markers=True, dashes=False, linewidth=2.5, markersize=8)

plt.title('RMSE Changes in Different Areas', fontsize=8)
plt.xlabel('Region', fontsize=8)
plt.ylabel('RMSE', fontsize=8)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model', loc='upper left')
plt.tight_layout()
plt.savefig('')
plt.show()

# 3. 绘制 CORR (CORR_时间段均值) 的折线图
# 步骤与MAE类似

plt.figure(figsize=(8, 6))

sns.lineplot(data=df, x='Region', y='CORR', hue='Model', style='Model', 
             markers=True, dashes=False, linewidth=2.5, markersize=8)

plt.title('CORR Changes in Different Areas', fontsize=8)
plt.xlabel('Region', fontsize=8)
plt.ylabel('CORR', fontsize=8)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model', loc='lower left') # CORR指标值较高，图例放在左下角
plt.tight_layout()
plt.show()