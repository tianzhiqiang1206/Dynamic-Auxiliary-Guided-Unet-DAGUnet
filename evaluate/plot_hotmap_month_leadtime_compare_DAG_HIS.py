'''
This code can be used to reproduce Supplementrary Fig.1
'''
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ===================== 1. 自定义紫-白-绿发散型配色 =====================
# 定义颜色节点（可根据需求调整深浅）
# 格式：[(数值位置, RGB颜色), ...]，数值位置范围 0-1
colors = [
    (0.0, '#4A148C'),    # 深紫色（最小值）
    (0.2, '#9C27B0'),    # 浅紫色
    (0.5, '#FFFFFF'),    # 白色（中间值）
    (0.8, '#4CAF50'),    # 浅绿色
    (1.0, '#1B5E20')     # 深绿色（最大值）
]

# 创建自定义颜色映射
purple_white_green = LinearSegmentedColormap.from_list(
    'purple_white_green',  # 配色名称
    colors,                # 颜色节点
    N=256                  # 颜色渐变精度
)

yl_or_rd_pu_black = [
    (0.0, '#FFFFCC'),    # 亮黄色（最大值）
    (0.25, '#FF7F00'),   # 橙色
    (0.5, '#D62728'),    # 红色
    (0.75, '#9467BD'),   # 深紫色
    (1.0, '#000000')     # 黑色（最小值）
]

# 创建可直接复用的颜色映射对象
yl_or_rd_pu_black = LinearSegmentedColormap.from_list(
    'yl_or_rd_pu_black',  # 自定义配色名称（可任意命名）
    yl_or_rd_pu_black,
    N=256  # 渐变精度，数值越大过渡越自然
)

year_list = [[2000,2023]]

# ===================== 可选：批量绘制SIV_u和SIV_v的热力图 =====================
# 如果需要绘制SIV_u/SIV_v的指标热力图，可参考以下代码：
def plot_siv_heatmap(data, title, var_name, cmap, vmin=None, vmax=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    
    ax = sns.heatmap(data,
                    annot=True,
                    annot_kws={'size': 14},  # 这里设置字体大小为8（可按需调整，比如10、12）
                    fmt='.4f',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    xticklabels=month_labels,
                    yticklabels=day_labels,
                    cbar=True,
                    cbar_kws={'label': '', 'shrink': 0.8},
                    linewidths=0.5,
                    square=True)
    
    # 调整坐标轴刻度字体大小
    ax.set_xticklabels(ax.get_xticklabels(), size=14)
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    ax.tick_params(labelsize=14)  # 刻度标签的字体大小（根据需要调整数值）
    
    # plt.title(f'2019 {var_name} {title} Changes', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=15)
    plt.ylabel('Lead Time (Days)', fontsize=15)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(rf'F:\DAGUnet_code\figures\{year[1]}_{var_name}_{title}_heatmap.png', dpi=600, bbox_inches='tight')
    # plt.show()

# ===================== 通用绘图函数 =====================
def plot_sic_heatmap(data, title, cmap, vmin=None, vmax=None, fmt='.4f'):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示
    plt.figure(figsize=(12, 6))
    
    # 绘制热力图
    ax = sns.heatmap(data, 
                    annot=True,        # 显示数值
                    annot_kws={'size': 14},  # 这里设置字体大小为8（可按需调整，比如10、12）
                    fmt=fmt,           # 数值格式
                    cmap=cmap,         # 颜色映射
                    vmin=vmin,         # 颜色最小值
                    vmax=vmax,         # 颜色最大值
                    xticklabels=month_labels,  # x轴标签（月份）
                    yticklabels=day_labels,    # y轴标签（Day）
                    cbar=True,         # 显示颜色条
                    cbar_kws={'label': '', 'shrink': 0.8},  # 颜色条配置
                    linewidths=0.5,    # 格子边框宽度
                    square=True)       # 格子正方形
    
    # 调整坐标轴刻度字体大小
    ax.set_xticklabels(ax.get_xticklabels(), size=14)
    ax.set_yticklabels(ax.get_yticklabels(), size=14)
    
    # 设置标题和标签
    plt.xlabel('Month', fontsize=15, labelpad=10)
    plt.ylabel('Lead Time (Days)', fontsize=15, labelpad=10)
    
    # 调整刻度标签
    ax.tick_params(axis='x', rotation=45)  # x轴标签旋转45度
    ax.tick_params(axis='y', rotation=0)    # y轴标签不旋转
    ax.tick_params(labelsize=14)  # 刻度标签的字体大小（根据需要调整数值）
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（可选）
    plt.savefig(rf'F:\DAGUnet_code\figures\{year[1]}_SIC_{title}_heatmap.png', dpi=600, bbox_inches='tight')
    # plt.show()

# 定义逆归一化函数
def inverse_normalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def calculate_metrics(pred, true):
    """计算相关系数(R)、均方根误差(RMSE)和平均绝对误差(MAE)"""
    # 计算相关系数R
    numerator = np.sum((pred - np.mean(pred)) * (true - np.mean(true)))
    denominator = np.sqrt(np.sum((pred - np.mean(pred))**2) * np.sum((true - np.mean(true))**2))
    R = numerator / denominator if denominator != 0 else 0
    # 计算均方根误差RMSE
    RMSE = np.sqrt(np.mean((pred - true)**2))
    # 计算平均绝对误差MAE
    MAE = np.mean(np.abs(pred - true))
    return R, RMSE, MAE
'''
HISUnet的绘图结果
'''
for year in year_list:
    # 读取数据
    file_path = rf"F:\DAGUnet_code\newresult\predict_HISUnet_pre7_{year[0]}_{year[1]}.nc"
    file_path1 = rf"F:\DAGUnet_code\newresult\predict_DAGUnet_pre7_{year[0]}_{year[1]}.nc"
    ds = xr.open_dataset(file_path)
    ds1 = xr.open_dataset(file_path1)

    # 定义归一化时使用的最小值和最大值
    sic_min = 0
    sic_max = 100
    siv_u_min = -60
    siv_u_max = 54
    siv_v_min = -58
    siv_v_max = 54

    # 定义月份日期范围
    # 2. 计算每个月的天数
    month_days = [24,28,31,30,31,30,31,31,30,31,30,24]
    # 3. 计算累计索引（每个月的起始索引）
    cumulative_index = [0]  # 初始索引为0
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    # 4. 生成月份索引字典（包含起始/结束索引）
    month_indexs_2021 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2021.append([start_idx,end_idx])

    # 2. 计算每个月的天数
    month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
    # 3. 计算累计索引（每个月的起始索引）
    cumulative_index = [358]  # 初始索引为0
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    # 4. 生成月份索引字典（包含起始/结束索引）
    month_indexs_2022 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2022.append([start_idx,end_idx])
    
    # 2. 计算每个月的天数
    month_days = [31,28,31,30,31,30,31,31,30,31,30,24]
    # 3. 计算累计索引（每个月的起始索引）
    cumulative_index = [358+365]  # 初始索引为0
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    # 4. 生成月份索引字典（包含起始/结束索引）
    month_indexs_2023 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2023.append([start_idx,end_idx])

    SIC_MAE_list = []   # 保存 SIC 数据的 MAE 指标
    SIC_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
    SIC_CORR_list = []  # 保存 SIC 数据的相关系数指标

    SIV_U_MAE_list = []   # 保存 SIC 数据的 MAE 指标
    SIV_U_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
    SIV_U_CORR_list = []  # 保存 SIC 数据的相关系数指标

    SIV_V_MAE_list = []   # 保存 SIC 数据的 MAE 指标
    SIV_V_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
    SIV_V_CORR_list = []  # 保存 SIC 数据的相关系数指标

    for day_inx in range(1,8):
        SIC_MAE_day = []   # 保存 SIC 数据的 MAE 指标
        SIC_RMSE_day = []  # 保存 SIC 数据的 RMSE 指标
        SIC_CORR_day = []  # 保存 SIC 数据的相关系数指标

        SIV_U_MAE_day = []   # 保存 SIC 数据的 MAE 指标
        SIV_U_RMSE_day = []  # 保存 SIC 数据的 RMSE 指标
        SIV_U_CORR_day = []  # 保存 SIC 数据的相关系数指标

        SIV_V_MAE_day = []   # 保存 SIC 数据的 MAE 指标
        SIV_V_RMSE_day = []  # 保存 SIC 数据的 RMSE 指标
        SIV_V_CORR_day = []  # 保存 SIC 数据的相关系数指标

        for month_inx in range(1,13):       
            # 2021年当前月份索引
            s21, e21 = month_indexs_2021[month_inx-1]
            # 2022年当前月份索引
            s22, e22 = month_indexs_2022[month_inx-1]
            # 2023年当前月份索引
            s23, e23 = month_indexs_2023[month_inx-1]

            # SIC 数据拼接
            SIC_pred_21 = ds[f'SIC_pred_day{day_inx}'][s21:e21] # [24,256,256]
            SIC_pred_22 = ds[f'SIC_pred_day{day_inx}'][s22:e22] # [31,256,256]
            SIC_pred_23 = ds[f'SIC_pred_day{day_inx}'][s23:e23] # [31,256,256]
            SIC_pred_all = np.concatenate([SIC_pred_21, SIC_pred_22, SIC_pred_23], axis=0)  # 合并三年数据

            SIC_true_21 = ds[f'SIC_true_day{day_inx}'][s21:e21]
            SIC_true_22 = ds[f'SIC_true_day{day_inx}'][s22:e22]
            SIC_true_23 = ds[f'SIC_true_day{day_inx}'][s23:e23]
            SIC_true_all = np.concatenate([SIC_true_21, SIC_true_22, SIC_true_23], axis=0)

            # SIV U分量数据拼接
            SIV_U_pred_21 = ds[f'SIV_u_pred_day{day_inx}'][s21:e21]
            SIV_U_pred_22 = ds[f'SIV_u_pred_day{day_inx}'][s22:e22]
            SIV_U_pred_23 = ds[f'SIV_u_pred_day{day_inx}'][s23:e23]
            SIV_U_pred_all = np.concatenate([SIV_U_pred_21, SIV_U_pred_22, SIV_U_pred_23], axis=0)

            SIV_U_true_21 = ds[f'SIV_u_true_day{day_inx}'][s21:e21]
            SIV_U_true_22 = ds[f'SIV_u_true_day{day_inx}'][s22:e22]
            SIV_U_true_23 = ds[f'SIV_u_true_day{day_inx}'][s23:e23]
            SIV_U_true_all = np.concatenate([SIV_U_true_21, SIV_U_true_22, SIV_U_true_23], axis=0)

            # SIV V分量数据拼接
            SIV_V_pred_21 = ds[f'SIV_v_pred_day{day_inx}'][s21:e21]
            SIV_V_pred_22 = ds[f'SIV_v_pred_day{day_inx}'][s22:e22]
            SIV_V_pred_23 = ds[f'SIV_v_pred_day{day_inx}'][s23:e23]
            SIV_V_pred_all = np.concatenate([SIV_V_pred_21, SIV_V_pred_22, SIV_V_pred_23], axis=0)

            SIV_V_true_21 = ds[f'SIV_v_true_day{day_inx}'][s21:e21]
            SIV_V_true_22 = ds[f'SIV_v_true_day{day_inx}'][s22:e22]
            SIV_V_true_23 = ds[f'SIV_v_true_day{day_inx}'][s23:e23]
            SIV_V_true_all = np.concatenate([SIV_V_true_21, SIV_V_true_22, SIV_V_true_23], axis=0)

            # 对SIC和SIV数据进行逆归一化
            SIC_pred = inverse_normalize(SIC_pred_all, sic_min, sic_max)
            SIC_true = inverse_normalize(SIC_true_all, sic_min, sic_max)
            SIV_U_pred = inverse_normalize(SIV_U_pred_all, siv_u_min, siv_u_max)
            SIV_U_true = inverse_normalize(SIV_U_true_all, siv_u_min, siv_u_max)
            SIV_V_pred = inverse_normalize(SIV_V_pred_all, siv_v_min, siv_v_max)
            SIV_V_true = inverse_normalize(SIV_V_true_all, siv_v_min, siv_v_max)

            # 计算SIC和SIV的评估指标
            SIC_CORR, SIC_RMSE, SIC_MAE = calculate_metrics(SIC_pred, SIC_true)
            SIV_U_CORR, SIV_U_RMSE, SIV_U_MAE = calculate_metrics(SIV_U_pred, SIV_U_true)
            SIV_V_CORR, SIV_V_RMSE, SIV_V_MAE = calculate_metrics(SIV_V_pred, SIV_V_true)

            # SIC 数据拼接
            SIC_pred_21 = ds1[f'SIC_pred_day{day_inx}'][s21:e21] # [24,256,256]
            SIC_pred_22 = ds1[f'SIC_pred_day{day_inx}'][s22:e22] # [31,256,256]
            SIC_pred_23 = ds1[f'SIC_pred_day{day_inx}'][s23:e23] # [31,256,256]
            SIC_pred_all = np.concatenate([SIC_pred_21, SIC_pred_22, SIC_pred_23], axis=0)  # 合并三年数据

            SIC_true_21 = ds1[f'SIC_true_day{day_inx}'][s21:e21]
            SIC_true_22 = ds1[f'SIC_true_day{day_inx}'][s22:e22]
            SIC_true_23 = ds1[f'SIC_true_day{day_inx}'][s23:e23]
            SIC_true_all = np.concatenate([SIC_true_21, SIC_true_22, SIC_true_23], axis=0)

            # SIV U分量数据拼接
            SIV_U_pred_21 = ds1[f'SIV_u_pred_day{day_inx}'][s21:e21]
            SIV_U_pred_22 = ds1[f'SIV_u_pred_day{day_inx}'][s22:e22]
            SIV_U_pred_23 = ds1[f'SIV_u_pred_day{day_inx}'][s23:e23]
            SIV_U_pred_all = np.concatenate([SIV_U_pred_21, SIV_U_pred_22, SIV_U_pred_23], axis=0)

            SIV_U_true_21 = ds1[f'SIV_u_true_day{day_inx}'][s21:e21]
            SIV_U_true_22 = ds1[f'SIV_u_true_day{day_inx}'][s22:e22]
            SIV_U_true_23 = ds1[f'SIV_u_true_day{day_inx}'][s23:e23]
            SIV_U_true_all = np.concatenate([SIV_U_true_21, SIV_U_true_22, SIV_U_true_23], axis=0)

            # SIV V分量数据拼接
            SIV_V_pred_21 = ds1[f'SIV_v_pred_day{day_inx}'][s21:e21]
            SIV_V_pred_22 = ds1[f'SIV_v_pred_day{day_inx}'][s22:e22]
            SIV_V_pred_23 = ds1[f'SIV_v_pred_day{day_inx}'][s23:e23]
            SIV_V_pred_all = np.concatenate([SIV_V_pred_21, SIV_V_pred_22, SIV_V_pred_23], axis=0)

            SIV_V_true_21 = ds1[f'SIV_v_true_day{day_inx}'][s21:e21]
            SIV_V_true_22 = ds1[f'SIV_v_true_day{day_inx}'][s22:e22]
            SIV_V_true_23 = ds1[f'SIV_v_true_day{day_inx}'][s23:e23]
            SIV_V_true_all = np.concatenate([SIV_V_true_21, SIV_V_true_22, SIV_V_true_23], axis=0)

            # 对SIC和SIV数据进行逆归一化
            SIC_pred1 = inverse_normalize(SIC_pred_all, sic_min, sic_max)
            SIC_true1 = inverse_normalize(SIC_true_all, sic_min, sic_max)
            SIV_U_pred1 = inverse_normalize(SIV_U_pred_all, siv_u_min, siv_u_max)
            SIV_U_true1 = inverse_normalize(SIV_U_true_all, siv_u_min, siv_u_max)
            SIV_V_pred1 = inverse_normalize(SIV_V_pred_all, siv_v_min, siv_v_max)
            SIV_V_true1 = inverse_normalize(SIV_V_true_all, siv_v_min, siv_v_max)

            # 计算SIC和SIV的评估指标
            SIC_CORR1, SIC_RMSE1, SIC_MAE1 = calculate_metrics(SIC_pred1, SIC_true1)
            SIV_U_CORR1, SIV_U_RMSE1, SIV_U_MAE1 = calculate_metrics(SIV_U_pred1, SIV_U_true1)
            SIV_V_CORR1, SIV_V_RMSE1, SIV_V_MAE1 = calculate_metrics(SIV_V_pred1, SIV_V_true1)

            # 存储 SIC 每日指标到 daily_metrics 字典
            SIC_MAE_day.append(round(float(SIC_MAE-SIC_MAE1),4))
            SIC_RMSE_day.append(round(float(SIC_RMSE-SIC_RMSE1),4))
            SIC_CORR_day.append(round(float(SIC_CORR-SIC_CORR1),4))

            SIV_U_MAE_day.append(round(float(SIV_U_MAE-SIV_U_MAE1),4))
            SIV_U_RMSE_day.append(round(float(SIV_U_RMSE-SIV_U_RMSE1),4))
            SIV_U_CORR_day.append(round(float(SIV_U_CORR-SIV_U_CORR1),4))
            
            SIV_V_MAE_day.append(round(float(SIV_V_MAE-SIV_V_MAE1),4))
            SIV_V_RMSE_day.append(round(float(SIV_V_RMSE-SIV_V_RMSE1),4))
            SIV_V_CORR_day.append(round(float(SIV_V_CORR-SIV_V_CORR1),4))

        SIC_MAE_list.append(SIC_MAE_day)
        SIC_RMSE_list.append(SIC_RMSE_day)
        SIC_CORR_list.append(SIC_CORR_day)

        SIV_U_MAE_list.append(SIV_U_MAE_day)
        SIV_U_RMSE_list.append(SIV_U_RMSE_day)
        SIV_U_CORR_list.append(SIV_U_CORR_day)

        SIV_V_MAE_list.append(SIV_V_MAE_day)
        SIV_V_RMSE_list.append(SIV_V_RMSE_day)
        SIV_V_CORR_list.append(SIV_V_CORR_day)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # ===================== 数据预处理与格式调整 =====================
    SIC_MAE_array = np.array(SIC_MAE_list)       # shape: (7, 12) 7个day_inx, 12个月份
    SIC_RMSE_array = np.array(SIC_RMSE_list)     # shape: (7, 12)
    SIC_CORR_array = np.array(SIC_CORR_list)     # shape: (7, 12)

    SIV_U_MAE_array = np.array(SIV_U_MAE_list)       # shape: (7, 12) 7个day_inx, 12个月份
    SIV_U_RMSE_array = np.array(SIV_U_RMSE_list)     # shape: (7, 12)
    SIV_U_CORR_array = np.array(SIV_U_CORR_list)     # shape: (7, 12)

    SIV_V_MAE_array = np.array(SIV_V_MAE_list)       # shape: (7, 12) 7个day_inx, 12个月份
    SIV_V_RMSE_array = np.array(SIV_V_RMSE_list)     # shape: (7, 12)
    SIV_V_CORR_array = np.array(SIV_V_CORR_list)     # shape: (7, 12)

    # print(SIC_MAE_array)
    # print(SIC_RMSE_array)
    # print(SIC_CORR_array)

    # 定义坐标轴标签
    day_labels = [f'{i}' for i in range(1, 8)]  # Day 1 到 Day 7
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(np.min(SIC_MAE_array))

    # 1. MAE热力图（数值越小越好，用冷色调）
    plot_sic_heatmap(SIC_MAE_array, 
                    title='Average MAE', 
                    cmap='RdBu_r',  # 反向蓝色系，越小越亮
                    vmin=0.25, 
                    vmax=-0.25
    )

    # 2. RMSE热力图（数值越小越好，用绿色系）
    plot_sic_heatmap(SIC_RMSE_array, 
                    title='Average RMSE', 
                    #  cmap = purple_white_green,
                    cmap='RdBu_r',  # 反向绿色系
                    vmin=0.25, 
                    vmax=-0.25
    )

    # # 3. CORR热力图（数值越接近1越好，用红黄色系）
    # plot_sic_heatmap(SIC_CORR_array, 
    #                 title='Average CORR', 
    #                 #  cmap='YlOrRd',    # 黄-橙-红，越大越红
    #                 cmap = 'RdBu_r',
    #                 vmin=0.05,  # 相关系数通常≥0
    #                 vmax=-0.05,
    #                 fmt='.4f'
    # )

    # 示例：绘制SIV_u MAE热力图（取消注释即可运行）
    plot_siv_heatmap(SIV_U_MAE_array, title = 'Average MAE', var_name = 'SIV_u', cmap = 'RdBu_r', vmin = -0.15, vmax = 0.15)
    plot_siv_heatmap(SIV_U_RMSE_array, title = 'Average RMSE', var_name = 'SIV_u', cmap = 'RdBu_r',vmin = -0.15, vmax = 0.15)
    # plot_siv_heatmap(SIV_U_CORR_array, title = 'Average CORR', var_name = 'SIV_u', cmap = 'RdBu', vmin = -0.05, vmax = 0.05)

    # 示例：绘制SIV_v CORR热力图（取消注释即可运行）
    plot_siv_heatmap(SIV_V_MAE_array, title = 'Average MAE', var_name = 'SIV_v', cmap = 'RdBu_r', vmin = -0.15, vmax = 0.15)
    plot_siv_heatmap(SIV_V_RMSE_array, title = 'Average RMSE', var_name = 'SIV_v', cmap = 'RdBu_r', vmin = -0.15, vmax = 0.15)
    # plot_siv_heatmap(SIV_V_CORR_array, title = 'Average CORR', var_name = 'SIV_v', cmap = 'RdBu', vmin = -0.05, vmax = 0.05)


    