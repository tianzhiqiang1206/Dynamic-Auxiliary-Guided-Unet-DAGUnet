'''
This code can be used to reproduce Fig.1
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

colors = [
    (0.0, '#4A148C'),  
    (0.2, '#9C27B0'),
    (0.5, '#FFFFFF'), 
    (0.8, '#4CAF50'),   
    (1.0, '#1B5E20') 
]

purple_white_green = LinearSegmentedColormap.from_list(
    'purple_white_green', 
    colors,     
    N=256         
)

yl_or_rd_pu_black = [
    (0.0, '#FFFFCC'),
    (0.25, '#FF7F00'), 
    (0.5, '#D62728'),
    (0.75, '#9467BD'),
    (1.0, '#000000') 
]

yl_or_rd_pu_black = LinearSegmentedColormap.from_list(
    'yl_or_rd_pu_black', 
    yl_or_rd_pu_black,
    N=256 
)

year_list = [[2000,2023]] # [2000,2007],[2005,2012],[2012,2019],[2013,2020]

def plot_siv_heatmap(data, title, var_name, cmap, vmin=None, vmax=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    
    ax = sns.heatmap(data,
                    annot=True,
                    annot_kws={'size': 14}, 
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
    
    ax.set_xticklabels(ax.get_xticklabels(), size=14)
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    ax.tick_params(labelsize=14)
    
    plt.xlabel('Month', fontsize=15)
    plt.ylabel('Lead Time (Days)', fontsize=15)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(rf'F:\DAGUnet_code\figures\{year[1]}_{var_name}_{title}_heatmap.png', dpi=600, bbox_inches='tight')
    
def plot_sic_heatmap(data, title, cmap, vmin=None, vmax=None, fmt='.4f'):
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.figure(figsize=(12, 6))
    
    ax = sns.heatmap(data, 
                    annot=True,
                    annot_kws={'size': 14},
                    fmt=fmt,  
                    cmap=cmap,  
                    vmin=vmin,  
                    vmax=vmax,   
                    xticklabels=month_labels, 
                    yticklabels=day_labels,
                    cbar=True, 
                    cbar_kws={'label': '', 'shrink': 0.8},
                    linewidths=0.5,
                    square=True) 
    
    ax.set_xticklabels(ax.get_xticklabels(), size=14)
    ax.set_yticklabels(ax.get_yticklabels(), size=14)
    
    plt.xlabel('Month', fontsize=15, labelpad=10)
    plt.ylabel('Lead Time (Days)', fontsize=15, labelpad=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0) 
    ax.tick_params(labelsize=14) 
    
    plt.tight_layout()
    plt.savefig(rf'F:\DAGUnet_code\figures\{year[1]}_SIC_{title}_heatmap.png', dpi=600, bbox_inches='tight')

def inverse_normalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def calculate_metrics(pred, true):
    numerator = np.sum((pred - np.mean(pred)) * (true - np.mean(true)))
    denominator = np.sqrt(np.sum((pred - np.mean(pred))**2) * np.sum((true - np.mean(true))**2))
    R = numerator / denominator if denominator != 0 else 0
    RMSE = np.sqrt(np.mean((pred - true)**2))
    MAE = np.mean(np.abs(pred - true))
    return R, RMSE, MAE

for year in year_list:
    file_path = rf"F:\DAGUnet_code\newresult\predict_DAGUnet_pre7_{year[0]}_{year[1]}.nc"
    ds = xr.open_dataset(file_path)

    sic_min = 0
    sic_max = 100
    siv_u_min = -60
    siv_u_max = 54
    siv_v_min = -58
    siv_v_max = 54

    start_date = datetime(2019, 1, 8)
    end_date = datetime(2019, 12, 24)

    month_days = [24,28,31,30,31,30,31,31,30,31,30,31]
    cumulative_index = [0]
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    month_indexs_2021 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2021.append([start_idx,end_idx])

    month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
    cumulative_index = [358] 
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    month_indexs_2022 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2022.append([start_idx,end_idx])
    
    month_indexs_2022 = [[358, 389], [389, 417], [417, 448], [448, 478], 
                         [478, 509], [509, 539], [539, 570], [570, 601], 
                         [601, 631], [631, 662], [662, 692], [692, 723]]
    
    month_days = [31,28,31,30,31,30,31,31,30,31,30,24]
    cumulative_index = [358+365] 
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)
    month_indexs_2023 = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs_2023.append([start_idx,end_idx])

    SIC_MAE_list = []
    SIC_RMSE_list = [] 
    SIC_CORR_list = [] 

    SIV_U_MAE_list = [] 
    SIV_U_RMSE_list = [] 
    SIV_U_CORR_list = [] 

    SIV_V_MAE_list = [] 
    SIV_V_RMSE_list = []
    SIV_V_CORR_list = [] 

    for day_inx in range(1,8):
        SIC_MAE_day = [] 
        SIC_RMSE_day = []
        SIC_CORR_day = [] 

        SIV_U_MAE_day = []  
        SIV_U_RMSE_day = [] 
        SIV_U_CORR_day = []

        SIV_V_MAE_day = [] 
        SIV_V_RMSE_day = []
        SIV_V_CORR_day = []  

        for month_inx in range(1,13):       
            s21, e21 = month_indexs_2021[month_inx-1]
            s22, e22 = month_indexs_2022[month_inx-1]
            s23, e23 = month_indexs_2023[month_inx-1]

            SIC_pred_21 = ds[f'SIC_pred_day{day_inx}'][s21:e21]
            SIC_pred_22 = ds[f'SIC_pred_day{day_inx}'][s22:e22]
            SIC_pred_23 = ds[f'SIC_pred_day{day_inx}'][s23:e23] 
            SIC_pred_all = np.concatenate([SIC_pred_21, SIC_pred_22, SIC_pred_23], axis=0)

            SIC_true_21 = ds[f'SIC_true_day{day_inx}'][s21:e21]
            SIC_true_22 = ds[f'SIC_true_day{day_inx}'][s22:e22]
            SIC_true_23 = ds[f'SIC_true_day{day_inx}'][s23:e23]
            SIC_true_all = np.concatenate([SIC_true_21, SIC_true_22, SIC_true_23], axis=0)

            SIV_U_pred_21 = ds[f'SIV_u_pred_day{day_inx}'][s21:e21]
            SIV_U_pred_22 = ds[f'SIV_u_pred_day{day_inx}'][s22:e22]
            SIV_U_pred_23 = ds[f'SIV_u_pred_day{day_inx}'][s23:e23]
            SIV_U_pred_all = np.concatenate([SIV_U_pred_21, SIV_U_pred_22, SIV_U_pred_23], axis=0)

            SIV_U_true_21 = ds[f'SIV_u_true_day{day_inx}'][s21:e21]
            SIV_U_true_22 = ds[f'SIV_u_true_day{day_inx}'][s22:e22]
            SIV_U_true_23 = ds[f'SIV_u_true_day{day_inx}'][s23:e23]
            SIV_U_true_all = np.concatenate([SIV_U_true_21, SIV_U_true_22, SIV_U_true_23], axis=0)

            SIV_V_pred_21 = ds[f'SIV_v_pred_day{day_inx}'][s21:e21]
            SIV_V_pred_22 = ds[f'SIV_v_pred_day{day_inx}'][s22:e22]
            SIV_V_pred_23 = ds[f'SIV_v_pred_day{day_inx}'][s23:e23]
            SIV_V_pred_all = np.concatenate([SIV_V_pred_21, SIV_V_pred_22, SIV_V_pred_22,], axis=0)

            SIV_V_true_21 = ds[f'SIV_v_true_day{day_inx}'][s21:e21]
            SIV_V_true_22 = ds[f'SIV_v_true_day{day_inx}'][s22:e22]
            SIV_V_true_23 = ds[f'SIV_v_true_day{day_inx}'][s23:e23]
            SIV_V_true_all = np.concatenate([SIV_V_true_21, SIV_V_true_22, SIV_V_true_22,], axis=0)

            SIC_pred = inverse_normalize(SIC_pred_all, sic_min, sic_max)
            SIC_true = inverse_normalize(SIC_true_all, sic_min, sic_max)
            SIV_U_pred = inverse_normalize(SIV_U_pred_all, siv_u_min, siv_u_max)
            SIV_U_true = inverse_normalize(SIV_U_true_all, siv_u_min, siv_u_max)
            SIV_V_pred = inverse_normalize(SIV_V_pred_all, siv_v_min, siv_v_max)
            SIV_V_true = inverse_normalize(SIV_V_true_all, siv_v_min, siv_v_max)

            SIC_CORR, SIC_RMSE, SIC_MAE = calculate_metrics(SIC_pred, SIC_true)
            SIV_U_CORR, SIV_U_RMSE, SIV_U_MAE = calculate_metrics(SIV_U_pred, SIV_U_true)
            SIV_V_CORR, SIV_V_RMSE, SIV_V_MAE = calculate_metrics(SIV_V_pred, SIV_V_true)
            
            SIC_MAE_day.append(round(float(SIC_MAE),4))
            SIC_RMSE_day.append(round(float(SIC_RMSE),4))
            SIC_CORR_day.append(round(float(SIC_CORR),4))

            SIV_U_MAE_day.append(round(float(SIV_U_MAE),4))
            SIV_U_RMSE_day.append(round(float(SIV_U_RMSE),4))
            SIV_U_CORR_day.append(round(float(SIV_U_CORR),4))
            
            SIV_V_MAE_day.append(round(float(SIV_V_MAE),4))
            SIV_V_RMSE_day.append(round(float(SIV_V_RMSE),4))
            SIV_V_CORR_day.append(round(float(SIV_V_CORR),4))

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

    SIC_MAE_array = np.array(SIC_MAE_list)
    SIC_RMSE_array = np.array(SIC_RMSE_list)
    SIC_CORR_array = np.array(SIC_CORR_list)

    SIV_U_MAE_array = np.array(SIV_U_MAE_list)
    SIV_U_RMSE_array = np.array(SIV_U_RMSE_list)
    SIV_U_CORR_array = np.array(SIV_U_CORR_list) 

    SIV_V_MAE_array = np.array(SIV_V_MAE_list)
    SIV_V_RMSE_array = np.array(SIV_V_RMSE_list) 
    SIV_V_CORR_array = np.array(SIV_V_CORR_list) 

    day_labels = [f'{i}' for i in range(1, 8)] 
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plot_sic_heatmap(SIC_MAE_array, 
                    title='Average MAE', 
                    cmap=purple_white_green, 
                    vmin=np.min(SIC_MAE_array), 
                    vmax=np.max(SIC_MAE_array))

    plot_sic_heatmap(SIC_RMSE_array, 
                    title='Average RMSE', 
                    cmap=purple_white_green,
                    vmin=np.min(SIC_RMSE_array), 
                    vmax=np.max(SIC_RMSE_array))

    # plot_sic_heatmap(SIC_CORR_array, 
    #                 title='Average CORR', 
    #                 #  cmap='YlOrRd',    # 黄-橙-红，越大越红
    #                 cmap = yl_or_rd_pu_black,
    #                 vmin=0.96,  # 相关系数通常≥0
    #                 vmax=1,
    #                 fmt='.4f')

    plot_siv_heatmap(SIV_U_MAE_array, title = 'Average MAE', var_name = 'SIV_u', cmap = purple_white_green)
    plot_siv_heatmap(SIV_U_RMSE_array, title = 'Average RMSE', var_name = 'SIV_u', cmap = purple_white_green)
    # plot_siv_heatmap(SIV_U_CORR_array, title = 'Average CORR', var_name = 'SIV_u', cmap = yl_or_rd_pu_black, vmin = 0.90, vmax = 1.00)

    plot_siv_heatmap(SIV_V_MAE_array, title = 'Average MAE', var_name = 'SIV_v', cmap = purple_white_green)
    plot_siv_heatmap(SIV_V_RMSE_array, title = 'Average RMSE', var_name = 'SIV_v', cmap = purple_white_green)
    # plot_siv_heatmap(SIV_V_CORR_array, title = 'Average CORR', var_name = 'SIV_v', cmap = yl_or_rd_pu_black, vmin = 0.90, vmax = 1.00)


    
