import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import os

def denormalize(data, min_val, max_val):
    """将归一化后的数据反归一化回原始范围"""
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

# , 'Unet_siv','MCPN_istl1','MCPN_istl2','MCPN_istl3',
# 'MCPN_istl4','MCPN_sit', 'MCPN_v10', 'MCPN_u10', 'MCPN_v100', 'MCPN_u100'

model_names = ['DAGUnet']
# year_indexs = [[2000,2007],[2005,2012],[2012,2019],[2013,2020],[2000,2023]]
year_indexs = [[2000,2023]]

all_metrics_data = []
SIC_FLAG = True
SIT_FLAG = False
SIV_FLAG = True
DRAW_FLAG = False
num_days = 7

for model_name in model_names:
    all_metrics_data = []
    for year_index in year_indexs:
        start_year = year_index[0]
        end_year = year_index[1]
        time_period = f'{start_year}-{end_year}'
        print(f"--- {model_name} --- 正在处理时间段: {time_period} ---")

        # 读取保存的预测结果
        ds = xr.open_dataset(rf'E:\DAGUnet_code\newfolder\predict_{model_name}_pre7_{start_year}_{end_year}.nc')

        SIC_MAE_list = []   # 保存 SIC 数据的 MAE 指标
        SIC_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
        SIC_CORR_list = []  # 保存 SIC 数据的相关系数指标

        SIT_MAE_list = []   # 保存 SIC 数据的 MAE 指标
        SIT_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
        SIT_CORR_list = []  # 保存 SIC 数据的相关系数指标

        SIV_u_MAE_list = []   # 保存 SIC 数据的 MAE 指标
        SIV_u_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
        SIV_u_CORR_list = []  # 保存 SIC 数据的相关系数指标

        SIV_v_MAE_list = []   # 保存 SIC 数据的 MAE 指标
        SIV_v_RMSE_list = []  # 保存 SIC 数据的 RMSE 指标
        SIV_v_CORR_list = []  # 保存 SIC 数据的相关系数指标

        for day_idx in range(num_days):
            day_number = day_idx + 1
            # 用于存储当前 (Time_Period, Day_Number) 的所有指标
            daily_metrics = {
                'Time_Period': time_period,
                'Day_Index': day_number
            }

            if SIC_FLAG:
                SIC_pred = ds[f'SIC_pred_day{day_idx+1}'].values
                SIC_true = ds[f'SIC_true_day{day_idx+1}'].values
                SIC_pred = denormalize(SIC_pred, 0, 100)
                SIC_true = denormalize(SIC_true, 0, 100)
                # 计算SIC的评估指标
                SIC_R, SIC_RMSE, SIC_MAE = calculate_metrics(SIC_pred, SIC_true)
                # 存储 SIC 每日指标到 daily_metrics 字典
                daily_metrics['SIC_MAE'] = round(float(SIC_MAE), 6)
                daily_metrics['SIC_RMSE'] = round(float(SIC_RMSE), 6)
                daily_metrics['SIC_CORR'] = round(float(SIC_R), 6)
                SIC_MAE_list.append(SIC_MAE)
                SIC_RMSE_list.append(SIC_RMSE)
                SIC_CORR_list.append(SIC_R)

            if SIT_FLAG:
                SIT_pred = ds[f'SIT_pred_day{day_idx+1}'].values
                SIT_true = ds[f'SIT_true_day{day_idx+1}'].values
                SIT_pred = denormalize(SIT_pred, 0, 10)
                SIT_true = denormalize(SIT_true, 0, 10)
                # 计算SIT的评估指标
                SIT_R, SIT_RMSE, SIT_MAE = calculate_metrics(SIT_pred, SIT_true)
                daily_metrics['SIT_MAE'] = round(float(SIT_MAE), 6)
                daily_metrics['SIT_RMSE'] = round(float(SIT_RMSE), 6)
                daily_metrics['SIT_CORR'] = round(float(SIT_R), 6)
                SIT_MAE_list.append(SIT_MAE)
                SIT_RMSE_list.append(SIT_RMSE)
                SIT_CORR_list.append(SIT_R)

            if SIV_FLAG:
                SIV_u_pred = ds[f'SIV_u_pred_day{day_idx+1}'].values
                SIV_u_true = ds[f'SIV_u_true_day{day_idx+1}'].values
                SIV_v_pred = ds[f'SIV_v_pred_day{day_idx+1}'].values
                SIV_v_true = ds[f'SIV_v_true_day{day_idx+1}'].values
                SIV_u_pred = denormalize(SIV_u_pred, -60, 54) * 0.864
                SIV_u_true = denormalize(SIV_u_true, -60, 54) * 0.864
                SIV_v_pred = denormalize(SIV_v_pred, -58, 54) * 0.864
                SIV_v_true = denormalize(SIV_v_true, -58, 54) * 0.864
                # 计算SIV的u分量评估指标
                SIV_u_R, SIV_u_RMSE, SIV_u_MAE = calculate_metrics(SIV_u_pred, SIV_u_true)
                daily_metrics['SIV_U_MAE'] = round(float(SIV_u_MAE), 6)
                daily_metrics['SIV_U_RMSE'] = round(float(SIV_u_RMSE), 6)
                daily_metrics['SIV_U_CORR'] = round(float(SIV_u_R), 6)
                SIV_u_MAE_list.append(SIV_u_MAE)
                SIV_u_RMSE_list.append(SIV_u_RMSE)
                SIV_u_CORR_list.append(SIV_u_R)

                # 计算SIV的v分量评估指标
                SIV_v_R, SIV_v_RMSE, SIV_v_MAE = calculate_metrics(SIV_v_pred, SIV_v_true)
                daily_metrics['SIV_V_MAE'] = round(float(SIV_v_MAE), 6)
                daily_metrics['SIV_V_RMSE'] = round(float(SIV_v_RMSE), 6)
                daily_metrics['SIV_V_CORR'] = round(float(SIV_v_R), 6)
                SIV_v_MAE_list.append(SIV_v_MAE)
                SIV_v_RMSE_list.append(SIV_v_RMSE)
                SIV_v_CORR_list.append(SIV_v_R)

            # 将当前天数的指标结果保存在总列表中
            all_metrics_data.append(daily_metrics)

        if SIC_FLAG:
            print("=================== SIC指标 =======================")
            MAE_list = SIC_MAE_list
            RMSE_list = SIC_RMSE_list
            CORR_list = SIC_CORR_list
            # 转换为普通浮点数并保留四位小数
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR的结果为: {formatted_results}。")
            print('MAE的平均值为',sum(MAE_list)/len(MAE_list))
            print('RMSE的平均值为',sum(RMSE_list)/len(RMSE_list))
            print('CORR的平均值为',sum(CORR_list)/len(CORR_list))

        if SIT_FLAG:
            print("=================== SIT指标 =======================")
            MAE_list = SIT_MAE_list
            RMSE_list = SIT_RMSE_list
            CORR_list = SIT_CORR_list
            # 转换为普通浮点数并保留四位小数
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR的结果为: {formatted_results}。")
            print('MAE的平均值为',sum(MAE_list)/len(MAE_list))
            print('RMSE的平均值为',sum(RMSE_list)/len(RMSE_list))
            print('CORR的平均值为',sum(CORR_list)/len(CORR_list))

        if SIV_FLAG:
            print("=================== U 指标 =======================")
            MAE_list = SIV_u_MAE_list
            RMSE_list = SIV_u_RMSE_list
            CORR_list = SIV_u_CORR_list
            # 转换为普通浮点数并保留四位小数
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR的结果为: {formatted_results}。")
            print('MAE的平均值为',sum(MAE_list)/len(MAE_list))
            print('RMSE的平均值为',sum(RMSE_list)/len(RMSE_list))
            print('CORR的平均值为',sum(CORR_list)/len(CORR_list))

            print("=================== V 指标 =======================")
            MAE_list = SIV_v_MAE_list
            RMSE_list = SIV_v_RMSE_list
            CORR_list = SIV_v_CORR_list
            # 转换为普通浮点数并保留四位小数
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE的结果为: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR的结果为: {formatted_results}。")
            print('MAE的平均值为',sum(MAE_list)/len(MAE_list))
            print('RMSE的平均值为',sum(RMSE_list)/len(RMSE_list))
            print('CORR的平均值为',sum(CORR_list)/len(CORR_list))
