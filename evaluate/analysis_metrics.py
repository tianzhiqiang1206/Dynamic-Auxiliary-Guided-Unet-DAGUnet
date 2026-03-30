'''
This code is used to calculate the MAE, RMSE, and CORR metrics, and to compute their averages for different lead times.
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import os

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def calculate_metrics(pred, true):
    numerator = np.sum((pred - np.mean(pred)) * (true - np.mean(true)))
    denominator = np.sqrt(np.sum((pred - np.mean(pred))**2) * np.sum((true - np.mean(true))**2))
    R = numerator / denominator if denominator != 0 else 0
    RMSE = np.sqrt(np.mean((pred - true)**2))
    MAE = np.mean(np.abs(pred - true))
    return R, RMSE, MAE

model_names = ['DAGUnet'] # 'HISUnet', 'Unet_sic','Unet_siv'
year_indexs = [[2000,2023]] # '2000,2007', '2005,2012', '2012,2019', '2013,2020', '2000,2023'

all_metrics_data = []
SIC_FLAG = True
SIV_FLAG = True
num_days = 7

for model_name in model_names:
    all_metrics_data = []
    for year_index in year_indexs:
        start_year = year_index[0]
        end_year = year_index[1]
        time_period = f'{start_year}-{end_year}'
        print(f"--- {model_name} --- processing period: {time_period} ---")

        ds = xr.open_dataset(rf'E:\DAGUnet_code\newfolder\predict_{model_name}_pre7_{start_year}_{end_year}.nc')

        SIC_MAE_list = []
        SIC_RMSE_list = []
        SIC_CORR_list = []

        SIV_u_MAE_list = []
        SIV_u_RMSE_list = []
        SIV_u_CORR_list = []

        SIV_v_MAE_list = []
        SIV_v_RMSE_list = []
        SIV_v_CORR_list = []

        for day_idx in range(num_days):
            day_number = day_idx + 1
            daily_metrics = {
                'Time_Period': time_period,
                'Day_Index': day_number
            }

            if SIC_FLAG:
                SIC_pred = ds[f'SIC_pred_day{day_idx+1}'].values
                SIC_true = ds[f'SIC_true_day{day_idx+1}'].values
                SIC_pred = denormalize(SIC_pred, 0, 100)
                SIC_true = denormalize(SIC_true, 0, 100)
                SIC_R, SIC_RMSE, SIC_MAE = calculate_metrics(SIC_pred, SIC_true)
                daily_metrics['SIC_MAE'] = round(float(SIC_MAE), 6)
                daily_metrics['SIC_RMSE'] = round(float(SIC_RMSE), 6)
                daily_metrics['SIC_CORR'] = round(float(SIC_R), 6)
                SIC_MAE_list.append(SIC_MAE)
                SIC_RMSE_list.append(SIC_RMSE)
                SIC_CORR_list.append(SIC_R)

            if SIV_FLAG:
                SIV_u_pred = ds[f'SIV_u_pred_day{day_idx+1}'].values
                SIV_u_true = ds[f'SIV_u_true_day{day_idx+1}'].values
                SIV_v_pred = ds[f'SIV_v_pred_day{day_idx+1}'].values
                SIV_v_true = ds[f'SIV_v_true_day{day_idx+1}'].values
                SIV_u_pred = denormalize(SIV_u_pred, -60, 54) * 0.864
                SIV_u_true = denormalize(SIV_u_true, -60, 54) * 0.864
                SIV_v_pred = denormalize(SIV_v_pred, -58, 54) * 0.864
                SIV_v_true = denormalize(SIV_v_true, -58, 54) * 0.864
                SIV_u_R, SIV_u_RMSE, SIV_u_MAE = calculate_metrics(SIV_u_pred, SIV_u_true)
                daily_metrics['SIV_U_MAE'] = round(float(SIV_u_MAE), 6)
                daily_metrics['SIV_U_RMSE'] = round(float(SIV_u_RMSE), 6)
                daily_metrics['SIV_U_CORR'] = round(float(SIV_u_R), 6)
                SIV_u_MAE_list.append(SIV_u_MAE)
                SIV_u_RMSE_list.append(SIV_u_RMSE)
                SIV_u_CORR_list.append(SIV_u_R)

                SIV_v_R, SIV_v_RMSE, SIV_v_MAE = calculate_metrics(SIV_v_pred, SIV_v_true)
                daily_metrics['SIV_V_MAE'] = round(float(SIV_v_MAE), 6)
                daily_metrics['SIV_V_RMSE'] = round(float(SIV_v_RMSE), 6)
                daily_metrics['SIV_V_CORR'] = round(float(SIV_v_R), 6)
                SIV_v_MAE_list.append(SIV_v_MAE)
                SIV_v_RMSE_list.append(SIV_v_RMSE)
                SIV_v_CORR_list.append(SIV_v_R)

            all_metrics_data.append(daily_metrics)

        if SIC_FLAG:
            print("=================== SIC =======================")
            MAE_list = SIC_MAE_list
            RMSE_list = SIC_RMSE_list
            CORR_list = SIC_CORR_list
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR: {formatted_results}。")
            print('Mean MAE:',sum(MAE_list)/len(MAE_list))
            print('Mean RMSE:',sum(RMSE_list)/len(RMSE_list))
            print('Mean CORR:',sum(CORR_list)/len(CORR_list))

        if SIV_FLAG:
            print("=================== U =======================")
            MAE_list = SIV_u_MAE_list
            RMSE_list = SIV_u_RMSE_list
            CORR_list = SIV_u_CORR_list
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR: {formatted_results}。")
            print('Mean MAE:',sum(MAE_list)/len(MAE_list))
            print('Mean RMSE:',sum(RMSE_list)/len(RMSE_list))
            print('Mean CORR:',sum(CORR_list)/len(CORR_list))

            print("=================== V =======================")
            MAE_list = SIV_v_MAE_list
            RMSE_list = SIV_v_RMSE_list
            CORR_list = SIV_v_CORR_list
            formatted_results = [round(float(num), 6) for num in MAE_list]
            print(f"MAE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in RMSE_list]
            print(f"RMSE: {formatted_results}。")
            formatted_results = [round(float(num), 6) for num in CORR_list]
            print(f"CORR: {formatted_results}。")
            print('Mean MAE:',sum(MAE_list)/len(MAE_list))
            print('Mean RMSE:',sum(RMSE_list)/len(RMSE_list))
            print('Mean CORR:',sum(CORR_list)/len(CORR_list))
