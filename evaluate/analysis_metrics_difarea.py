'''
This code is used to calculate metrics for the predictions of different models across various sea ice regions.
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import os
from read_data import crop_center

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def calculate_metrics(pred, true, mask=None):
    if mask is not None:
        mask_3d = np.expand_dims(mask, axis=0)
        mask_3d = np.repeat(mask_3d, pred.shape[0], axis=0)
        
        mask_bool = mask_3d.astype(bool)
        pred_flat = pred.flatten()[mask_bool.flatten()]
        true_flat = true.flatten()[mask_bool.flatten()]
    else:
        pred_flat = pred.flatten()
        true_flat = true.flatten()
    
    # 处理空数据情况
    if len(pred_flat) == 0 or len(true_flat) == 0:
        return 0, 0, 0

    numerator = np.sum((pred_flat - np.mean(pred_flat)) * (true_flat - np.mean(true_flat)))
    denominator = np.sqrt(np.sum((pred_flat - np.mean(pred_flat))**2) * np.sum((true_flat - np.mean(true_flat))** 2))
    R = numerator / denominator if denominator != 0 else 0
    RMSE = np.sqrt(np.mean((pred_flat - true_flat)**2))
    MAE = np.mean(np.abs(pred_flat - true_flat))
    
    return R, RMSE, MAE

def save_metrics_to_excel(all_metrics_data, save_path):
    metrics_records = []
    
    for model_data in all_metrics_data:
        for period_data in model_data:
            model_name = period_data['model']
            time_period = period_data['time_period']
            
            if 'sic' in period_data:
                sic_metrics = period_data['sic']
                for area, metrics in sic_metrics.items():
                    mae_mean = round(np.mean(metrics['MAE']), 4)
                    rmse_mean = round(np.mean(metrics['RMSE']), 4)
                    corr_mean = round(np.mean(metrics['CORR']), 4)
                    
                    metrics_records.append({
                        'model': model_name,
                        'period': time_period,
                        'area': area,
                        'variable': 'SIC',
                        'MAE': mae_mean,
                        'RMSE': rmse_mean,
                        'CORR': corr_mean,
                        'MAE': str(metrics['MAE']),
                        'RMSE': str(metrics['RMSE']),
                        'CORR': str(metrics['CORR'])
                    })
            
            if 'siv_u' in period_data:
                siv_u_metrics = period_data['siv_u']
                for area, metrics in siv_u_metrics.items():
                    mae_mean = round(np.mean(metrics['MAE']), 4)
                    rmse_mean = round(np.mean(metrics['RMSE']), 4)
                    corr_mean = round(np.mean(metrics['CORR']), 4)
                    
                    metrics_records.append({
                        'model': model_name,
                        'period': time_period,
                        'area': area,
                        'variable': 'SIV_U',
                        'MAE': mae_mean,
                        'RMSE': rmse_mean,
                        'CORR': corr_mean,
                        'MAE': str(metrics['MAE']),
                        'RMSE': str(metrics['RMSE']),
                        'CORR': str(metrics['CORR'])
                    })

            if 'siv_v' in period_data:
                siv_v_metrics = period_data['siv_v']
                for area, metrics in siv_v_metrics.items():
                    mae_mean = round(np.mean(metrics['MAE']), 4)
                    rmse_mean = round(np.mean(metrics['RMSE']), 4)
                    corr_mean = round(np.mean(metrics['CORR']), 4)
                    
                    metrics_records.append({
                        'model': model_name,
                        'period': time_period,
                        'area': area,
                        'variable': 'SIV_V',
                        'MAE': mae_mean,
                        'RMSE': rmse_mean,
                        'CORR': corr_mean,
                        'MAE': str(metrics['MAE']),
                        'RMSE': str(metrics['RMSE']),
                        'CORR': str(metrics['CORR'])
                    })
    
    df = pd.DataFrame(metrics_records)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='metric sum', index=False)
        for metric_type in ['SIC', 'SIV_U', 'SIV_V']:
            df_sub = df[df['variable'] == metric_type]
            df_sub.to_excel(writer, sheet_name=f'{metric_type}指标', index=False)
    
    print(f"✅ The metrics are saved in {save_path}")
    return df

model_names = ['Unet_siv']  # 'Unet_sic', 'Unet_siv', 'HISUnet', 'DAGUnet'
year_indexs = [[2000,2007], [2005,2012], [2012,2019], [2013,2020], [2000,2023]] # 
area_indexs = ['Central Arctic','Beaufort Sea','Chukchi Sea','East Siberian Sea',
               'Laptev Sea','Kara Sea','Barents Sea','East Greenland Sea',
               'baffin and Labrador Seas','Canadian Archipelago']
area_mask_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

all_metrics_data = []
SIC_FLAG = False
SIV_FLAG = True
num_days = 7

ds_mask = xr.open_dataset(rf'data/mask_reprojected.nc')
arctic_mask = crop_center(ds_mask['arctic_mask'].values)  # shape[256, 256]

area_masks = {
    area: (arctic_mask == num) for area, num in zip(area_indexs, area_mask_num)
}
area_masks['Whole Arctic'] = (arctic_mask != 0)

for model_name in model_names:
    model_metrics = [] 
    for year_index in year_indexs:
        start_year = year_index[0]
        end_year = year_index[1]
        time_period = f'{start_year}-{end_year}'
        print(f"--- processing: {model_name}，period: {time_period} ---")
        
        ds = xr.open_dataset(rf'data/predict_{model_name}_pre7_{start_year}_{end_year}.nc')

        sic_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}
        siv_u_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}
        siv_v_metrics = {area: {'MAE': [], 'RMSE': [], 'CORR': []} for area in area_masks.keys()}

        for day_idx in range(num_days):
            day_number = day_idx + 1
            print(f"---- processing No.{day_number} day ----")

            if SIC_FLAG:
                SIC_pred = ds[f'SIC_pred_day{day_idx+1}'].values
                SIC_true = ds[f'SIC_true_day{day_idx+1}'].values
                SIC_pred = denormalize(SIC_pred, 0, 100)
                SIC_true = denormalize(SIC_true, 0, 100)

                for area, mask in area_masks.items():
                    r, rmse, mae = calculate_metrics(SIC_pred, SIC_true, mask)
                    sic_metrics[area]['MAE'].append(round(mae, 4))
                    sic_metrics[area]['RMSE'].append(round(rmse, 4))
                    sic_metrics[area]['CORR'].append(round(r, 4))

            if SIV_FLAG:
                SIV_u_pred = ds[f'SIV_u_pred_day{day_idx+1}'].values
                SIV_u_true = ds[f'SIV_u_true_day{day_idx+1}'].values
                SIV_v_pred = ds[f'SIV_v_pred_day{day_idx+1}'].values
                SIV_v_true = ds[f'SIV_v_true_day{day_idx+1}'].values
                
                SIV_u_pred = denormalize(SIV_u_pred, -60, 54) * 0.864
                SIV_u_true = denormalize(SIV_u_true, -60, 54) * 0.864
                SIV_v_pred = denormalize(SIV_v_pred, -58, 54) * 0.864
                SIV_v_true = denormalize(SIV_v_true, -58, 54) * 0.864

                for area, mask in area_masks.items():
                    r, rmse, mae = calculate_metrics(SIV_u_pred, SIV_u_true, mask)
                    siv_u_metrics[area]['MAE'].append(round(mae, 4))
                    siv_u_metrics[area]['RMSE'].append(round(rmse, 4))
                    siv_u_metrics[area]['CORR'].append(round(r, 4))

                for area, mask in area_masks.items():
                    r, rmse, mae = calculate_metrics(SIV_v_pred, SIV_v_true, mask)
                    siv_v_metrics[area]['MAE'].append(round(mae, 4))
                    siv_v_metrics[area]['RMSE'].append(round(rmse, 4))
                    siv_v_metrics[area]['CORR'].append(round(r, 4))

        if SIC_FLAG:
            print("\n=================== SIC =======================")
            for area in area_masks.keys():
                print(f"\n----- {area} -----")
                print(f"MAE: {sic_metrics[area]['MAE']}，average: {round(np.mean(sic_metrics[area]['MAE']), 4)}")
                print(f"RMSE: {sic_metrics[area]['RMSE']}, average: {round(np.mean(sic_metrics[area]['RMSE']), 4)}")
                print(f"CORR: {sic_metrics[area]['CORR']}，average: {round(np.mean(sic_metrics[area]['CORR']), 4)}")

        if SIV_FLAG:
            print("\n=================== SIV U =======================")
            for area in area_masks.keys():
                print(f"\n----- {area} -----")
                print(f"MAE: {siv_u_metrics[area]['MAE']}，average: {round(np.mean(siv_u_metrics[area]['MAE']), 4)}")
                print(f"RMSE: {siv_u_metrics[area]['RMSE']}，average: {round(np.mean(siv_u_metrics[area]['RMSE']), 4)}")
                print(f"CORR: {siv_u_metrics[area]['CORR']}，average: {round(np.mean(siv_u_metrics[area]['CORR']), 4)}")

            print("\n=================== SIV V =======================")
            for area in area_masks.keys():
                print(f"\n----- {area} -----")
                print(f"MAE: {siv_v_metrics[area]['MAE']}，avarage: {round(np.mean(siv_v_metrics[area]['MAE']), 4)}")
                print(f"RMSE: {siv_v_metrics[area]['RMSE']}，average: {round(np.mean(siv_v_metrics[area]['RMSE']), 4)}")
                print(f"CORR: {siv_v_metrics[area]['CORR']}，avarage: {round(np.mean(siv_v_metrics[area]['CORR']), 4)}")

        model_metrics.append({
            'model': model_name,
            'time_period': time_period,
            'sic': sic_metrics,
            'siv_u': siv_u_metrics,
            'siv_v': siv_v_metrics
        })
    
    all_metrics_data.append(model_metrics)

    save_path = rf'metrics_diff_area_{model_name}.xlsx'
    save_metrics_to_excel(all_metrics_data, save_path)
    print('done')
