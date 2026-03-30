'''
This code is used to reproduce Fig. 4 and calculate the IIEE/UE/OE in Fig.4
'''
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.lines import Line2D 
import pandas as pd
from datetime import datetime

years = [[2000,2007], [2005,2012],[2012,2019],[2013,2020]]
for year in years:
    target_year = year[1]
    file_path = rf"F:\DAGUnet_code\newresult\predict_Unet_sic_pre7_{year[0]}_{year[1]}.nc"
    file_path1 = rf"F:\DAGUnet_code\newresult\predict_HISUnet_pre7_{year[0]}_{year[1]}.nc"
    file_path2 = rf"F:\DAGUnet_code\newresult\predict_DAGUnet_pre7_{year[0]}_{year[1]}.nc"
    ds = xr.open_dataset(file_path)
    ds1 = xr.open_dataset(file_path1)
    ds2 = xr.open_dataset(file_path2)

    def inverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    def calculate_iiee(sic_pred, sic_true, grid_area, boundary):
        pred_mask = (sic_pred > boundary).astype(np.float32)
        true_mask = (sic_true > boundary).astype(np.float32)
        oe_mask = np.logical_and(pred_mask == 1, true_mask == 0)
        ue_mask = np.logical_and(pred_mask == 0, true_mask == 1)
        oe = np.sum(oe_mask * grid_area)
        ue = np.sum(ue_mask * grid_area)
        iiee = oe + ue
        return iiee, oe, ue

    sic_min = 0
    sic_max = 100
    projection = ccrs.LambertAzimuthalEqualArea(central_latitude=90, central_longitude=0)
    original_size = 361
    new_size = 256
    start_idx = (original_size - new_size) // 2
    end_idx = start_idx + new_size
    x_min = -4524688.2625
    x_max = 4524688.2625
    y_min = -4524688.2625
    y_max = 4524688.2625
    x = np.linspace(x_min, x_max, original_size)[start_idx:end_idx]
    y = np.linspace(y_min, y_max, original_size)[start_idx:end_idx]
    x_grid, y_grid = np.meshgrid(x, y)
    boundary_value = 15

    start_date = datetime(year[1], 1, 8)
    end_date = datetime(year[1], 12, 24)

    if year[1] % 4 == 0:
        month_days = [24,29,31,30,31,30,31,31,30,31,30,24]
    else:
        month_days = [24,28,31,30,31,30,31,31,30,31,30,24]

    cumulative_index = [0] 
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)

    month_indexs = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs.append([start_idx,end_idx])

    print(month_indexs)

    for month_num in range(7, 10):
        results = []
        for day_inx in range(1, 8):
            SIC_pred_Unet = inverse_normalize(ds[f'SIC_pred_day{day_inx}'][month_indexs[month_num-1][0]:month_indexs[month_num-1][1],:,:], sic_min, sic_max)
            SIC_true = inverse_normalize(ds[f'SIC_true_day{day_inx}'][month_indexs[month_num-1][0]:month_indexs[month_num-1][1],:,:], sic_min, sic_max)
            SIC_pred_HISUnet = inverse_normalize(ds1[f'SIC_pred_day{day_inx}'][month_indexs[month_num-1][0]:month_indexs[month_num-1][1],:,:], sic_min, sic_max)
            SIC_pred_DAGUnet = inverse_normalize(ds2[f'SIC_pred_day{day_inx}'][month_indexs[month_num-1][0]:month_indexs[month_num-1][1],:,:], sic_min, sic_max)

            sic_pred_mean_Unet = SIC_pred_Unet
            sic_true_mean = SIC_true
            sic_pred_mean_HISUnet = SIC_pred_HISUnet
            sic_pred_mean_DAGUnet = SIC_pred_DAGUnet

            sic_pred_mean_Unet = SIC_pred_Unet.mean(dim='sample')
            sic_true_mean = SIC_true.mean(dim='sample')
            sic_pred_mean_HISUnet = SIC_pred_HISUnet.mean(dim='sample')
            sic_pred_mean_DAGUnet = SIC_pred_DAGUnet.mean(dim='sample')

            iiee_unet, oe_unet, ue_unet = calculate_iiee(sic_pred_mean_Unet, sic_true_mean, 25*25, boundary_value)
            iiee_hisunet, oe_hisunet, ue_hisunet = calculate_iiee(sic_pred_mean_HISUnet, sic_true_mean, 25*25, boundary_value)
            iiee_dagunet, oe_dagunet, ue_dagunet = calculate_iiee(sic_pred_mean_DAGUnet, sic_true_mean, 25*25, boundary_value)

            results.append({
                'Day': day_inx,
                'Model': 'Unet',
                'IIEE (km²)': round(float(iiee_unet), 2),
                'OE (km²)': round(float(oe_unet), 2),
                'UE (km²)': round(float(ue_unet), 2)
            })
            results.append({
                'Day': day_inx,
                'Model': 'HISUnet',
                'IIEE (km²)': round(float(iiee_hisunet), 2),
                'OE (km²)': round(float(oe_hisunet), 2),
                'UE (km²)': round(float(ue_hisunet), 2)
            })
            results.append({
                'Day': day_inx,
                'Model': 'DAGUnet',
                'IIEE (km²)': round(float(iiee_dagunet), 2),
                'OE (km²)': round(float(oe_dagunet), 2),
                'UE (km²)': round(float(ue_dagunet), 2)
            })

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})

            ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='darkgray', alpha=0.7)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
            ax.gridlines(draw_labels=False, alpha=0.3)

            true_contour = ax.contour(x_grid, y_grid, sic_true_mean, levels=[boundary_value], 
                                    transform=projection, colors='k', linewidths=1, 
                                    linestyles='--')
            pred_contour_Unet = ax.contour(x_grid, y_grid, sic_pred_mean_Unet, levels=[boundary_value], 
                                    transform=projection, colors='red', linewidths=1, 
                                    linestyles='-')
            pred_contour_HISUnet = ax.contour(x_grid, y_grid, sic_pred_mean_HISUnet, levels=[boundary_value], 
                                    transform=projection, colors='green', linewidths=1, 
                                    linestyles='-')
            pred_contour_DAGUnet = ax.contour(x_grid, y_grid, sic_pred_mean_DAGUnet, levels=[boundary_value], 
                                    transform=projection, colors='blue', linewidths=1, 
                                    linestyles='-')

            # 图例
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Real'),
                Line2D([0], [0], color='red', linestyle='-', linewidth=1.5, label='Unet'),
                Line2D([0], [0], color='green', linestyle='-', linewidth=1.5, label='HISUnet'),
                Line2D([0], [0], color='blue', linestyle='-', linewidth=1.5, label='DAGUnet')
            ]
            ax.legend(handles=legend_elements,
                    loc='lower right',
                    fontsize=14,
                    frameon=True,
                    facecolor='white',
                    edgecolor='gray',

                    title_fontsize=11)

            plt.tight_layout()
            plt.savefig(rf'figure\{target_year}_{month_num}_SIE_Compare_day{day_inx}.png', dpi=600, bbox_inches='tight')
            plt.close()

            print(f"Day {day_inx} IIEE value (km²):")
            print(f"Unet:    IIEE = {iiee_unet:.2f}, OE = {oe_unet:.2f}, UE = {ue_unet:.2f}")
            print(f"HISUnet: IIEE = {iiee_hisunet:.2f}, OE = {oe_hisunet:.2f}, UE = {ue_hisunet:.2f}")
            print(f"DAGUnet: IIEE = {iiee_dagunet:.2f}, OE = {oe_dagunet:.2f}, UE = {ue_dagunet:.2f}")

        df = pd.DataFrame(results)
        output_path = rf'figure\{target_year}_{month_num}_IIEE_day1_to_day7.xlsx'
        df.to_excel(output_path, index=False) 

        print(f"\nMetrics are save in {output_path}")
