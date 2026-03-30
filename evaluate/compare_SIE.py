import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.lines import Line2D  # 手动创建图例所需
import pandas as pd  # 用于Excel保存
from datetime import datetime

years = [[2000,2007], [2005,2012],[2012,2019],[2013,2020]]  #, [2000,2023],[2005,2012],[2012,2019],[2013,2020]
for year in years:
    target_year = year[1]
    # 读取数据
    file_path = rf"F:\DAGUnet_code\newresult\predict_Unet_sic_pre7_{year[0]}_{year[1]}.nc"
    file_path1 = rf"F:\DAGUnet_code\newresult\predict_HISUnet_pre7_{year[0]}_{year[1]}.nc"
    file_path2 = rf"F:\DAGUnet_code\newresult\predict_DAGUnet_pre7_{year[0]}_{year[1]}.nc"
    ds = xr.open_dataset(file_path)
    ds1 = xr.open_dataset(file_path1)
    ds2 = xr.open_dataset(file_path2)

    # 定义逆归一化函数
    def inverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    def calculate_iiee(sic_pred, sic_true, grid_area, boundary):
        """计算IIEE、OE、UE"""
        pred_mask = (sic_pred > boundary).astype(np.float32)
        true_mask = (sic_true > boundary).astype(np.float32)
        oe_mask = np.logical_and(pred_mask == 1, true_mask == 0)
        ue_mask = np.logical_and(pred_mask == 0, true_mask == 1)
        oe = np.sum(oe_mask * grid_area)
        ue = np.sum(ue_mask * grid_area)
        iiee = oe + ue
        return iiee, oe, ue

    # 定义参数
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

    # 定义月份日期范围
    start_date = datetime(year[1], 1, 8)
    end_date = datetime(year[1], 12, 24)

    if year[1] % 4 == 0:
        month_days = [24,29,31,30,31,30,31,31,30,31,30,24]
    else:
        month_days = [24,28,31,30,31,30,31,31,30,31,30,24]

    # 3. 计算累计索引（每个月的起始索引）
    cumulative_index = [0]  # 初始索引为0
    for days in month_days[:-1]:
        cumulative_index.append(cumulative_index[-1] + days)

    # 4. 生成月份索引字典（包含起始/结束索引）
    month_indexs = []
    for month in range(1, 13):
        start_idx = cumulative_index[month-1]
        end_idx = cumulative_index[month-1] + month_days[month-1]
        month_indexs.append([start_idx,end_idx])

    print(month_indexs)

    for month_num in range(7, 10):
        # 存储所有结果的列表（用于生成Excel）
        results = []
        for day_inx in range(1, 8):
            # 数据处理
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

            # 计算IIEE（注意：这里保留你原代码中的25*25网格面积，若需精确可替换为calculate_grid_area函数）
            iiee_unet, oe_unet, ue_unet = calculate_iiee(sic_pred_mean_Unet, sic_true_mean, 25*25, boundary_value)
            iiee_hisunet, oe_hisunet, ue_hisunet = calculate_iiee(sic_pred_mean_HISUnet, sic_true_mean, 25*25, boundary_value)
            iiee_dagunet, oe_dagunet, ue_dagunet = calculate_iiee(sic_pred_mean_DAGUnet, sic_true_mean, 25*25, boundary_value)

            # 存储结果到列表（每条记录对应一个模型+一天的数据）
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

            # 绘图部分（保持不变）
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': projection})
            # plt.title(f'SIC Boundary Comparison (Value = {boundary_value}) - Prediction Day {day_inx}', fontsize=14, pad=20)

            ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='darkgray', alpha=0.7)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
            ax.gridlines(draw_labels=False, alpha=0.3)

            # 绘制边界线
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
                    # title='Boundary Legend',
                    title_fontsize=11)

            plt.tight_layout()
            # plt.savefig(rf'F:\DAGUnet_code\newresult\SIE_visualization\{target_year}_{month_num}_SIE_Compare_day{day_inx}.png', dpi=600, bbox_inches='tight')
            # plt.savefig(f'./SIE_visualization/SIE_Compare_20120916.png', dpi=600, bbox_inches='tight')
            plt.close()

            # 打印结果
            print(f"Day {day_inx} IIEE value (km²):")
            print(f"Unet:    IIEE = {iiee_unet:.2f}, 高估面积 = {oe_unet:.2f}, 低估面积 = {ue_unet:.2f}")
            print(f"HISUnet: IIEE = {iiee_hisunet:.2f}, 高估面积 = {oe_hisunet:.2f}, 低估面积 = {ue_hisunet:.2f}")
            print(f"DAGUnet: IIEE = {iiee_dagunet:.2f}, 高估面积 = {oe_dagunet:.2f}, 低估面积 = {ue_dagunet:.2f}")

        df = pd.DataFrame(results)  # 转换为DataFrame
        output_path = rf'F:\DAGUnet_code\newresult\SIE_visualization\{target_year}_{month_num}_IIEE_day1_to_day7.xlsx'  # 保存路径
        df.to_excel(output_path, index=False)  # 保存为Excel，不包含行索引

        print(f"\n所有指标已保存至Excel文件：{output_path}")
