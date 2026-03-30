import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SeaIcePredictionDataset(Dataset):
    def __init__(self, sst, sea_ice_concentration, u, v, sea_ice_thickness, 
                input_days, predict_days):
        # 输入数据
        self.input_data = {}
        self.sst = sst
        # self.input_data['sst'] = sst
        # self.input_data['t2m'] = t2m
        # self.input_data['u10'] = u10
        # self.input_data['v10'] = v10
        # self.input_data['z'] = z
        self.input_data['u'] = u
        self.input_data['v'] = v
        self.input_data['sea_ice_concentration'] = sea_ice_concentration
        self.input_data['sea_ice_thickness'] = sea_ice_thickness
        # 位置信息 (不随时间变化)
        # self.latitude = latitude[0]
        # self.longitude = longitude[0]
        # 序列长度和转换函数
        self.input_days = input_days
        self.predict_days = predict_days
        # 计算样本数 (考虑序列长度)
        self.num_samples = u.shape[0] - input_days - predict_days + 1
        # 生成二维坐标信息
        height, width = u.shape[1:]
        self.x_coords = np.arange(width) / width  # 归一化到 [0, 1]
        self.y_coords = np.arange(height) / height  # 归一化到 [0, 1]

        # 获取掩码信息
        self.mask = sst[0] != 0  # 假设sst中值为0的区域是掩码区域

    def __len__(self):
        """返回数据集的样本数量"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # print(idx + self.seq_length)
        # print(self.input_data['sst'].shape[0])
        if idx + self.input_days + self.predict_days >= self.input_data['sea_ice_concentration'].shape[0]:
            raise IndexError("索引超出范围")
        
        # 提取3天的输入数据并合并到通道维度
        input_channels = []
        # 提取9个变量，每个变量有3天的数据
        # for var_name in ['u', 'v', 'sea_ice_concentration', 'sea_ice_thickness']:
        for var_name in ['u','v','sea_ice_concentration']:
        # for var_name in ['sea_ice_concentration']:
            var_data = self.input_data[var_name]
            # 提取连续3天的数据并展平到通道维度
            for day in range(self.input_days):
                input_channels.append(var_data[idx + day])

        # 添加x轴和y轴坐标信息 (2个通道)
        # input_channels.append(np.tile(self.x_coords, (self.x_coords.size, 1)))
        # input_channels.append(np.tile(self.y_coords, (self.y_coords.size, 1)).T)
        # # 添加latitude和longitude (2个通道)
        # input_channels.append(self.latitude)
        # input_channels.append(self.longitude)
        # 转换为numpy数组
        # 形状: (26, 361, 361)
        input_data = np.stack(input_channels, axis=0)

        # 目标数据: [predict_days天的u, v, SIC, SIT]
        target_data = []
        for day in range(self.predict_days):
            target_day = idx + self.input_days + day
            target_data.extend([
                self.input_data['u'][target_day],
                self.input_data['v'][target_day],
                self.input_data['sea_ice_concentration'][target_day],
                self.input_data['sea_ice_thickness'][target_day],
                # self.input_data['u10'][target_day],
                # self.input_data['v10'][target_day]      
            ])
        target_data = np.stack(target_data, axis=0)

        mask = self.mask
        # 返回输入和目标数据
        return input_data, target_data, mask

def create_data_loaders_by_year(sst, sea_ice_concentration, u, v, sea_ice_thickness, 
                                input_days, predict_days, year_list, batch_size=8, shuffle=True):
    """
    按照年份划分训练、验证和测试数据加载器
    
    训练集: 2016-2020年
    验证集: 2021年
    测试集: 2022年
    """
    # 每年的天数 (2000-2023)
    days_per_year = [366, 365, 365, 365,
                     366, 365, 365, 365, 
                     366, 365, 365, 365,
                     366, 365, 365, 365,
                     366, 365, 365, 365,
                     366, 365, 365, 365]  # 2016是闰年
    
    # 计算每年的索引范围 (考虑序列长度为7天)
    year_indices = {}
    start_idx = 0
    for i, days in enumerate(range(2000, 2024)):
        end_idx = start_idx + days_per_year[i] - input_days - predict_days  # 减去序列长度
        year_indices[i] = (start_idx, end_idx)
        start_idx = end_idx + input_days + predict_days

    # 确定各数据集的年份范围
    train_start_year = year_list[0]
    train_end_year = year_list[1]
    train_years = list(range(train_start_year, train_end_year + 1))
    # print(train_years)
    val_start_year = year_list[2]
    val_end_year = year_list[3]
    val_years = list(range(val_start_year, val_end_year + 1))
    # print(val_years)
    test_start_year = year_list[4]
    test_end_year = year_list[5]
    test_years = list(range(test_start_year, test_end_year + 1))
    # print(test_years)

    # 计算各数据集的索引
    def get_indices_for_years(years):
        indices = []
        for year in years:
            year_idx = year - 2000 # train_years[0]
            start, end = year_indices[year_idx]
            indices.extend(range(start, end + 1))
        return list(range(indices[0], indices[-1]))
    
    train_indices = get_indices_for_years(train_years)
    val_indices = get_indices_for_years(val_years)
    test_indices = get_indices_for_years(test_years)

    # 打印索引范围
    print(f"训练集 ({len(train_indices)} samples): {train_years[0]}-{train_years[-1]}年")
    print(f"验证集 ({len(val_indices)} samples): {val_years[0]}-{val_years[-1]}年")
    print(f"测试集 ({len(test_indices)} samples): {test_years[0]}-{test_years[-1]}年")
    
    # 创建完整数据集
    dataset = SeaIcePredictionDataset(
        sst, sea_ice_concentration=sea_ice_concentration, 
        u=u, v=v, sea_ice_thickness = sea_ice_thickness,
        input_days=input_days, predict_days=predict_days,
    )
    
    # 使用Subset划分数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# 定义归一化函数
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def crop_center(data, crop_size=256):
    """从数据中心裁剪指定大小的区域"""
    if len(data.shape) == 3:  # 处理时间序列数据 [time, height, width]
        _, h, w = data.shape
    elif len(data.shape) == 2:  # 处理单张图像 [height, width]
        h, w = data.shape
    else:
        raise ValueError(f"数据形状 {data.shape} 不支持裁剪")
    
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    
    if len(data.shape) == 3:
        return data[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    else:
        return data[start_h:start_h+crop_size, start_w:start_w+crop_size]

# def save_multiday_test_results_to_nc(SIV_pred_list, SIC_pred_list, SIT_pred_list, targets_list, 
#                            output_path, predict_days=7):
#     """
#     将多天预测结果保存为NetCDF文件
    
#     参数:
#         SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
#         SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
#         SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
#         targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
#         predict_days: 预测天数 (N)
#     """
#     # 合并所有批次的结果
#     all_SIV_preds = [np.concatenate([pred[day] for pred in SIV_pred_list], axis=0) 
#                     for day in range(predict_days)]  # N个[total_samples,2,H,W]
    
#     all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
#                     for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
#     all_SIT_preds = [np.concatenate([pred[day] for pred in SIT_pred_list], axis=0) 
#                     for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
#     all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

#     # 创建数据字典
#     data_vars = {}
    
#     # 为每一天的预测添加数据变量
#     for day in range(predict_days):
#         # 预测的海冰速度 (u和v分量)
#         data_vars[f'SIV_u_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 0])
#         data_vars[f'SIV_v_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 1])
        
#         # 预测的海冰浓度
#         data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])
        
#         # 预测的海冰厚度
#         data_vars[f'SIT_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIT_preds[day][:, 0])

#         # 预测的海面风速 (u和v分量)
#         # data_vars[f'u10_pred_day{day+1}'] = (['sample', 'y', 'x'], all_wind10_preds[day][:, 0])
#         # data_vars[f'v10_pred_day{day+1}'] = (['sample', 'y', 'x'], all_wind10_preds[day][:, 1])
        
#         # 真实值 (每天有4个变量: u,v,sic,sit)
#         start_idx = day * 4
#         data_vars[f'SIV_u_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
#         data_vars[f'SIV_v_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+1])
#         data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])
#         data_vars[f'SIT_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+3])
#         # data_vars[f'u10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+4])
#         # data_vars[f'v10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+5])

#     # 创建坐标
#     total_samples = all_targets.shape[0]
#     sample_coords = np.arange(total_samples)
#     y_coords = np.arange(all_targets.shape[2])  # 高度维度
#     x_coords = np.arange(all_targets.shape[3])  # 宽度维度
    
#     # 创建xarray数据集
#     ds = xr.Dataset(
#         data_vars=data_vars,
#         coords={
#             'sample': sample_coords,
#             'y': y_coords,
#             'x': x_coords
#         }
#     )
    
#     # 保存为NetCDF文件
#     ds.to_netcdf(output_path)
#     print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
#     return ds

def save_multiday_test_results_to_nc(SIV_pred_list, SIC_pred_list, targets_list,
                           output_path, predict_days=7):
    """
    将多天预测结果保存为NetCDF文件
    
    参数:
        SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
        SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
        SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
        targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
        predict_days: 预测天数 (N)
    """
    # 合并所有批次的结果
    all_SIV_preds = [np.concatenate([pred[day] for pred in SIV_pred_list], axis=0) 
                    for day in range(predict_days)]  # N个[total_samples,2,H,W]
    
    all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
                    for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
    # all_SIT_preds = [np.concatenate([pred[day] for pred in SIT_pred_list], axis=0)
    #                 for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
    all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

    # 创建数据字典
    data_vars = {}
    
    # 为每一天的预测添加数据变量
    for day in range(predict_days):
        # 预测的海冰速度 (u和v分量)
        data_vars[f'SIV_u_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 0])
        data_vars[f'SIV_v_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 1])
        
        # 预测的海冰浓度
        data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])
        
        # 预测的海冰厚度
        # data_vars[f'SIT_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIT_preds[day][:, 0])

        # 预测的海面风速 (u和v分量)
        # data_vars[f'u10_pred_day{day+1}'] = (['sample', 'y', 'x'], all_wind10_preds[day][:, 0])
        # data_vars[f'v10_pred_day{day+1}'] = (['sample', 'y', 'x'], all_wind10_preds[day][:, 1])
        
        # 真实值 (每天有4个变量: u,v,sic,sit)
        start_idx = day * 4
        data_vars[f'SIV_u_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        data_vars[f'SIV_v_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+1])
        data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])
        data_vars[f'SIT_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+3])
        # data_vars[f'u10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+4])
        # data_vars[f'v10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+5])

    # 创建坐标
    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])  # 高度维度
    x_coords = np.arange(all_targets.shape[3])  # 宽度维度
    
    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )
    
    # 保存为NetCDF文件
    ds.to_netcdf(output_path)
    print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
    return ds


def save_multiday_test_results_to_nc_siv(SIV_pred_list, targets_list, output_path, predict_days=7):
    """
    将多天预测结果保存为NetCDF文件

    参数:
        SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
        SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
        SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
        targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
        predict_days: 预测天数 (N)
    """
    # 合并所有批次的结果
    all_SIV_preds = [np.concatenate([pred[day] for pred in SIV_pred_list], axis=0)
                     for day in range(predict_days)]  # N个[total_samples,2,H,W]

    all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

    # 创建数据字典
    data_vars = {}

    # 为每一天的预测添加数据变量
    for day in range(predict_days):
        # 预测的海冰速度 (u和v分量)
        data_vars[f'SIV_u_pred_day{day + 1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 0])
        data_vars[f'SIV_v_pred_day{day + 1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 1])

        # 真实值 (每天有4个变量: u,v,sic,sit)
        start_idx = day * 4
        data_vars[f'SIV_u_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        data_vars[f'SIV_v_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 1])
        # data_vars[f'SIC_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 2])
        # data_vars[f'SIT_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 3])
        # data_vars[f'u10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+4])
        # data_vars[f'v10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+5])

    # 创建坐标
    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])  # 高度维度
    x_coords = np.arange(all_targets.shape[3])  # 宽度维度

    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )

    # 保存为NetCDF文件
    ds.to_netcdf(output_path)
    print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
    return ds

def save_multiday_test_results_to_nc_sit(SIT_pred_list, targets_list, output_path, predict_days=7):
    """
    将多天预测结果保存为NetCDF文件

    参数:
        SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
        SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
        SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
        targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
        predict_days: 预测天数 (N)
    """
    # 合并所有批次的结果
    all_SIT_preds = [np.concatenate([pred[day] for pred in SIT_pred_list], axis=0)
                     for day in range(predict_days)]  # N个[total_samples,1,H,W]

    all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

    # 创建数据字典
    data_vars = {}

    # 为每一天的预测添加数据变量
    for day in range(predict_days):
        # 预测的海冰厚度
        data_vars[f'SIT_pred_day{day + 1}'] = (['sample', 'y', 'x'], all_SIT_preds[day][:, 0])

        # 真实值 (每天有4个变量: u,v,sic,sit)
        start_idx = day * 4
        # data_vars[f'SIV_u_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        # data_vars[f'SIV_v_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 1])
        # data_vars[f'SIC_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 2])
        data_vars[f'SIT_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 3])
        # data_vars[f'u10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+4])
        # data_vars[f'v10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+5])

    # 创建坐标
    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])  # 高度维度
    x_coords = np.arange(all_targets.shape[3])  # 宽度维度

    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )

    # 保存为NetCDF文件
    ds.to_netcdf(output_path)
    print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
    return ds

def save_multiday_test_results_to_nc_sic(SIC_pred_list, targets_list, output_path, predict_days=7):
    """
    将多天预测结果保存为NetCDF文件
    
    参数:
        SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
        SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
        SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
        targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
        predict_days: 预测天数 (N)
    """
    # 合并所有批次的结果
    all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
                    for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
    all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

    # 创建数据字典
    data_vars = {}
    
    # 为每一天的预测添加数据变量
    for day in range(predict_days):
        
        # 预测的海冰浓度
        data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])
        
        # 真实值 (每天有4个变量: u,v,sic,sit)
        start_idx = day * 4
        # data_vars[f'SIV_u_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        # data_vars[f'SIV_v_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+1])
        data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])
        # data_vars[f'SIT_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+3])

    # 创建坐标
    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])  # 高度维度
    x_coords = np.arange(all_targets.shape[3])  # 宽度维度
    
    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )
    
    # 保存为NetCDF文件
    ds.to_netcdf(output_path)
    print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
    return ds

def save_multiday_test_results_to_nc_swinUnet(SIC_pred_list, targets_list, output_path, predict_days=7):
    """
    将多天预测结果保存为NetCDF文件
    参数:
        SIV_pred_list: 列表，每个元素是N天的SIV预测结果（每个形状为[B,2,H,W]）
        SIC_pred_list: 列表，每个元素是N天的SIC预测结果（每个形状为[B,1,H,W]）
        SIT_pred_list: 列表，每个元素是N天的SIT预测结果（每个形状为[B,1,H,W]）
        targets_list: 列表，每个元素是N天的真实目标值（形状为[B,4*N,H,W]）
        predict_days: 预测天数 (N)
    """
    # 合并所有批次的结果
    all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
                    for day in range(predict_days)]  # N个[total_samples,1,H,W]
    
    all_targets = np.concatenate(targets_list, axis=0)  # [total_samples,4*N,H,W]

    # 创建数据字典
    data_vars = {}
    
    # 为每一天的预测添加数据变量
    for day in range(predict_days):
        # 预测的海冰浓度
        data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])
        
        # 真实值 (每天有4个变量: u,v,sic,sit)
        start_idx = day * 4
        data_vars[f'SIV_u_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        data_vars[f'SIV_v_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+1])
        data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])
        data_vars[f'SIT_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+3])
        # data_vars[f'u10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+4])
        # data_vars[f'v10_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+5])

    # 创建坐标
    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])  # 高度维度
    x_coords = np.arange(all_targets.shape[3])  # 宽度维度
    
    # 创建xarray数据集
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )
    
    # 保存为NetCDF文件
    ds.to_netcdf(output_path)
    print(f"测试结果已保存至 {output_path} (包含{predict_days}天预测)")
    return ds

# 使用示例
# if __name__ == "__main__":
#     time_range = 2557 # 介于0到2557之间
#     # 读取era5_data_masked.nc中的数据
#     era5_data = xr.open_dataset(r'data\era5_data_masked.nc')
#     sst = era5_data['sst'].data[0:time_range,:,:] # 海面温度信息, (2557, 361, 361)
#     t2m = era5_data['t2m'].data[0:time_range,:,:] # 2m温度, (2557, 361, 361)
#     u10 = era5_data['u10'].data[0:time_range,:,:] # 10米风速信息_u, (2557, 361, 361)
#     v10 = era5_data['v10'].data[0:time_range,:,:] # 10米风速信息_v, (2557, 361, 361)
#     z = era5_data['z'].data[0:time_range,:,:] # 500高压信息, (2557, 361, 361)
#     # 读取sic_data_masked.nc中的数据
#     sic_data = xr.open_dataset(r'data\sic_data_masked.nc')
#     sea_ice_concentration = sic_data['sea_ice_concentration'].data[0:time_range,:,:] # 海冰密集度信息, (2557, 361, 361)
#     # 读取siv_data_masked.nc中的数据
#     siv_data = xr.open_dataset(r'data\siv_data_masked.nc')
#     u = siv_data['u'].data[0:time_range,:,:] # 海冰速度信息_u, (2557, 361, 361)
#     v = siv_data['v'].data[0:time_range,:,:]  # 修复：添加了索引范围
#     latitude = siv_data['latitude'].data[0:time_range,:,:] # 纬度信息, (2557, 361, 361)
#     longitude = siv_data['longitude'].data[0:time_range,:,:] # 经度信息, (2557, 361, 361)

#     # 归一化
#     sst = normalize(sst, 269, 310)
#     t2m = normalize(t2m, 223, 319)
#     u10 = normalize(u10, -30, 33)
#     v10 = normalize(v10, -33, 32)
#     z = normalize(z, 44783, 59000)
#     sea_ice_concentration = normalize(sea_ice_concentration, 0, 100)
#     u = normalize(u, -60, 54)
#     v = normalize(v, -58, 54)
#     latitude = normalize(latitude, 30, 90)
#     longitude = normalize(longitude, -180, 180)

#     # 将每个数据中的nan值置为0
#     sst[np.isnan(sst)] = 0
#     t2m[np.isnan(t2m)] = 0
#     u10[np.isnan(u10)] = 0
#     v10[np.isnan(v10)] = 0
#     z[np.isnan(z)] = 0
#     sea_ice_concentration[np.isnan(sea_ice_concentration)] = 0
#     u[np.isnan(u)] = 0
#     v[np.isnan(v)] = 0
#     latitude[np.isnan(latitude)] = 0
#     longitude[np.isnan(longitude)] = 0

#     # 裁剪操作：从361x361取中间的256x256区域
#     sst = crop_center(sst)
#     t2m = crop_center(t2m)
#     u10 = crop_center(u10)
#     v10 = crop_center(v10)
#     z = crop_center(z)
#     sea_ice_concentration = crop_center(sea_ice_concentration)
#     u = crop_center(u)
#     v = crop_center(v)
#     latitude = crop_center(latitude)
#     longitude = crop_center(longitude)

#     # 创建数据加载器
#     train_loader, val_loader, test_loader = create_data_loaders_by_year(
#         sst=sst, t2m=t2m, u10=u10, v10=v10, z=z,
#         sea_ice_concentration=sea_ice_concentration, 
#         u=u, v=v, 
#         latitude=latitude, longitude=longitude,
#         batch_size=8
#     )
    
#     # 打印数据加载器信息
#     print(f"训练集样本数: {len(train_loader.dataset)}")
#     print(f"验证集样本数: {len(val_loader.dataset)}")
#     print(f"测试集样本数: {len(test_loader.dataset)}")
    
#     # 示例：获取一个批次的数据
#     inputs, targets = next(iter(train_loader))
#     print(f"输入数据形状: {inputs.shape}")  # 应输出: [8, 26, 361, 361]
#     print(f"目标数据形状: {targets.shape}")  # 应输出: [8, 3, 361, 361]