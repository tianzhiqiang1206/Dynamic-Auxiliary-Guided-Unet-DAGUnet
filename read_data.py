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
        self.input_data['u'] = u
        self.input_data['v'] = v
        self.input_data['sea_ice_concentration'] = sea_ice_concentration
        self.input_data['sea_ice_thickness'] = sea_ice_thickness
        self.input_days = input_days
        self.predict_days = predict_days
        self.num_samples = u.shape[0] - input_days - predict_days + 1
        height, width = u.shape[1:]
        self.x_coords = np.arange(width) / width
        self.y_coords = np.arange(height) / height
        self.mask = sst[0] != 0

    def __len__(self):
        """返回数据集的样本数量"""
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx + self.input_days + self.predict_days >= self.input_data['sea_ice_concentration'].shape[0]:
            raise IndexError("索引超出范围")
            
        input_channels = []
        for var_name in ['u','v','sea_ice_concentration']: # for DAGUnet and HISUnet model train/validate/test
        # for var_name in ['sea_ice_concentration']: # for Unet_sic model train/validate/test
        # for var_name in ['u','v']: # for Unet_siv model train/validate/test
            var_data = self.input_data[var_name]
            for day in range(self.input_days):
                input_channels.append(var_data[idx + day])

        input_data = np.stack(input_channels, axis=0)

        target_data = []
        for day in range(self.predict_days):
            target_day = idx + self.input_days + day
            target_data.extend([
                self.input_data['u'][target_day],
                self.input_data['v'][target_day],
                self.input_data['sea_ice_concentration'][target_day], 
            ])
        target_data = np.stack(target_data, axis=0)
        mask = self.mask
        return input_data, target_data, mask

def create_data_loaders_by_year(sst, sea_ice_concentration, u, v, sea_ice_thickness, 
                                input_days, predict_days, year_list, batch_size=8, shuffle=True):
    days_per_year = [366, 365, 365, 365,
                     366, 365, 365, 365, 
                     366, 365, 365, 365,
                     366, 365, 365, 365,
                     366, 365, 365, 365,
                     366, 365, 365, 365]
    
    year_indices = {}
    start_idx = 0
    for i, days in enumerate(range(2000, 2024)):
        end_idx = start_idx + days_per_year[i] - input_days - predict_days 
        year_indices[i] = (start_idx, end_idx)
        start_idx = end_idx + input_days + predict_days

    train_start_year = year_list[0]
    train_end_year = year_list[1]
    train_years = list(range(train_start_year, train_end_year + 1))
    val_start_year = year_list[2]
    val_end_year = year_list[3]
    val_years = list(range(val_start_year, val_end_year + 1))
    test_start_year = year_list[4]
    test_end_year = year_list[5]
    test_years = list(range(test_start_year, test_end_year + 1))

    def get_indices_for_years(years):
        indices = []
        for year in years:
            year_idx = year - 2000
            start, end = year_indices[year_idx]
            indices.extend(range(start, end + 1))
        return list(range(indices[0], indices[-1]))
    
    train_indices = get_indices_for_years(train_years)
    val_indices = get_indices_for_years(val_years)
    test_indices = get_indices_for_years(test_years)

    print(f"Training Set ({len(train_indices)} samples): {train_years[0]}-{train_years[-1]} year")
    print(f"Validating Set ({len(val_indices)} samples): {val_years[0]}-{val_years[-1]} year")
    print(f"Testing Set ({len(test_indices)} samples): {test_years[0]}-{test_years[-1]} year")
    
    dataset = SeaIcePredictionDataset(
        sst, sea_ice_concentration=sea_ice_concentration, 
        u=u, v=v, sea_ice_thickness = sea_ice_thickness,
        input_days=input_days, predict_days=predict_days,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
                                    
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def crop_center(data, crop_size=256):
    if len(data.shape) == 3: 
        _, h, w = data.shape
    elif len(data.shape) == 2: 
        h, w = data.shape
    else:
        raise ValueError(f" {data.shape} is not supported to crop!")
    
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    
    if len(data.shape) == 3:
        return data[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    else:
        return data[start_h:start_h+crop_size, start_w:start_w+crop_size]

def save_multiday_test_results_to_nc(SIV_pred_list, SIC_pred_list, targets_list,
                           output_path, predict_days=7):
                               
    all_SIV_preds = [np.concatenate([pred[day] for pred in SIV_pred_list], axis=0) 
                    for day in range(predict_days)] 
    all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
                    for day in range(predict_days)]
    all_targets = np.concatenate(targets_list, axis=0)
    data_vars = {}
    
    for day in range(predict_days):
        data_vars[f'SIV_u_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 0])
        data_vars[f'SIV_v_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 1])
        data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])

        start_idx = day * 3
        data_vars[f'SIV_u_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        data_vars[f'SIV_v_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+1])
        data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])

    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])
    x_coords = np.arange(all_targets.shape[3]) 
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )
    
    ds.to_netcdf(output_path)
    print(f"The predicted results are saved in {output_path}")
    return ds

def save_multiday_test_results_to_nc_siv(SIV_pred_list, targets_list, output_path, predict_days=7):

    all_SIV_preds = [np.concatenate([pred[day] for pred in SIV_pred_list], axis=0)
                     for day in range(predict_days)]
    all_targets = np.concatenate(targets_list, axis=0)
    data_vars = {}

    for day in range(predict_days):
        data_vars[f'SIV_u_pred_day{day + 1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 0])
        data_vars[f'SIV_v_pred_day{day + 1}'] = (['sample', 'y', 'x'], all_SIV_preds[day][:, 1])

        start_idx = day * 3
        data_vars[f'SIV_u_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx])
        data_vars[f'SIV_v_true_day{day + 1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx + 1])

    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])
    x_coords = np.arange(all_targets.shape[3])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )

    ds.to_netcdf(output_path)
    print(f"The predicted results are saved in {output_path}")
    return ds

def save_multiday_test_results_to_nc_sic(SIC_pred_list, targets_list, output_path, predict_days=7):

    all_SIC_preds = [np.concatenate([pred[day] for pred in SIC_pred_list], axis=0) 
                    for day in range(predict_days)] 
    
    all_targets = np.concatenate(targets_list, axis=0)

    data_vars = {}

    for day in range(predict_days):
        data_vars[f'SIC_pred_day{day+1}'] = (['sample', 'y', 'x'], all_SIC_preds[day][:, 0])
        start_idx = day * 3
        data_vars[f'SIC_true_day{day+1}'] = (['sample', 'y', 'x'], all_targets[:, start_idx+2])

    total_samples = all_targets.shape[0]
    sample_coords = np.arange(total_samples)
    y_coords = np.arange(all_targets.shape[2])
    x_coords = np.arange(all_targets.shape[3])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'sample': sample_coords,
            'y': y_coords,
            'x': x_coords
        }
    )

    ds.to_netcdf(output_path)
    print(f"The predicted results are saved in {output_path}")
    return ds
