import torch
import torch.nn as nn
import torch.optim as optim
from model_Unet import HISUnet
from read_data import *
import numpy as np
import xarray as xr
import random
import matplotlib.pyplot as plt
import sys

def run_experiment(year_list):
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    era5_data = xr.open_dataset(r'data\\era5_data_masked.nc')
    sic_data = xr.open_dataset(r'data\\sic_masked_with_sic.nc')
    siv_data = xr.open_dataset(r'data\\siv_masked_with_siv.nc')
    sit_data = xr.open_dataset(r'data\\sit_masked_with_sit.nc')

    sst = era5_data['sst'].data
    sea_ice_concentration = sic_data['sea_ice_concentration'].data
    u = siv_data['u'].data
    v = siv_data['v'].data
    sea_ice_thickness = sit_data['sea_ice_thickness'].data

    sst = normalize(sst, 269, 310)
    sea_ice_concentration = normalize(sea_ice_concentration, 0, 100)
    u = normalize(u, -60, 54)
    v = normalize(v, -58, 54)
    sea_ice_thickness = normalize(sea_ice_thickness, 0, 10)

    sst[np.isnan(sst)] = 0
    sea_ice_concentration[np.isnan(sea_ice_concentration)] = 0
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0
    sea_ice_thickness[np.isnan(sea_ice_thickness)] = 0

    sst = crop_center(sst)
    sea_ice_concentration = crop_center(sea_ice_concentration)
    u = crop_center(u)
    v = crop_center(v)
    sea_ice_thickness = crop_center(sea_ice_thickness)

    def convert_to_float32(inputs, targets, mask):
        inputs = inputs.float()
        targets = targets.float()
        mask = mask.float()
        return inputs, targets, mask

    input_days = 7
    predict_days = 7
    patience = 10  # 早停：10个epoch不提升就停（按你要求）
    lr_patience = 5  # 学习率衰减：5个epoch不提升就×0.1（按你要求）
    counter = 0
    best_val_loss = float('inf')
    train_flag = True  # 训练必须打开
    pretrain_flag = False

    best_model_path = rf'model/best_Unet_siv_pre7_{year_list[0]}_{year_list[-1]}.pth'
    output_nc_path = f'model/predict_Unet_siv_pre7_{year_list[0]}_{year_list[-1]}.nc'

    train_loader, val_loader, test_loader = create_data_loaders_by_year(
        sst, sea_ice_concentration=sea_ice_concentration,
        u=u, v=v, sea_ice_thickness=sea_ice_thickness, year_list=year_list,
        batch_size=4, input_days=input_days, predict_days=predict_days)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HISUnet(in_channels = 14, predict_days=predict_days).to(device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    model.apply(init_weights)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=lr_patience, 
        verbose=True
    )

    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"load pretrained model: {best_model_path}")
    except:
        print("No pretrained model, training from scratch")

    mse_criterion = nn.MSELoss()

    if train_flag:
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets, mask in train_loader:
                inputs, targets, mask = convert_to_float32(inputs, targets, mask)
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                optimizer.zero_grad()
                SIV_preds = model(inputs)

                mse_loss = 0
                for day in range(predict_days):
                    SIV_pred = SIV_preds[day]
                    target_day = day * 4
                    target_SIV = targets[:, target_day:target_day+2]
                    target_SIC = targets[:, target_day+2:target_day+3]
                    target_SIT = targets[:, target_day+3:target_day+4]
                    
                    loss_SIV = mse_criterion(SIV_pred[:, 0], target_SIV[:, 0]) * mask + mse_criterion(SIV_pred[:, 1], target_SIV[:, 1]) * mask
                    loss_SIV = loss_SIV.sum() / mask.sum()
                    mse_loss += loss_SIV

                loss = mse_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets, mask in val_loader:
                    inputs, targets, mask = convert_to_float32(inputs, targets, mask)
                    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                    SIV_preds = model(inputs)

                    mse_loss = 0
                    for day in range(predict_days):
                        SIV_pred = SIV_preds[day]
                        target_day = day * 4
                        target_SIV = targets[:, target_day:target_day+2]
                        loss_SIV = mse_criterion(SIV_pred[:, 0], target_SIV[:, 0]) * mask + mse_criterion(SIV_pred[:, 1], target_SIV[:, 1]) * mask
                        loss_SIV = loss_SIV.sum() / mask.sum()
                        mse_loss += loss_SIV

                    loss = mse_loss
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr:.8f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                counter = 0
                print("✅ Best model saved!")
            else:
                counter += 1
                print(f"⚠️ Early stopping counter: {counter}/{patience}")
                if counter >= patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss_SIV = 0.0
    test_loss = 0.0
    all_SIV_pred = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs, targets, mask = convert_to_float32(inputs, targets, mask)
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            SIV_preds = model(inputs)

            mse_loss_SIV = 0
            for day in range(predict_days):
                SIV_pred = SIV_preds[day]
                target_day = day * 4
                target_SIV = targets[:, target_day:target_day+2]
                loss_SIV = mse_criterion(SIV_pred[:, 0], target_SIV[:, 0]) * mask + mse_criterion(SIV_pred[:, 1], target_SIV[:, 1]) * mask
                loss_SIV = loss_SIV.sum() / mask.sum()
                mse_loss_SIV += loss_SIV

            test_loss_SIV += mse_loss_SIV.item()
            all_SIV_pred.append([pred.detach().cpu().numpy() for pred in SIV_preds])
            all_targets.append(targets.detach().cpu().numpy())

        test_loss_SIV /= len(test_loader)
        print(f'\n✅ Test Loss (SIV): {test_loss_SIV:.6f}')

    save_multiday_test_results_to_nc_siv(
        all_SIV_pred,
        all_targets,
        output_path=output_nc_path,
        predict_days=predict_days
    )

if __name__ == "__main__":
    year_sets = [
        [2000, 2016, 2017, 2020, 2021, 2023],
        [2000, 2004, 2005, 2006, 2007, 2007],       
        [2005, 2009, 2010, 2011, 2012, 2012],
        [2012, 2016, 2017, 2018, 2019, 2019],
        [2013, 2017, 2018, 2019, 2020, 2020]
    ]

    for i, year_list in enumerate(year_sets):
        print(f"\n===== start {i+1}/{len(year_sets)} experiment，year list: {year_list} =====")
        run_experiment(year_list)
        print(f"===== {i+1}/{len(year_sets)} experiment completed =====\n")
