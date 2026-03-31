'''
This code can be used to reproduce Fig.5
'''
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from windrose import WindroseAxes
import warnings
import matplotlib.cm as cm
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS = ['DAGUnet', 'HISUNet', 'Unet_siv']
YEARS = [[2000, 2023]]  # [2000,2007],[2005,2012],[2012,2019],[2013,2020]
ROOT_DIR = Path(r"E:/DAGUnet_code/newfolder")
OUTPUT_DIR = ROOT_DIR / "windrose"
OUTPUT_DIR.mkdir(exist_ok=True)

SIV_U_MIN = -60
SIV_U_MAX = 54
SIV_V_MIN = -58
SIV_V_MAX = 54

SPEED_BINS = np.arange(0, 32, 4)  
CMAP = cm.get_cmap('tab20c')  
FIG_SIZE = (8, 8)        
DPI = 600         
RADIAL_TICKS = [2, 4, 6, 8, 10, 12, 14, 16]  
RADIAL_LIM = 18         

TRUE_COLOR = 'skyblue'
PRED_COLOR = 'lightcoral' 
ALPHA = 0.7  
DAY_RANGE = range(1, 2) 

def inverse_normalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def calculate_speed_and_dir(u, v):
    speed = np.sqrt(u**2 + v**2)
    angle_deg = np.rad2deg(np.arctan2(u, v))
    dest_dir = (360 + 90 - angle_deg) % 360 
    source_dir = (dest_dir + 180) % 360 
    
    return speed, source_dir

def plot_combined_speed_histogram(speeds_true, speeds_pred, model_name, year_range, save_path, unit="km/day"):
    if len(speeds_true) == 0 and len(speeds_pred) == 0:
        return
    
    plt.figure(figsize=(8, 6))
    
    if len(speeds_true) > 0:
        true_mean = np.mean(speeds_true)
        plt.hist(
            speeds_true, 
            bins=30, 
            edgecolor='black', 
            alpha=ALPHA, 
            color=TRUE_COLOR,
            label=f'True (Mean: {true_mean:.2f} {unit})'
        )
        plt.axvline(true_mean, color='blue', linestyle='dashed', linewidth=1, alpha=ALPHA)
    
    if len(speeds_pred) > 0:
        pred_mean = np.mean(speeds_pred)
        plt.hist(
            speeds_pred, 
            bins=30, 
            edgecolor='black', 
            alpha=ALPHA, 
            color=PRED_COLOR,
            label=f'Pred (Mean: {pred_mean:.2f} {unit})'
        )
        plt.axvline(pred_mean, color='red', linestyle='dashed', linewidth=1, alpha=ALPHA)
    
    plt.title(
        f"Sea Ice Speed Distribution ({model_name})", 
    )
    plt.xlabel(f'Sea Ice Speed ({unit})', fontsize=16)
    plt.ylabel('Frequency (Count)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'
    )
    plt.show()

def plot_windrose(directions, speeds, title_suffix, save_path, model_name=None, year_range=None):
    
    if len(directions) == 0 or len(speeds) == 0:
        print(f"warning: {title_suffix} no valid data")
        return

    fig = plt.figure(figsize=FIG_SIZE)
    ax = WindroseAxes(fig, rect=[0.1, 0.1, 0.8, 0.8])
    fig.add_axes(ax)

    ax.bar(
        directions,
        speeds,
        normed=True,
        opening=0.8,
        edgecolor='white',
        bins=SPEED_BINS,
        cmap=CMAP
    )
    
    ax.set_yticks(RADIAL_TICKS)
    ax.set_ylim(0, RADIAL_LIM)
    ax.set_yticklabels([f'{x}%' for x in RADIAL_TICKS], fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    
    ax.set_legend(
        title='Speed (km/day)',
        loc='lower left',
        bbox_to_anchor=(1.05, 0.05),
        fontsize=16
    )
    
    plt.title(f"{model_name} windrose")

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'
    )
    plt.close(fig)
    print(f"✅ wind-rose is saved in: {save_path}")

def main():
    for year_start, year_end in YEARS:
        year_range = (year_start, year_end)
        for model_name in MODELS:
            data_path = ROOT_DIR / f"predict_{model_name}_pre7_{year_start}_{year_end}.nc"
            if not data_path.exists():
                print(f"❌ file does not exist，pass: {data_path}")
                continue
            try:
                ds = xr.open_dataset(data_path)
            except Exception as e:
                print(f"❌ file loading fails {data_path}: {e}")
                continue

            all_speeds_true = []
            all_directions_true = []
            all_speeds_pred = []
            all_directions_pred = []

            for day_idx in DAY_RANGE:
                try:
                    u_pred_norm = ds[f'SIV_u_pred_day{day_idx}'].values
                    u_true_norm = ds[f'SIV_u_true_day{day_idx}'].values
                    v_pred_norm = ds[f'SIV_v_pred_day{day_idx}'].values
                    v_true_norm = ds[f'SIV_v_true_day{day_idx}'].values

                    u_true = inverse_normalize(u_true_norm, SIV_U_MIN, SIV_U_MAX) * 0.864
                    u_pred = inverse_normalize(u_pred_norm, SIV_U_MIN, SIV_U_MAX) * 0.864
                    v_true = inverse_normalize(v_true_norm, SIV_V_MIN, SIV_V_MAX) * 0.864
                    v_pred = inverse_normalize(v_pred_norm, SIV_V_MIN, SIV_V_MAX) * 0.864

                    u_true_flat = u_true.flatten()
                    v_true_flat = v_true.flatten()
                    u_pred_flat = u_pred.flatten()
                    v_pred_flat = v_pred.flatten()

                    speeds_true, dirs_true = calculate_speed_and_dir(u_true_flat, v_true_flat)
                    mask_true = (~np.isnan(speeds_true)) & (speeds_true <= 32)
                    all_speeds_true.append(speeds_true[mask_true])
                    all_directions_true.append(dirs_true[mask_true])

                    speeds_pred, dirs_pred = calculate_speed_and_dir(u_pred_flat, v_pred_flat)
                    mask_pred = (~np.isnan(speeds_pred)) & (speeds_pred <= 32)
                    all_speeds_pred.append(speeds_pred[mask_pred])
                    all_directions_pred.append(dirs_pred[mask_pred])
                    
                except KeyError:
                    print(f"⚠️  missing Day{day_idx} data，pass")
                except Exception as e:
                    print(f"⚠️  process Day{day_idx} fail: {e}")
            
            if not all_speeds_true or not all_speeds_pred:
                continue
            
            final_speeds_true = np.concatenate(all_speeds_true)
            final_directions_true = np.concatenate(all_directions_true)
            final_speeds_pred = np.concatenate(all_speeds_pred)
            final_directions_pred = np.concatenate(all_directions_pred)
            
            # true_save_path = OUTPUT_DIR / f"zhifangtu_{model_name}_7days_{year_start}_{year_end}.png"
            # plot_combined_speed_histogram(
            #     final_speeds_true,
            #     final_speeds_pred,
            #     model_name,
            #     year_range,
            #     save_path = true_save_path
            # )
            
            true_save_path = OUTPUT_DIR / f"true_windrose_{year_start}_{year_end}.png"
            plot_windrose(
                final_directions_true,
                final_speeds_true,
                title_suffix="True (7 Days)",
                save_path=true_save_path,
                year_range=year_range
            )
            
            pred_save_path = OUTPUT_DIR / f"pred_{model_name}_windrose_{year_start}_{year_end}.png"
            plot_windrose(
                final_directions_pred,
                final_speeds_pred,
                title_suffix=f"Pred ({model_name}) (7 Days)",
                save_path=pred_save_path,
                model_name=model_name,
                year_range=year_range
            )

if __name__ == "__main__":
    main()
