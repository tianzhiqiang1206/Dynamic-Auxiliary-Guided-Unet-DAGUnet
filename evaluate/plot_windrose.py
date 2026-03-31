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

# ===================== 全局配置 =====================
# 忽略无关警告
warnings.filterwarnings("ignore")

# 实验参数配置（集中管理，便于修改）
MODELS = ['DAGUnet', 'HISUNet', 'Unet_siv']
YEARS = [[2000, 2023]]  # 可扩展：[2000,2007],[2013,2020],[2012,2019]
ROOT_DIR = Path(r"E:/DAGUnet_code/newfolder")
OUTPUT_DIR = ROOT_DIR / "windrose"
OUTPUT_DIR.mkdir(exist_ok=True)  # 自动创建输出目录，避免路径不存在报错

# 归一化参数
SIV_U_MIN = -60
SIV_U_MAX = 54
SIV_V_MIN = -58
SIV_V_MAX = 54

# 可视化参数
SPEED_BINS = np.arange(0, 32, 4)  # 速度分箱
CMAP = cm.get_cmap('tab20c')      # 颜色映射
FIG_SIZE = (8, 8)                 # 风玫瑰图尺寸
DPI = 600                         # 保存图片分辨率
RADIAL_TICKS = [2, 4, 6, 8, 10, 12, 14, 16]  # 径向刻度
RADIAL_LIM = 18                   # 径向最大值

# 直方图配色（区分真实值/预测值）
TRUE_COLOR = 'skyblue'    # 真实值颜色
PRED_COLOR = 'lightcoral' # 预测值颜色
ALPHA = 0.7               # 透明度（避免重叠遮挡）
DAY_RANGE = range(1, 2)    # 处理第1-7天数据

# ===================== 工具函数 =====================
def inverse_normalize(data, min_val, max_val):
    """逆归一化函数：将归一化数据恢复到原始范围"""
    return data * (max_val - min_val) + min_val

def calculate_speed_and_dir(u, v):
    """
    计算海冰速度大小和方向（适配风玫瑰图格式）
    参数:
        u/v: 速度分量数组
    返回:
        speed: 速度大小 (km/day)
        source_dir: 适配风玫瑰图的方向（来向，0°=北，顺时针）
    """
    # 计算速度大小
    speed = np.sqrt(u**2 + v**2)
    
    # 计算方向并转换为气象学风玫瑰图格式
    angle_deg = np.rad2deg(np.arctan2(u, v))
    dest_dir = (360 + 90 - angle_deg) % 360  # 海冰去向（0°=北，顺时针）
    source_dir = (dest_dir + 180) % 360      # 转换为来向（风玫瑰图标准）
    
    return speed, source_dir

def plot_combined_speed_histogram(speeds_true, speeds_pred, model_name, year_range, save_path, unit="km/day"):
    """
    绘制7天聚合后的真实值+预测值合并直方图
    参数:
        speeds_true: 7天聚合的真实速度数组
        speeds_pred: 7天聚合的预测速度数组
        model_name: 模型名称（用于标题）
        year_range: 年份范围（用于标题）
        unit: 速度单位
    """
    # 检查数据有效性
    if len(speeds_true) == 0 and len(speeds_pred) == 0:
        print(f"警告: 模型 {model_name} ({year_range[0]}-{year_range[1]}) 无有效数据，跳过直方图绘制")
        return
    
    plt.figure(figsize=(8, 6))
    
    # 绘制真实值直方图
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
        # 真实值均值线
        plt.axvline(true_mean, color='blue', linestyle='dashed', linewidth=1, alpha=ALPHA)
    
    # 绘制预测值直方图
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
        # 预测值均值线
        plt.axvline(pred_mean, color='red', linestyle='dashed', linewidth=1, alpha=ALPHA)
    
    # 图表样式配置
    plt.title(
        f"Sea Ice Speed Distribution ({model_name})", 
        # fontsize=16
    )
    plt.xlabel(f'Sea Ice Speed ({unit})', fontsize=16)
    plt.ylabel('Frequency (Count)', fontsize=16)
    plt.legend(fontsize=12)  # 显示图例
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'  # 自动适配边界，避免图例截断
    )
    plt.show()

def plot_windrose(directions, speeds, title_suffix, save_path, model_name=None, year_range=None):
    """
    绘制并保存7天聚合后的风玫瑰图
    参数:
        directions: 7天聚合的方向数组
        speeds: 7天聚合的速度数组
        title_suffix: 标题后缀（True/Pred）
        save_path: 保存路径
        model_name: 模型名称（仅预测值需要）
        year_range: 年份范围（用于标题）
    """
    if len(directions) == 0 or len(speeds) == 0:
        print(f"警告: {title_suffix} 无有效数据，跳通风玫瑰图绘制")
        return
    
    # 创建风玫瑰图画布
    fig = plt.figure(figsize=FIG_SIZE)
    ax = WindroseAxes(fig, rect=[0.1, 0.1, 0.8, 0.8])
    fig.add_axes(ax)
    
    # 绘制风玫瑰图
    ax.bar(
        directions,
        speeds,
        normed=True,
        opening=0.8,
        edgecolor='white',
        bins=SPEED_BINS,
        cmap=CMAP
    )
    
    # 统一样式
    ax.set_yticks(RADIAL_TICKS)
    ax.set_ylim(0, RADIAL_LIM)
    ax.set_yticklabels([f'{x}%' for x in RADIAL_TICKS], fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    
    # 图例配置
    ax.set_legend(
        title='Speed (km/day)',
        loc='lower left',
        bbox_to_anchor=(1.05, 0.05),
        fontsize=16
    )
    
    plt.title(f"{model_name} windrose")
    # 保存图片
    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'  # 自动适配边界，避免图例截断
    )
    plt.close(fig)
    print(f"✅ 风玫瑰图已保存: {save_path}")

# ===================== 主程序 =====================
def main():
    for year_start, year_end in YEARS:
        year_range = (year_start, year_end)
        for model_name in MODELS:
            # 1. 加载数据
            data_path = ROOT_DIR / f"predict_{model_name}_pre7_{year_start}_{year_end}.nc"
            if not data_path.exists():
                print(f"❌ 文件不存在，跳过: {data_path}")
                continue
            
            try:
                ds = xr.open_dataset(data_path)
            except Exception as e:
                print(f"❌ 读取文件失败 {data_path}: {e}")
                continue
            
            # 2. 初始化数据容器（聚合7天数据）
            all_speeds_true = []
            all_directions_true = []
            all_speeds_pred = []
            all_directions_pred = []
            
            print(f"\n📊 处理数据: {model_name} ({year_start}-{year_end})，聚合7天数据...")
            
            # 3. 循环处理7天数据（仅聚合，不绘图）
            for day_idx in DAY_RANGE:
                try:
                    # 提取归一化数据
                    u_pred_norm = ds[f'SIV_u_pred_day{day_idx}'].values
                    u_true_norm = ds[f'SIV_u_true_day{day_idx}'].values
                    v_pred_norm = ds[f'SIV_v_pred_day{day_idx}'].values
                    v_true_norm = ds[f'SIV_v_true_day{day_idx}'].values
                    
                    # 逆归一化 + 单位转换 (km/day)
                    u_true = inverse_normalize(u_true_norm, SIV_U_MIN, SIV_U_MAX) * 0.864
                    u_pred = inverse_normalize(u_pred_norm, SIV_U_MIN, SIV_U_MAX) * 0.864
                    v_true = inverse_normalize(v_true_norm, SIV_V_MIN, SIV_V_MAX) * 0.864
                    v_pred = inverse_normalize(v_pred_norm, SIV_V_MIN, SIV_V_MAX) * 0.864
                    
                    # 展平数据
                    u_true_flat = u_true.flatten()
                    v_true_flat = v_true.flatten()
                    u_pred_flat = u_pred.flatten()
                    v_pred_flat = v_pred.flatten()
                    
                    # 计算真实值速度/方向并过滤异常值
                    speeds_true, dirs_true = calculate_speed_and_dir(u_true_flat, v_true_flat)
                    mask_true = (~np.isnan(speeds_true)) & (speeds_true <= 32)
                    all_speeds_true.append(speeds_true[mask_true])
                    all_directions_true.append(dirs_true[mask_true])
                    
                    # 计算预测值速度/方向并过滤异常值
                    speeds_pred, dirs_pred = calculate_speed_and_dir(u_pred_flat, v_pred_flat)
                    mask_pred = (~np.isnan(speeds_pred)) & (speeds_pred <= 32)
                    all_speeds_pred.append(speeds_pred[mask_pred])
                    all_directions_pred.append(dirs_pred[mask_pred])
                    
                    print(f"   ✅ 完成 Day{day_idx} 数据处理")
                    
                except KeyError:
                    print(f"⚠️  缺少 Day{day_idx} 数据，跳过")
                except Exception as e:
                    print(f"⚠️  处理 Day{day_idx} 失败: {e}")
            
            # 4. 聚合7天所有有效数据
            if not all_speeds_true or not all_speeds_pred:
                print(f"❌ 模型 {model_name} 无足够有效数据，跳过绘图")
                continue
            
            final_speeds_true = np.concatenate(all_speeds_true)
            final_directions_true = np.concatenate(all_directions_true)
            final_speeds_pred = np.concatenate(all_speeds_pred)
            final_directions_pred = np.concatenate(all_directions_pred)
            
            print(f"\n📈 7天聚合后 - 真实值样本数: {len(final_speeds_true)}")
            print(f"📈 7天聚合后 - 预测值样本数: {len(final_speeds_pred)}")
            
            # 5. 绘制7天聚合后的合并直方图（核心修改：仅绘制1张）
            # true_save_path = OUTPUT_DIR / f"zhifangtu_{model_name}_7days_{year_start}_{year_end}.png"
            # plot_combined_speed_histogram(
            #     final_speeds_true,
            #     final_speeds_pred,
            #     model_name,
            #     year_range,
            #     save_path = true_save_path
            # )
            
            # 6. 绘制并保存7天聚合后的风玫瑰图（核心修改：仅绘制1组）
            # 真实值风玫瑰图
            true_save_path = OUTPUT_DIR / f"true_windrose_7days_{year_start}_{year_end}.png"
            plot_windrose(
                final_directions_true,
                final_speeds_true,
                title_suffix="True (7 Days)",
                save_path=true_save_path,
                year_range=year_range
            )
            
            # 预测值风玫瑰图
            pred_save_path = OUTPUT_DIR / f"pred_{model_name}_windrose_7days_{year_start}_{year_end}.png"
            plot_windrose(
                final_directions_pred,
                final_speeds_pred,
                title_suffix=f"Pred ({model_name}) (7 Days)",
                save_path=pred_save_path,
                model_name=model_name,
                year_range=year_range
            )
    
    print("\n🎉 所有模型7天数据处理+绘图完成！")

# 程序入口
if __name__ == "__main__":
    main()