# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from metrics_record import IIEE_UE_OE_metrics

# def plot_year_month_order(
#     multi_year_data: dict,
#     months: list = ['Jul', 'Aug', 'Sep'],
#     y_label: str = 'IIEE (10⁶ km²)',
#     title: str = 'IIEE/OE/UE',
#     # subtitle: str = 'Stacked OE/UE with IIEE Trend Lines',
#     save_path: str = './year_month_order_iiee.png',
#     dpi: int = 600
# ) -> None:
#     # 1. 生成横轴标签（按年份→月份顺序）
#     years = sorted(multi_year_data.keys())
#     x_labels = [f'{year}-{month}' for year in years for month in months]
#     x = np.arange(len(x_labels))  # 横轴位置索引

#     # 2. 样式配置：模型→颜色，年份→标记
#     model_styles = {
#         'Unet': {'color': '#1f77b4'},       # 蓝色
#         'HISUnet': {'color': '#ff7f0e'},    # 橙色
#         'DAGUnet': {'color': '#2ca02c'}     # 绿色
#     }
#     year_markers = {
#         '2007': 'o',  # 圆点
#         '2012': 's',  # 方块
#         '2019': '^',  # 上三角
#         '2020': 'D'   # 菱形
#     }

#     # 3. 创建画布
#     fig, ax = plt.subplots(figsize=(12, 6))
#     bar_width = 0.2  # 每个模型的条形宽度

#     # 4. 遍历模型和年份绘图
#     for model_idx, (model, model_style) in enumerate(model_styles.items()):
#         all_iiie = []  # 存储当前模型所有年份的IIEE数据
#         for year in years:
#             metrics = multi_year_data[year][model]
#             # 数据单位转换（10⁶ km²）
#             iiie = np.array(metrics['IIEE']) / 1e6
#             oe = np.array(metrics['OE']) / 1e6
#             ue = np.array(metrics['UE']) / 1e6
#             all_iiie.extend(iiie)

#             # 计算当前年份的x轴索引（修正：转换为numpy数组便于计算）
#             year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
#             # 修正：将偏移量转换为数组后与索引相加（解决类型错误）
#             x_offset = year_x_indices + model_idx * bar_width

#             # 绘制OE底层条形
#             ax.bar(
#                 x_offset, oe,
#                 width=bar_width,
#                 color=model_style['color'],
#                 alpha=0.7,
#                 label=f'{model} - OE' if (year == years[0] and model_idx == 0) else ""
#             )
#             # 绘制UE上层条形
#             ax.bar(
#                 x_offset, ue,
#                 bottom=oe,
#                 width=bar_width,
#                 color=model_style['color'],
#                 alpha=0.3,
#                 label=f'{model} - UE' if (year == years[0] and model_idx == 0) else ""
#             )

#         # 绘制当前模型的IIEE折线
#         ax.plot(
#             x + model_idx * bar_width,  # 折线x位置与条形对齐
#             all_iiie,
#             color=model_style['color'],
#             linewidth=2,
#             marker='o',
#             markersize=6,
#             label=f'{model} - IIEE'
#         )

#     # 5. 绘制年份标记（在IIEE折线上叠加年份专属标记）
#     for year in years:
#         marker = year_markers[year]
#         for model_idx, (model, model_style) in enumerate(model_styles.items()):
#             metrics = multi_year_data[year][model]
#             iiie = np.array(metrics['IIEE']) / 1e6
#             year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
#             x_offset = year_x_indices + model_idx * bar_width  # 同样修正为数组计算
#             ax.scatter(
#                 x_offset, iiie,
#                 color=model_style['color'],
#                 marker=marker,
#                 s=50,
#                 label=f'{year}' if (model == list(model_styles.keys())[0] and year == years[0]) else ""
#             )

#     # 6. 图表配置
#     ax.set_xlabel('Year-Month', fontsize=11, fontweight='bold')
#     ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
#     ax.set_title(f'{title}', fontsize=13, fontweight='bold', pad=20)
#     ax.set_xticks(x + bar_width)
#     ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
#     ax.tick_params(axis='y', labelsize=10)
#     ax.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

#     # # 7. 自定义图例
#     # handles, labels = ax.get_legend_handles_labels()
#     # oe_ue_labels = [l for l in labels if 'OE' in l or 'UE' in l][:2]
#     # oe_ue_handles = [h for h, l in zip(handles, labels) if l in oe_ue_labels]
#     # model_labels = [l for l in labels if 'IIEE' in l]
#     # model_handles = [h for h, l in zip(handles, labels) if l in model_labels]
#     # year_labels = [l for l in labels if l in years]
#     # year_handles = [h for h, l in zip(handles, labels) if l in year_labels]

#     # legend1 = ax.legend(oe_ue_handles, oe_ue_labels, loc='upper left', fontsize=9,
#     #                    frameon=True, title='Metrics', title_fontsize=10)
#     # ax.add_artist(legend1)
#     # legend2 = ax.legend(model_handles, model_labels, loc='upper center', fontsize=9,
#     #                    frameon=True, title='Model', title_fontsize=10)
#     # ax.add_artist(legend2)
#     # ax.legend(year_handles, year_labels, loc='upper right', fontsize=9,
#     #          frameon=True, title='Year', title_fontsize=10)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
#     plt.show()


# # 调用绘图函数
# if __name__ == "__main__":
#     multi_year_data = IIEE_UE_OE_metrics

#     plot_year_month_order(
#         multi_year_data=multi_year_data,
#         months=['Jul', 'Aug', 'Sep'],
#         title='IIEE/OE/UE',
#         # subtitle='Ordered as 2007-Jul → 2007-Aug → ... → 2020-Oct',
#         save_path='./year_month_order_iiee.png'
#     )



'''

'''
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from metrics_record import IIEE_UE_OE_metrics

# def plot_year_month_order(
#     multi_year_data: dict,
#     months: list = ['Jul', 'Aug', 'Sep'],
#     y_label: str = 'IIEE (10⁶ km²)',
#     title: str = 'IIEE/OE/UE',
#     save_path: str = './year_month_order_iiee_with_average.png',
#     dpi: int = 600
# ) -> None:
#     # 1. 生成横轴标签（按年份→月份顺序，包含Average）
#     years = sorted([y for y in multi_year_data.keys() if y != 'Average'])  # 先处理实际年份
#     years_with_avg = years + ['Average']  # 添加平均值作为最后一个"年份"
#     x_labels = [f'{year}-{month}' for year in years_with_avg for month in months]
#     x = np.arange(len(x_labels))  # 横轴位置索引

#     # 2. 样式配置：模型→颜色，年份→标记（为Average添加特殊标记）
#     model_styles = {
#         'Unet': {'color': '#FF0E0D'},       # 蓝色
#         'HISUnet': {'color': '#0606FB'},    # 橙色
#         'DAGUnet': {'color': '#30A355'}     # 绿色
#     }
#     year_markers = {
#         '2007': 's',  # 圆点
#         '2012': 's',  # 方块
#         '2019': 's',  # 上三角
#         '2020': 's',  # 菱形
#         'Average': 's' # 星号，用于区分平均值
#     }

#     # 3. 创建画布
#     fig, ax = plt.subplots(figsize=(14, 7))
#     bar_width = 0.2  # 每个模型的条形宽度

#     # 4. 遍历模型和年份（包括Average）绘图
#     for model_idx, (model, model_style) in enumerate(model_styles.items()):
#         all_iiie = []  # 存储当前模型所有年份的IIEE数据
#         for year in years_with_avg:  # 这里使用包含Average的年份列表
#             metrics = multi_year_data[year][model]
#             # 数据单位转换（10⁶ km²）
#             iiie = np.array(metrics['IIEE']) / 1e6
#             oe = np.array(metrics['OE']) / 1e6
#             ue = np.array(metrics['UE']) / 1e6
#             all_iiie.extend(iiie)

#             # 计算当前年份的x轴索引
#             year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
#             x_offset = year_x_indices + model_idx * bar_width

#             # 绘制OE底层条形
#             ax.bar(
#                 x_offset, oe,
#                 width=bar_width,
#                 color=model_style['color'],
#                 alpha=0.7,
#                 label=f'{model} - OE' if (year == years_with_avg[0] and model_idx == 0) else ""
#             )
#             # 绘制UE上层条形
#             ax.bar(
#                 x_offset, ue,
#                 bottom=oe,
#                 width=bar_width,
#                 color=model_style['color'],
#                 alpha=0.3,
#                 label=f'{model} - UE' if (year == years_with_avg[0] and model_idx == 0) else ""
#             )

#         # 绘制当前模型的IIEE折线（包括平均值部分）
#         ax.plot(
#             x + model_idx * bar_width,  # 折线x位置与条形对齐
#             all_iiie,
#             color=model_style['color'],
#             linewidth=2,
#             marker='o',
#             markersize=6,
#             label=f'{model} - IIEE'
#         )

#     # 5. 绘制年份标记（包括Average的特殊标记）
#     for year in years_with_avg:
#         marker = year_markers[year]
#         for model_idx, (model, model_style) in enumerate(model_styles.items()):
#             metrics = multi_year_data[year][model]
#             iiie = np.array(metrics['IIEE']) / 1e6
#             year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
#             x_offset = year_x_indices + model_idx * bar_width
#             ax.scatter(
#                 x_offset, iiie,
#                 color=model_style['color'],
#                 marker=marker,
#                 s=50,
#                 label=f'{year}' if (model == list(model_styles.keys())[0] and year == years_with_avg[0]) else ""
#             )

#     # 6. 图表配置
#     ax.set_xlabel('Target Month', fontsize=11, fontweight='bold')
#     ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
#     ax.set_title(f'{title}', fontsize=13, fontweight='bold', pad=20)
#     ax.set_xticks(x + bar_width)
#     ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
#     ax.tick_params(axis='y', labelsize=10)
#     ax.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
#     plt.show()


# # 调用绘图函数
# if __name__ == "__main__":
#     multi_year_data = IIEE_UE_OE_metrics

#     plot_year_month_order(
#         multi_year_data=multi_year_data,
#         months=['Jul', 'Aug', 'Sep'],
#         title='IIEE/OE/UE Performance',
#         save_path='./SIE_visualization/year_month_order_iiee_with_average.png'
#     )


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from metrics_record import IIEE_UE_OE_metrics

def plot_year_month_order(
    multi_year_data: dict,
    months: list = ['Jul', 'Aug', 'Sep'],
    y_label: str = 'IIEE (10⁶ km²)',
    title: str = 'IIEE/OE/UE',
    save_path: str = './year_month_order_iiee_with_average.png',
    dpi: int = 600
) -> None:
    # 1. 生成横轴标签（按年份→月份顺序，包含Average）
    years = sorted([y for y in multi_year_data.keys() if y != 'Average'])
    years_with_avg = years + ['Average']
    x_labels = [f'{year}-{month}' for year in years_with_avg for month in months]
    x = np.arange(len(x_labels))

    # 2. 样式配置：模型→颜色
    model_styles = {
        'Unet': {'color': '#FF0E0D'},       # 红色
        'HISUnet': {'color': '#0606FB'},    # 蓝色
        'DAGUnet': {'color': '#30A355'}     # 绿色
    }
    year_markers = {
        '2007': 's',
        '2012': 's',
        '2019': 's',
        '2020': 's',
        'Average': 's'
    }

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.2

    # 4. 遍历模型和年份绘图
    for model_idx, (model, model_style) in enumerate(model_styles.items()):
        all_iiie = []
        for year in years_with_avg:
            metrics = multi_year_data[year][model]
            iiie = np.array(metrics['IIEE']) / 1e6
            oe = np.array(metrics['OE']) / 1e6
            ue = np.array(metrics['UE']) / 1e6
            all_iiie.extend(iiie)

            year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
            x_offset = year_x_indices + model_idx * bar_width

            # 绘制OE和UE条形
            ax.bar(x_offset, oe, width=bar_width, color=model_style['color'], alpha=0.7)
            ax.bar(x_offset, ue, bottom=oe, width=bar_width, color=model_style['color'], alpha=0.3)

        # 绘制IIEE折线
        ax.plot(
            x + model_idx * bar_width,
            all_iiie,
            color=model_style['color'],
            linewidth=2,
            marker='o',
            markersize=6
        )

    # 5. 绘制年份标记
    for year in years_with_avg:
        marker = year_markers[year]
        for model_idx, (model, model_style) in enumerate(model_styles.items()):
            metrics = multi_year_data[year][model]
            iiie = np.array(metrics['IIEE']) / 1e6
            year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
            x_offset = year_x_indices + model_idx * bar_width
            ax.scatter(x_offset, iiie, color=model_style['color'], marker=marker, s=50)

    # 6. 创建清晰的模型颜色图例
    legend_patches = [
        mpatches.Patch(color=model_styles['Unet']['color'], label='Unet'),
        mpatches.Patch(color=model_styles['HISUnet']['color'], label='HISUnet'),
        mpatches.Patch(color=model_styles['DAGUnet']['color'], label='DAGUnet')
    ]
    ax.legend(
        handles=legend_patches,
        title='Model',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),  # 关键调整：把图例放到图内右上角
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0
    )

    # 7. 图表配置
    ax.set_xlabel('Target Month', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(f'{title}', fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# 调用绘图函数
if __name__ == "__main__":
    multi_year_data = IIEE_UE_OE_metrics

    plot_year_month_order(
        multi_year_data=multi_year_data,
        months=['Jul', 'Aug', 'Sep'],
        title='IIEE/OE/UE Performance',
        save_path='F:/DAGUnet_code/figures/Figure3.png'
    )