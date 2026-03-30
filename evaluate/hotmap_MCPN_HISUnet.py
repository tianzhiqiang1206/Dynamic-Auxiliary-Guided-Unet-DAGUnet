import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from metrics_record import *

# ---------------------- 1. 数据准备与百分比计算 ----------------------
# 原始数据（与你提供的一致）
data = SIC_MAE_diff_para
figurename = 'F:/DAGUnet_code/figures/不同外部数据对预测的影响(SIC_MAE).png'

df = pd.DataFrame(data)

# 计算相对于HISUnet的百分比变化（(模型值 - HISUnet值)/HISUnet值 * 100）
hisunet = df["HISUnet"]
percent_change = pd.DataFrame()
for col in df.columns:
    if col != "HISUnet":
        percent_change[col] = ((df[col] - hisunet) / hisunet * 100).round(2)

# 生成行标签（模拟示例中的Year_Day格式，这里用Sample_1~Sample_35）
percent_change.index = [f"Sample_{i+1}" for i in range(len(percent_change))]


# ---------------------- 2. 自定义配色（与示例一致：红→橙→黄→蓝） ----------------------
colors = ['#00008B', '#4169E1', '#87CEFA', '#FFFFE0', '#FFA500', '#FF6347', '#8B0000']
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)


# ---------------------- 3. 绘制热力图 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 18))  # 调整画布大小以匹配行列数

# 绘制热力图（转置后模型为列，样本为行）
im = ax.imshow(percent_change.T, cmap=cmap, aspect='auto', vmin=-30, vmax=30)

# 设置行列标签
ax.set_xticks(range(len(percent_change.index)))
ax.set_xticklabels(percent_change.index, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(percent_change.columns)))
ax.set_yticklabels(percent_change.columns, fontsize=9)

# 添加数值标注（与示例一致，显示两位小数）
for i in range(len(percent_change.columns)):
    for j in range(len(percent_change.index)):
        val = percent_change.iloc[j, i]
        # 调整字体颜色，确保深色背景下文字清晰
        text_color = 'white' if abs(val) > 15 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

# 设置标题
ax.set_title('SIC MAE Metric Change (%)', fontsize=14, pad=20)

# 添加颜色条（与示例样式一致）
cbar = plt.colorbar(im, ax=ax, shrink=0.8, orientation='vertical')
cbar.set_label('-30', rotation=0, labelpad=10, y=0.02, fontsize=10)
cbar.set_label('30', rotation=0, labelpad=10, y=0.98, fontsize=10)
cbar.ax.tick_params(labelsize=0)  # 隐藏颜色条刻度，只保留上下标签

# 调整布局
plt.tight_layout()
plt.savefig(figurename, dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.cm as cm
# from matplotlib.patches import Wedge  # 关键：导入Wedge类

# # ---------------------- 1. 数据准备与百分比计算 ----------------------
# # 原始数据（与提供的一致）
# data = {
#     "HISUnet": [0.8767, 1.0844, 1.2167, 1.3546, 1.4947, 1.5761, 1.6882, 
#                 0.8526, 1.0483, 1.208, 1.3521, 1.5092, 1.6175, 1.7548, 
#                 0.8400, 1.0536, 1.1915, 1.3602, 1.4958, 1.6062,
#                 1.7310, 0.8384, 1.0433, 1.1735, 1.3247, 1.4641, 1.5556, 1.6849, 0.8897, 1.0823,
#                 1.2094, 1.3405, 1.4714, 1.5627, 1.6574],
#     "MCPN_sit": [0.9096, 1.1167, 1.2680, 1.4351, 1.5646, 1.6670, 1.8259, 0.9499, 1.1439, 1.2847,
#                  1.4441, 1.5677, 1.6608, 1.8100, 0.8460, 1.0381, 1.1881, 1.3414, 1.4589, 1.5573,
#                  1.7004, 0.8791, 1.0847, 1.2369, 1.3784, 1.5063, 1.6078, 1.7472, 0.7779, 0.9720,
#                  1.1262, 1.2514, 1.3788, 1.4745, 1.5856],
#     "MCPN_istl1": [0.9353, 1.1249, 1.2607, 1.4077, 1.5083, 1.5941, 1.7254, 0.9075, 1.1070, 1.2321,
#                    1.3853, 1.5028, 1.5884, 1.7139, 0.8181, 1.0163, 1.1521, 1.2997, 1.3977, 1.4956,
#                    1.6220, 0.9130, 1.1071, 1.2325, 1.3888, 1.4958, 1.5922, 1.7272, 0.7577, 0.9483,
#                    1.1019, 1.2090, 1.3210, 1.4217, 1.5070],
#     "MCPN_istl2": [0.9796, 1.1885, 1.3209, 1.4982, 1.6067, 1.7200, 1.8644, 1.0090, 1.2313, 1.3609,
#                    1.5335, 1.6524, 1.7784, 1.9196, 0.8264, 1.0260, 1.1676, 1.3175, 1.4286, 1.5361,
#                    1.6575, 0.8622, 1.0645, 1.1769, 1.3289, 1.4225, 1.5162, 1.6176, 0.7966, 0.9867,
#                    1.1401, 1.2669, 1.3766, 1.4536, 1.5383],
#     "MCPN_istl3": [0.8802, 1.0813, 1.2228, 1.3584, 1.4597, 1.5557, 1.6754, 0.9425, 1.1503, 1.2812,
#                    1.4441, 1.5516, 1.6589, 1.7844, 0.8063, 1.0016, 1.1359, 1.2851, 1.3826, 1.4736,
#                    1.5945, 0.9140, 1.1247, 1.2643, 1.4256, 1.5354, 1.6383, 1.7914, 0.7444, 0.9421,
#                    1.0950, 1.2205, 1.3273, 1.4182, 1.5072],
#     "MCPN_istl4": [1.1151, 1.3118, 1.4094, 1.5859, 1.6750, 1.7672, 1.9120, 0.8691, 1.0512, 1.2049,
#                    1.3522, 1.4494, 1.5706, 1.6992, 0.8277, 1.0187, 1.1564, 1.3033, 1.4025, 1.5078,
#                    1.6396, 0.8954, 1.0858, 1.2044, 1.3590, 1.4500, 1.5599, 1.6965, 0.7571, 0.9448,
#                    1.1056, 1.2416, 1.3401, 1.4390, 1.5321],
#     "MCPN_u10": [0.9148, 1.0866, 1.2408, 1.3899, 1.5165, 1.6186, 1.7264, 0.9218, 1.0888, 1.2712,
#                  1.4215, 1.5937, 1.6886, 1.8017, 0.8813, 1.0619, 1.2342, 1.3733, 1.5288, 1.6199,
#                  1.7251, 0.8876, 1.0529, 1.2182, 1.3477, 1.4999, 1.5934, 1.6932, 0.8592, 1.0311,
#                  1.1870, 1.3019, 1.4294, 1.5228, 1.6172],
#     "MCPN_v10": [1.1663, 1.3120, 1.4161, 1.6261, 1.7638, 1.8493, 1.9885, 0.8865, 1.0742, 1.2425,
#                  1.3843, 1.5272, 1.6355, 1.7596, 0.9256, 1.1054, 1.2443, 1.4262, 1.5706, 1.6784,
#                  1.8077, 0.8542, 1.0522, 1.2002, 1.3482, 1.4780, 1.5751, 1.7081, 0.7435, 0.9337,
#                  1.0873, 1.2104, 1.3261, 1.4254, 1.5142],
#     "MCPN_u100": [0.8675, 1.0484, 1.1933, 1.3512, 1.4802, 1.5838, 1.6882, 0.8427, 1.0425, 1.1867,
#                   1.3332, 1.4654, 1.5862, 1.7086, 0.9864, 1.1604, 1.2853, 1.4568, 1.5872, 1.6973,
#                   1.8107, 0.9755, 1.1506, 1.2909, 1.4716, 1.6090, 1.6971, 1.8252, 0.7629, 0.9421,
#                   1.0921, 1.2271, 1.3344, 1.4228, 1.5302],
#     "MCPN_v100": [0.9391, 1.1200, 1.2479, 1.4116, 1.5459, 1.6487, 1.7593, 0.8853, 1.0729, 1.2304,
#                   1.3832, 1.5260, 1.6286, 1.7396, 0.8097, 1.0099, 1.1506, 1.2963, 1.4284, 1.5392,
#                   1.6571, 0.8700, 1.0498, 1.1958, 1.3422, 1.4677, 1.5732, 1.6813, 0.7772, 0.9584,
#                   1.1006, 1.2380, 1.3374, 1.4212, 1.5159]
# }

# df = pd.DataFrame(data)

# # 计算相对于HISUnet的百分比变化
# hisunet = df["HISUnet"]
# percent_change = pd.DataFrame()
# for col in df.columns:
#     if col != "HISUnet":
#         percent_change[col] = ((df[col] - hisunet) / hisunet * 100).round(2)

# # 生成行标签
# percent_change.index = [f"Sample_{i+1}" for i in range(len(percent_change))]

# # ---------------------- 2. 自定义配色 ----------------------
# colors = ['#00008B', '#4169E1', '#87CEFA', '#FFFFE0', '#FFA500', '#FF6347', '#8B0000']
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# # ---------------------- 3. 绘制环形热力图（最终修复版） ----------------------
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
# plt.rcParams['axes.unicode_minus'] = False  # 支持负号

# # 设置图形大小（14x14，保证标签不拥挤）
# fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

# # 数据维度获取
# n_samples = len(percent_change)  # 样本数：35
# n_models = len(percent_change.columns)  # 模型数：9

# # 计算每个样本的角度（圆周均匀分布）
# theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)  # 0~2π，不包含终点（避免重复）
# angle_width = 2 * np.pi / n_samples  # 每个扇形的角度宽度（弧度）

# # 定义每个模型环的内/外半径（从内到外递增，环宽固定0.8）
# inner_radii = np.linspace(1, n_models, n_models)  # 内半径：1,2,...,9
# outer_radii = inner_radii + 0.8  # 外半径：1.8,2.8,...,9.8（环宽统一0.8）

# # 核心：绘制每个模型的环形热力图
# for i, model in enumerate(percent_change.columns):
#     model_values = percent_change[model].values  # 当前模型的35个样本值
    
#     # 为每个样本绘制扇形（组成当前模型的环）
#     for j in range(n_samples):
#         # 扇形的角度范围（起始角、结束角）
#         start_angle = theta[j]
#         end_angle = theta[j] + angle_width
        
#         # 数值归一化（-30→0，30→1，适配颜色映射）
#         norm_value = (model_values[j] + 30) / 60
#         # 防止数值超出[-30,30]导致颜色异常（截断处理）
#         norm_value = max(0, min(1, norm_value))
#         face_color = cmap(norm_value)
        
#         # 绘制扇形（使用导入的Wedge类）
#         wedge = Wedge(
#             center=(0, 0),  # 圆心在极坐标原点
#             r=outer_radii[i],  # 扇形外半径
#             theta1=np.degrees(start_angle),  # 起始角（弧度转角度，Wedge要求角度输入）
#             theta2=np.degrees(end_angle),    # 结束角（弧度转角度）
#             width=outer_radii[i] - inner_radii[i],  # 扇形宽度（外半径-内半径=0.8）
#             facecolor=face_color,  # 填充色（根据数值映射）
#             edgecolor='lightgray',  # 扇形边框色（浅灰，增强分隔）
#             linewidth=0.2,  # 边框粗细（细边框避免遮挡）
#             alpha=0.95  # 透明度（略透明，提升视觉效果）
#         )
#         ax.add_patch(wedge)  # 将扇形添加到图中
        
#         # 中间3个环显示数值标注（避免所有环显示导致拥挤）
#         if n_models // 2 - 1 <= i <= n_models // 2 + 1:  # i=3,4,5（中间3个环）
#             # 计算标注位置（扇形中心）
#             mid_radius = (inner_radii[i] + outer_radii[i]) / 2  # 半径中点
#             mid_angle = (start_angle + end_angle) / 2  # 角度中点
#             # 根据背景颜色选择文字颜色（对比度优化）
#             text_color = 'white' if abs(model_values[j]) > 15 else 'black'
#             # 添加数值标注（保留1位小数）
#             ax.text(
#                 mid_angle, mid_radius, f'{model_values[j]:.1f}',
#                 ha='center', va='center', color=text_color,
#                 fontsize=5.5, weight='bold'  # 字体大小适中，加粗突出
#             )

# # ---------------------- 4. 添加标签与样式优化 ----------------------
# # 内环添加模型名称标签（每个环的中心位置）
# for i, model in enumerate(percent_change.columns):
#     ax.text(
#         x=0, y=inner_radii[i] + 0.4,  # 标签位置（环的垂直中心）
#         s=model, ha='center', va='center',
#         fontsize=7.5, weight='bold', rotation=0,
#         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)  # 白色背景框，增强可读性
#     )

# # 外环添加样本标签（Sample_1~Sample_35）
# for j in range(n_samples):
#     label_angle = theta[j] + angle_width / 2  # 标签角度（扇形中心）
#     label_radius = outer_radii[-1] + 0.4  # 标签半径（最外环外侧0.4处）
#     # 调整文字对齐方式和旋转角度（避免文字倒置）
#     ha = 'left' if label_angle < np.pi else 'right'
#     rotation = np.degrees(label_angle) if label_angle < np.pi else np.degrees(label_angle) + 180
#     # 添加样本标签
#     ax.text(
#         label_angle, label_radius, percent_change.index[j],
#         ha=ha, va='center', rotation=rotation,
#         fontsize=6, color='darkblue', weight='medium'
#     )

# # ---------------------- 5. 图形属性设置 ----------------------
# ax.set_ylim(0, outer_radii[-1] + 1.2)  # 图形范围（包含标签，避免截断）
# ax.set_yticks([])  # 隐藏半径刻度
# ax.set_xticks([])  # 隐藏角度刻度
# ax.spines['polar'].set_visible(False)  # 隐藏极坐标外圈线（更简洁）

# # 添加标题（突出主题）
# ax.set_title(
#     'SIC MAE 指标变化百分比环形热力图',
#     fontsize=16, pad=30, weight='bold', color='darkred'
# )

# # 添加颜色条（右侧垂直放置，说明数值与颜色对应关系）
# cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 颜色条位置：左=0.92，下=0.15，宽=0.02，高=0.7
# norm = plt.Normalize(-30, 30)  # 颜色条数值范围（与数据匹配）
# cb = cm.ScalarMappable(cmap=cmap, norm=norm)
# cb.set_array([])  # 必需：ScalarMappable需要数组输入（空数组不影响）
# cbar = fig.colorbar(cb, cax=cax)
# # 颜色条标签设置
# cbar.set_label(
#     '百分比变化 (%)', rotation=270, labelpad=20,
#     fontsize=10, weight='bold'
# )
# cbar.ax.tick_params(labelsize=8)  # 颜色条刻度大小

# # 调整布局（避免标签被截断）
# plt.tight_layout()

# # 保存高清图片（300dpi，适合汇报/论文）
# plt.savefig('环形热力图_最终修复版.png', dpi=300, bbox_inches='tight', facecolor='white')

# # 显示图片（运行后自动弹出）
# plt.show()