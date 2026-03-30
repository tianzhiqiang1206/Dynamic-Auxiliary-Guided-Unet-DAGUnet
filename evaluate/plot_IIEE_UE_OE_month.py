'''
This code is used to reproduce Fig. 3
'''
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
    
    years = sorted([y for y in multi_year_data.keys() if y != 'Average'])
    years_with_avg = years + ['Average']
    x_labels = [f'{year}-{month}' for year in years_with_avg for month in months]
    x = np.arange(len(x_labels))

    model_styles = {
        'Unet': {'color': '#FF0E0D'},
        'HISUnet': {'color': '#0606FB'},
        'DAGUnet': {'color': '#30A355'} 
    }
    year_markers = {
        '2007': 's',
        '2012': 's',
        '2019': 's',
        '2020': 's',
        'Average': 's'
    }

    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width = 0.2

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

            ax.bar(x_offset, oe, width=bar_width, color=model_style['color'], alpha=0.7)
            ax.bar(x_offset, ue, bottom=oe, width=bar_width, color=model_style['color'], alpha=0.3)

        ax.plot(
            x + model_idx * bar_width,
            all_iiie,
            color=model_style['color'],
            linewidth=2,
            marker='o',
            markersize=6
        )

    for year in years_with_avg:
        marker = year_markers[year]
        for model_idx, (model, model_style) in enumerate(model_styles.items()):
            metrics = multi_year_data[year][model]
            iiie = np.array(metrics['IIEE']) / 1e6
            year_x_indices = np.array([i for i, label in enumerate(x_labels) if label.startswith(year)])
            x_offset = year_x_indices + model_idx * bar_width
            ax.scatter(x_offset, iiie, color=model_style['color'], marker=marker, s=50)

    legend_patches = [
        mpatches.Patch(color=model_styles['Unet']['color'], label='Unet'),
        mpatches.Patch(color=model_styles['HISUnet']['color'], label='HISUnet'),
        mpatches.Patch(color=model_styles['DAGUnet']['color'], label='DAGUnet')
    ]
    ax.legend(
        handles=legend_patches,
        title='Model',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0
    )

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

if __name__ == "__main__":
    multi_year_data = IIEE_UE_OE_metrics

    plot_year_month_order(
        multi_year_data=multi_year_data,
        months=['Jul', 'Aug', 'Sep'],
        title='IIEE/OE/UE Performance',
        save_path='F:/DAGUnet_code/figures/Figure3.png'
    )
