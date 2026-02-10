import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text 

def plot_stacked_bar_repel(obs_df, group_col, type_col, 
                           mode='relative', log_scale=False, 
                           label_content='percentage', 
                           font_size=10, figsize=(12, 8), 
                           min_label_threshold=0.03, 
                           save_path=None):
    
    # 1. 数据准备
    counts = obs_df.groupby([type_col, group_col]).size().unstack(fill_value=0)
    row_sums = counts.sum(axis=1)
    
    if mode == 'relative':
        plot_data = counts.div(row_sums, axis=0)
        ylabel = 'Proportion'
        title = f'Relative Distribution: {group_col} per {type_col}'
    else:
        plot_data = counts
        ylabel = 'Number of Cells'
        title = f'Absolute Count: {group_col} per {type_col}'

    # 2. 绘图基础
    fig, ax = plt.subplots(figsize=figsize)
    plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor='black', linewidth=0.3, width=0.8)

    if log_scale and mode == 'absolute':
        ax.set_yscale('log')
        ax.set_ylim(bottom=1) 
    
    ax.grid(False)

    # 3. 标签逻辑 (保持之前的优化)
    texts_to_adjust = []
    x_centers = np.arange(len(counts.index))
    
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        cumulative_height = 0
        total_raw_count = row_sums.loc[idx]
        for group_name, val in row.items():
            if val <= 0: continue
            current_pct = val if mode == 'relative' else val / total_raw_count
            current_count = counts.loc[idx, group_name]
            center_y = cumulative_height + val / 2
            
            if label_content == 'percentage': label_text = f"{current_pct*100:.1f}%"
            elif label_content == 'count': label_text = f"{int(current_count)}"
            elif label_content == 'both': label_text = f"{int(current_count)}\n({current_pct*100:.1f}%)"
            else: label_text = ""

            if label_content:
                if current_pct < min_label_threshold:
                    anno = ax.annotate(label_text, xy=(x_centers[i], center_y),
                                       xytext=(x_centers[i] + 0.5, center_y),
                                       fontsize=font_size-1,
                                       arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
                    texts_to_adjust.append(anno)
                else:
                    ax.text(x_centers[i], center_y, label_text, ha='center', va='center',
                            rotation='vertical' if label_content != 'count' else 'horizontal',
                            fontweight='bold', fontsize=font_size)
            cumulative_height += val

    if texts_to_adjust:
        adjust_text(texts_to_adjust, ax=ax, expand_points=(1.2, 1.5),
                    only_move={'points':'y', 'text':'xy', 'objects':'xy'})

    # 4. 样式与图例顺序修正
    ax.set_title(title, fontsize=font_size + 4, pad=20)
    ax.set_xlabel(type_col, fontsize=font_size + 2)
    ax.set_ylabel(ylabel, fontsize=font_size + 2)
    plt.xticks(rotation=45, ha='right')

    # 获取并反转图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=group_col, 
              bbox_to_anchor=(1.1, 1), loc='upper left', frameon=False)
    
    plt.subplots_adjust(right=0.8)
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax