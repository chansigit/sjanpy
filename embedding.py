import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
from adjustText import adjust_text

def fancy_embedding_pro(adata, basis='umap', color='leiden', legend_title=None,
                        title=None, palette='tab20', dot_size=12, alpha=0.8, 
                        show_density=True, show_labels=True, 
                        save_path=None, dpi=300, figsize=(10, 8)):
    """
    单细胞高质量绘图函数：
    1. 支持自定义图例名称 (legend_title)
    2. 强制 XY 轴等比例 (aspect='equal')
    3. 同时保留图内标签与右侧图例
    4. 自动处理分类与连续变量
    """
    # 1. 获取坐标
    obsm_key = f'X_{basis}'
    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing {obsm_key} in adata.obsm")
    coords = adata.obsm[obsm_key][:, :2]
    df = pd.DataFrame(coords, columns=['dim1', 'dim2'], index=adata.obs_names)
    
    # 2. 获取颜色和显示名称
    display_name = legend_title if legend_title else color
    
    is_categorical = False
    if color in adata.obs.columns:
        df['color_val'] = adata.obs[color].values
        if adata.obs[color].dtype.name in ['category', 'object']:
            is_categorical = True
    elif color in adata.var_names:
        df['color_val'] = adata[:, color].X.toarray().flatten() if hasattr(adata.X, "toarray") else adata[:, color].X.flatten()
    else:
        raise ValueError(f"Key '{color}' not found.")

    # 3. 绘图初始化
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- 核心改进：设置 XY 轴等比例 ---
    # adjustable='datalim' 确保坐标轴范围根据数据自动调整，同时保持比例
    ax.set_aspect('equal', adjustable='datalim')

    # 4. 绘制背景密度
    if show_density and is_categorical:
        sns.kdeplot(data=df, x='dim1', y='dim2', fill=True, 
                    thresh=0.05, alpha=0.1, cmap='Greys', ax=ax, zorder=0)

    # 5. 绘制散点
    if is_categorical:
        scatter = sns.scatterplot(
            data=df, x='dim1', y='dim2', hue='color_val',
            palette=palette, s=dot_size, alpha=alpha, 
            edgecolor='none', ax=ax, legend='full'
        )
        # 修改图例标题
        ax.legend(title=display_name, loc='center left', bbox_to_anchor=(1, 0.5), 
                  frameon=False, markerscale=2)
    else:
        # 连续变量（基因表达量）
        scatter = ax.scatter(df['dim1'], df['dim2'], c=df['color_val'], 
                             cmap='viridis', s=dot_size, alpha=alpha, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label=display_name, shrink=0.6)

    # 6. 图上质心标签
    if is_categorical and show_labels:
        texts = []
        for cat in df['color_val'].unique():
            pos = df[df['color_val'] == cat][['dim1', 'dim2']].median()
            t = ax.text(pos['dim1'], pos['dim2'], str(cat), fontsize=9, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
            texts.append(t)
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    # 7. 美化修饰
    plot_title = title if title else f"{basis.upper()}: {display_name}"
    ax.set_title(plot_title, fontsize=15, loc='left', weight='bold', pad=20)
    ax.set_xlabel(f"{basis.upper()}1")
    ax.set_ylabel(f"{basis.upper()}2")
    
    # 移除刻度，保留干净的轴线
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, offset=5)

    # 8. 保存
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}")
    
    plt.show()
    return ax