import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text


def plot_volcano(
    df,
    logfc_col='logfc',
    padj_col='pvals_adj',
    lfc_thr=1.0,
    adj_p_thr=0.05,
    title='Volcano Plot',
    figsize=(8, 6)
):
    """
    使用 DataFrame 绘制火山图。

    参数:
    -----------
    df : pd.DataFrame
        包含差异分析结果的 DataFrame。
    logfc_col : str
        Log Fold Change 的列名。
    padj_col : str
        调整后 P 值的列名。
    lfc_thr : float
        LogFC 的阈值（绝对值）。
    adj_p_thr : float
        P 值的显著性阈值（通常为 0.05）。
    """
    # 1. 准备数据：计算 -log10(padj)
    # 处理 padj 为 0 的情况，避免 log10 报错
    plot_df = df.copy()
    plot_df['-log10padj'] = -np.log10(plot_df[padj_col].replace(0, 1e-300))

    # 2. 标记显著性分类
    # 定义分类逻辑
    plot_df['group'] = 'NS'  # Non-Significant
    plot_df.loc[(plot_df[logfc_col] > lfc_thr) & (plot_df[padj_col] < adj_p_thr), 'group'] = 'Up'
    plot_df.loc[(plot_df[logfc_col] < -lfc_thr) & (plot_df[padj_col] < adj_p_thr), 'group'] = 'Down'

    # 3. 绘图
    plt.figure(figsize=figsize)

    # 调色板：上调(Teal), 下调(Salmon), 不显著(Grey)
    colors = {'Up': '#4db6ac', 'Down': '#ff8a80', 'NS': '#e0e0e0'}

    sns.scatterplot(
        data=plot_df,
        x=logfc_col,
        y='-log10padj',
        hue='group',
        palette=colors,
        hue_order=['Down', 'NS', 'Up'],
        edgecolor=None,
        s=10,
        alpha=0.6
    )

    # 4. 添加阈值线
    plt.axhline(-np.log10(adj_p_thr), color='black', linestyle='--', lw=0.8, alpha=0.5)
    plt.axvline(lfc_thr, color='black', linestyle='--', lw=0.8, alpha=0.5)
    plt.axvline(-lfc_thr, color='black', linestyle='--', lw=0.8, alpha=0.5)

    # 5. 图形修饰
    plt.title(title, fontsize=15)
    plt.xlabel('$\log_2(Fold Change)$', fontsize=12)
    plt.ylabel('$-\log_{10}(P_{adj})$', fontsize=12)
    plt.legend(title='Expression', loc='upper right')

    sns.despine()
    plt.tight_layout()

    return plt.gca()


def plot_cluster_deg_jitter_highlight(
    deg_df,
    cluster_key='cluster',
    target_name='FIRES',
    reference_name='CTRL',
    highlight_dict=None,
    vrange=(-10, 10),
    x_label_rotation=0,
    save_path=None,      # 新增：保存路径，例如 'my_plot.pdf'
    figsize=(12, 8)
):
    """
    绘制带有基因标注的集群差异表达抖动图。
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 1. 准备绘图数据
    clusters = deg_df[cluster_key].unique()
    cluster_map = {cat: i for i, cat in enumerate(clusters)}

    plot_df = deg_df.copy()
    plot_df['x_idx'] = plot_df[cluster_key].map(cluster_map)

    # 视觉修复：仅为坐标定位进行 Clip，解决无限 logFC 导致的视觉断层问题
    plot_df['y_plot'] = plot_df['logfc'].clip(vrange[0], vrange[1])

    # 生成水平抖动
    plot_df['x_jittered'] = plot_df['x_idx'] + np.random.uniform(-0.3, 0.3, size=len(plot_df))

    # 2. 绘制散点
    up = plot_df[plot_df['logfc'] > 0]
    down = plot_df[plot_df['logfc'] < 0]

    # 使用 Teal 和 Salmon 配色
    ax.scatter(up['x_jittered'], up['y_plot'], c='#4db6ac', s=5, alpha=0.4, rasterized=True)
    ax.scatter(down['x_jittered'], down['y_plot'], c='#ff8a80', s=5, alpha=0.4, rasterized=True)

    # 3. 背景装饰（交替灰色条纹）
    for i in range(len(clusters)):
        ax.axvspan(i - 0.4, i + 0.4, color='grey', alpha=0.05, zorder=0)

    # 4. 基因标注
    texts = []
    if highlight_dict:
        for cluster, genes_to_label in highlight_dict.items():
            subset = plot_df[(plot_df[cluster_key] == cluster) & (plot_df['gene'].isin(genes_to_label))]
            for _, row in subset.iterrows():
                texts.append(ax.text(row['x_jittered'], row['y_plot'], row['gene'], fontsize=8))

    # 自动调整位置以防重叠
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    # 5. 样式修饰
    ax.set_ylim(vrange[0] - 1, vrange[1] + 1)
    ax.set_xticks(range(len(clusters)))

    # 处理 X 轴标签旋转
    # 如果旋转角度不为 0，ha='right' 能让标签末尾对齐刻度线
    ax.set_xticklabels(
        clusters,
        rotation=x_label_rotation,
        ha='right' if x_label_rotation != 0 else 'center'
    )

    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel(f'Average $\log_2(\text{{Fold Change}})$ ({target_name} : {reference_name})')

    # 坐标轴两端的条件说明文字
    ax.text(-0.5, vrange[1], target_name, fontweight='bold', va='top')
    ax.text(-0.5, vrange[0], reference_name, fontweight='bold', va='bottom')

    sns.despine()
    plt.tight_layout()

    # 6. 保存功能
    if save_path:
        # bbox_inches='tight' 非常关键，防止旋转后的文字被裁掉
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至: {save_path}")

    return fig, ax
