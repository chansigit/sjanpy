import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, optimal_leaf_ordering
from sklearn.cluster import KMeans
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --- Module 1: Data Calculation ---
def _prepare_dotplot_data(adata, genes, groupby, z_score):
    valid_genes = [g for g in genes if g in adata.var_names]
    exp_matrix = adata[:, valid_genes].X
    if issparse(exp_matrix): exp_matrix = exp_matrix.toarray()

    df = pd.DataFrame(exp_matrix, columns=valid_genes)
    df['group'] = adata.obs[groupby].values

    pct_exp = df.groupby('group').agg(lambda x: (x > 0).mean() * 100)
    avg_exp = df.groupby('group').mean()

    plot_data = avg_exp.apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=0) if z_score else avg_exp
    return pct_exp, avg_exp, plot_data

# --- Module 2: Clustering & Ordering ---
def _compute_ordering(plot_data, cluster_rows, cluster_cols, use_olo, row_km, col_km, gene_order, group_order):
    # --- Row Reordering (Genes) ---
    row_link = None
    row_clusters = None
    if gene_order is not None:
        gene_names = [g for g in gene_order if g in plot_data.columns]
    elif row_km:
        km = KMeans(n_clusters=row_km, n_init=10, random_state=42).fit(plot_data.T)
        row_clusters = pd.Series(km.labels_, index=plot_data.columns)
        gene_names = row_clusters.sort_values().index.tolist()
    elif cluster_rows:
        row_link = linkage(plot_data.T, method='ward')
        if use_olo: row_link = optimal_leaf_ordering(row_link, plot_data.T)
        gene_names = [plot_data.columns[i] for i in leaves_list(row_link)]
    else:
        gene_names = plot_data.columns.tolist()

    # --- Column Reordering (Groups) ---
    col_link = None
    col_clusters = None
    if group_order is not None:
        group_names = [g for g in group_order if g in plot_data.index]
    elif col_km:
        km = KMeans(n_clusters=col_km, n_init=10, random_state=42).fit(plot_data)
        col_clusters = pd.Series(km.labels_, index=plot_data.index)
        group_names = col_clusters.sort_values().index.tolist()
    elif cluster_cols:
        col_link = linkage(plot_data, method='ward')
        if use_olo: col_link = optimal_leaf_ordering(col_link, plot_data)
        group_names = [plot_data.index[i] for i in leaves_list(col_link)]
    else:
        group_names = plot_data.index.tolist()

    return gene_names, group_names, row_link, col_link, row_clusters, col_clusters


def complex_dotplot(
    adata, genes, groupby,
    z_score=True, cluster_rows=True, cluster_cols=True, use_olo=True,
    manual_gene_order=None, manual_group_order=None,
    show_dendrogram_x=True, show_dendrogram_y=True,
    dendrogram_ratio=(0.12, 0.08),
    row_km=None, col_km=None,
    x_rotation=90, dot_scale=5,
    dot_spacing_ratio=(0.8, 0.5), frame_margin=1.2,
    # 恢复这些参数以解决 TypeError
    legend_pos=(1.1, 0.8),
    colorbar_pos=(1.1, 0.2),
    colorbar_width="10%",
    colorbar_height="30%",
    legend_label_spacing=1.5,
    vmin=None, vmax=None,
    cmap='RdBu_r', title=None, save_path=None
):
    # 1. Prepare Data
    pct_exp, avg_exp, plot_data = _prepare_dotplot_data(adata, genes, groupby, z_score)
    gene_names, group_names, row_link, col_link, row_clusters, col_clusters = \
        _compute_ordering(plot_data, cluster_rows, cluster_cols, use_olo,
                          row_km, col_km, manual_gene_order, manual_group_order)

    # 2. Setup Figure
    width_per_col, height_per_row = dot_spacing_ratio
    dendro_h, dendro_w = dendrogram_ratio
    # 动态调整 figsize，确保右侧留出空间
    fig = plt.figure(figsize=(len(group_names) * width_per_col + 6, len(gene_names) * height_per_row + 3))

    # 主网格：最后一列专门给图例 (width_ratios 中最后一个 3 是图例区)
    gs = gridspec.GridSpec(2, 4, width_ratios=[0.2, 10, 10 * dendro_w, 3.5],
                           height_ratios=[10 * dendro_h, 10], wspace=0.1, hspace=0.05)
    ax_main = fig.add_subplot(gs[1, 1])

    # 3. Dendrograms
    if show_dendrogram_x and col_link is not None and manual_group_order is None:
        ax_top = fig.add_subplot(gs[0, 1])
        dendrogram(col_link, ax=ax_top, orientation='top', no_labels=True, color_threshold=0, above_threshold_color='black')
        ax_top.axis('off')
    if show_dendrogram_y and row_link is not None and manual_gene_order is None:
        ax_right = fig.add_subplot(gs[1, 2])
        dendrogram(row_link, ax=ax_right, orientation='right', no_labels=True, color_threshold=0, above_threshold_color='black')
        ax_right.axis('off')

    # 4. Prepare Scatter DF
    final_pct = pct_exp.loc[group_names, gene_names].T
    final_color = plot_data.loc[group_names, gene_names].T
    plot_df = final_pct.reset_index().melt(id_vars='index', var_name='group', value_name='pct')
    plot_df.columns = ['gene', 'group', 'pct']
    plot_df['color'] = final_color.reset_index().melt(id_vars='index', var_name='group', value_name='color')['color']

    # 5. Color Limits
    if vmin is None or vmax is None:
        abs_max = max(abs(plot_df['color'].min()), abs(plot_df['color'].max()))
        _vmin, _vmax = -abs_max, abs_max
    else: _vmin, _vmax = vmin, vmax

    # 6. Main Scatter Plot
    sc = ax_main.scatter(x=plot_df['group'], y=plot_df['gene'], s=plot_df['pct'] * dot_scale,
                         c=plot_df['color'], cmap=cmap, linewidths=0, vmin=_vmin, vmax=_vmax)

    # 7. Aesthetics
    ax_main.grid(False)
    ax_main.set_xticks(range(len(group_names)))
    ax_main.set_xticklabels(group_names, rotation=x_rotation)
    ax_main.set_yticks(range(len(gene_names)))
    ax_main.set_yticklabels(gene_names)
    ax_main.set_xlim(-frame_margin, (len(group_names)-1) + frame_margin)
    ax_main.set_ylim(-frame_margin, (len(gene_names)-1) + frame_margin)

    # 8. Legend 区域 (使用你传入的参数进行定位)
    # 创建一个覆盖右侧整列的隐藏 axis
    ax_leg_container = fig.add_subplot(gs[1, 3])
    ax_leg_container.axis('off')

    # 点大小图例
    legend_elements = [plt.scatter([], [], c='gray', s=p*dot_scale, label=f'{p}%', lw=0) for p in [25, 50, 75, 100]]
    ax_leg_container.legend(handles=legend_elements, title="Percent\nExpressed",
                            loc='upper left', bbox_to_anchor=legend_pos, frameon=False,
                            labelspacing=legend_label_spacing)

    # Colorbar：根据传入的 colorbar_pos, width, height 定制
    cax = inset_axes(ax_leg_container, width=colorbar_width, height=colorbar_height,
                     loc='lower left', bbox_to_anchor=(*colorbar_pos, 1, 1),
                     bbox_transform=ax_leg_container.transAxes, borderpad=0)
    plt.colorbar(sc, cax=cax, label='Z-score' if z_score else 'Mean Exp')

    if title: fig.suptitle(title, fontsize=16, y=0.98)
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)

    return fig, ax_main


####################################################################################################
# Fan Dotplot
####################################################################################################

def get_dotplot_df(adata, genes, groupby, expr_threshold=0.0, min_pct=0, keep_genes=None):
    """
    从 AnnData 提取数据，并过滤低表达基因，但保留指定的核心基因。

    参数:
    -----------
    adata : AnnData
    genes : list
        候选基因列表。
    groupby : str
        adata.obs 中的分组列名。
    expr_threshold : float
        全局最大平均表达量阈值。
    min_pct : float
        全局最大表达占比阈值。
    keep_genes : list, optional
        白名单基因列表。即使达不到阈值，也强制保留。
    """
    # 1. 确保输入基因存在于 adata 中
    genes = [g for g in genes if g in adata.var_names]
    if keep_genes is None:
        keep_genes = []
    else:
        # 确保白名单里的基因也在候选列表里
        keep_genes = [g for g in keep_genes if g in adata.var_names]

    # 2. 提取数据
    exp_data = adata[:, genes].X
    if hasattr(exp_data, "toarray"):
        exp_data = exp_data.toarray()

    temp_df = pd.DataFrame(exp_data, columns=genes, index=adata.obs[groupby])

    results = []

    # 3. 计算每个 Cluster 的统计量 (AvgExp 和 Pct)
    for cluster, group_data in temp_df.groupby(level=0):
        avg_exp = group_data.mean(axis=0)
        pct_exp = (group_data > 0).sum(axis=0) / len(group_data) * 100

        cluster_df = pd.DataFrame({
            'Cluster': cluster,
            'Gene': genes,
            'AvgExp': avg_exp.values,
            'Pct': pct_exp.values
        })
        results.append(cluster_df)

    full_df = pd.concat(results, ignore_index=True)

    # 4. 过滤逻辑
    # 计算每个基因在所有 Cluster 中的最大表现
    gene_stats = full_df.groupby('Gene').agg({'AvgExp': 'max', 'Pct': 'max'})

    # 逻辑判断：(满足表达量阈值 AND 满足占比阈值) OR (在白名单中)
    passed_filter = (
        ((gene_stats['AvgExp'] >= expr_threshold) & (gene_stats['Pct'] >= min_pct)) |
        (gene_stats.index.isin(keep_genes))
    )

    final_gene_list = gene_stats[passed_filter].index

    # 只保留这些基因的数据
    filtered_df = full_df[full_df['Gene'].isin(final_gene_list)].copy()

    # 按照原始 genes 列表的顺序排序，保证绘图时的顺序
    filtered_df['Gene'] = pd.Categorical(filtered_df['Gene'], categories=[g for g in genes if g in final_gene_list])
    filtered_df = filtered_df.sort_values(['Cluster', 'Gene'])

    print(f"输入基因: {len(genes)}, 强制保留: {len(keep_genes)}, 最终剩余: {len(final_gene_list)}")

    return filtered_df


def fan_dotplot(df, start_deg=-60, end_deg=60, figsize=(14, 10),
                cmap='RdYlBu_r', grid_lw=0.8, grid_color='lightgray',
                title=None, save_path=None):
    """
    绘制高度可定制的扇形 DotPlot

    参数:
    -----------
    df : pd.DataFrame
        包含 'Gene', 'Cluster', 'AvgExp', 'Pct' 列
    start_deg, end_deg : int
        扇形的起始和结束角度（0度为正北/12点方向）
    figsize : tuple
        画布尺寸
    cmap : str
        颜色映射方案，如 'viridis', 'magma', 'RdYlBu_r', 'plasma' 等
    grid_lw : float
        网格线的粗细 (linewidth)
    grid_color : str
        网格线的颜色
    title : str, optional
        图表标题
    save_path : str, optional
        保存路径（支持自动创建文件夹）
    """
    # 1. 准备基础数据
    genes = df['Gene'].unique()
    clusters = df['Cluster'].unique()

    theta_pos = np.linspace(np.radians(start_deg), np.radians(end_deg), len(genes))
    inner_r = 10
    r_pos = np.arange(len(clusters)) + inner_r

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')

    feat_map = {f: t for f, t in zip(genes, theta_pos)}
    grp_map = {g: r for g, r in zip(clusters, r_pos)}

    # 2. 绘制散点 (Dot Plot)
    sc = ax.scatter(
        df['Gene'].map(feat_map),
        df['Cluster'].map(grp_map),
        s=df['Pct'] * 2.5,  # 点的大小根据百分比调整
        c=df['AvgExp'],
        cmap=cmap,         # 使用传入的 cmap
        alpha=1.0,
        edgecolors='white',
        linewidth=0.5,
        zorder=10,
        clip_on=False
    )

    # 3. 设置极坐标系外观
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # 设置显示的角度范围
    padding_deg = 5
    ax.set_thetamin(start_deg - padding_deg)
    ax.set_thetamax(end_deg + padding_deg)

    # 隐藏默认边框和坐标轴
    ax.set_frame_on(False)
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # 4. 手动绘制背景网格 (Manual Grid Controls)
    ax.set_axisbelow(True)
    grid_style = dict(linestyle='--', color=grid_color, alpha=0.6, linewidth=grid_lw, zorder=1)

    # A. 径向线（圆弧，区分不同 Cluster）
    for r in r_pos:
        x = np.linspace(np.radians(start_deg), np.radians(end_deg), 100)
        y = np.full_like(x, r)
        ax.plot(x, y, **grid_style)

    # B. 角度线（直线，区分不同 Gene）
    for theta in theta_pos:
        x = [theta, theta]
        y = [inner_r, r_pos[-1]]
        ax.plot(x, y, **grid_style)

    # 隐藏刻度标签
    ax.set_xticks(theta_pos)
    ax.set_yticks(r_pos)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 5. 绘制标签 (Labels Logic)
    def calc_radial_params(theta_rad):
        angle_deg = np.degrees(theta_rad)
        rot = 90 - angle_deg
        if 90 < rot <= 270: rot -= 180; ha = 'right'
        elif rot > 270: rot -= 360; ha = 'left'
        elif rot < -90: rot += 180; ha = 'right'
        else: ha = 'left'
        return rot, ha

    # 绘制基因标签
    for theta, label in zip(theta_pos, genes):
        rot, ha = calc_radial_params(theta)
        ax.text(theta, r_pos[-1] + 1.2, label,
                rotation=rot, rotation_mode='anchor',
                va='center', ha=ha, fontsize=9, zorder=11, clip_on=False)

    # 绘制 Cluster/Sample 标签
    edge_theta = np.radians(start_deg)
    sample_rot = -start_deg
    for r, label in zip(r_pos, clusters):
        ax.text(edge_theta - 0.1, r, label,
                rotation=sample_rot,
                rotation_mode='anchor',
                va='center', ha='right',
                fontweight='bold', fontsize=10, zorder=11, clip_on=False)

    # 6. 颜色条 (Colorbar)
    cbar = plt.colorbar(sc, shrink=0.4, pad=0.1)
    cbar.set_label('Avg Expression')

    # 7. 标题
    if title:
        ax.set_title(title, pad=60, fontsize=16, fontweight='bold')

    # 8. 保存与展示
    if save_path:
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()
