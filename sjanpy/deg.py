import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from scipy import stats
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

def fast_two_group_deg(adata, label_col, lst1, lst2):
    """
    High-speed DEG calculation using vectorized operations.
    Focuses on raw matrix extraction and batch statistics.
    """
    # 1. Get indices for both groups
    idx1 = np.where(adata.obs[label_col].isin(lst1))[0]
    idx2 = np.where(adata.obs[label_col].isin(lst2))[0]
    
    # 2. Extract Data (avoiding adata subsetting overhead)
    # Using .X for speed; ensure adata.X is normalized/log-transformed as expected
    X1 = adata.X[idx1, :]
    X2 = adata.X[idx2, :]
    
    # 3. Vectorized Percentage Calculation
    if issparse(adata.X):
        pct1 = np.array((X1 > 0).mean(axis=0)).flatten()
        pct2 = np.array((X2 > 0).mean(axis=0)).flatten()
        # Convert to dense for stats if memory allows, or use sparse-friendly stats
        m1 = np.array(X1.mean(axis=0)).flatten()
        m2 = np.array(X2.mean(axis=0)).flatten()
        v1 = np.array(X1.power(2).mean(axis=0)).flatten() - m1**2
        v2 = np.array(X2.power(2).mean(axis=0)).flatten() - m2**2
    else:
        pct1 = (X1 > 0).mean(axis=0)
        pct2 = (X2 > 0).mean(axis=0)
        m1 = X1.mean(axis=0)
        m2 = X2.mean(axis=0)
        v1 = X1.var(axis=0)
        v2 = X2.var(axis=0)

    # 4. Calculate Log2FC (Assuming data is already log-transformed)
    # If data is log1p transformed: log2FC = (m1 - m2) / log(2)
    log2FC = (m1 - m2) / np.log(2)

    # 5. Fast Welch's T-Test (Vectorized)
    # This is much faster than Wilcoxon for large datasets
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat = (m1 - m2) / np.sqrt(v1/len(idx1) + v2/len(idx2))
        df = (v1/len(idx1) + v2/len(idx2))**2 / (
            (v1/len(idx1))**2 / (len(idx1)-1) + (v2/len(idx2))**2 / (len(idx2)-1)
        )
        pvals = stats.t.sf(np.abs(t_stat), df) * 2
        pvals = np.nan_to_num(pvals, nan=1.0)

    # 6. Multiple Testing Correction
    padj = multipletests(pvals, method='fdr_bh')[1]

    # 7. Construct Result
    res = pd.DataFrame({
        'gene': adata.var_names,
        'log2FC': log2FC,
        'pct.1': pct1,
        'pct.2': pct2,
        'pval': pvals,
        'padj': padj
    }).sort_values('padj')
    return res


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


from scipy import sparse

def compute_nested_deg_df(
    adata,
    cluster_key,
    condition_key,
    target_condition,
    reference_condition,
    method="wilcoxon",
    min_cells=10,
    compute_pct=True,
    expr_layer=None,      # e.g. "log1p" or "counts"; None -> adata.X
    expr_threshold=0.0,   # > threshold means "expressed"
):
    """
    Compute within-cluster differential expression between two conditions, and
    optionally report per-gene detection rate (fraction of cells expressing)
    in each condition.

    For each cluster defined by `cluster_key`, the function subsets `adata` to
    that cluster and runs `scanpy.tl.rank_genes_groups` comparing
    `target_condition` vs `reference_condition` within `condition_key`.
    Clusters are skipped if either group has fewer than `min_cells`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Uses `adata.X` by default for detection-rate
        computation; DEG testing is performed by Scanpy on the subset AnnData.
    cluster_key : str
        Column name in `adata.obs` defining clusters to iterate over.
    condition_key : str
        Column name in `adata.obs` defining the two conditions to compare.
    target_condition : str
        Name of the condition treated as the numerator / "case" group in the
        comparison (e.g., "Disease", "Adult").
    reference_condition : str
        Name of the condition treated as the reference / "control" group in the
        comparison (e.g., "Normal", "Fetal").
    method : str, default "wilcoxon"
        Method passed to `scanpy.tl.rank_genes_groups` (e.g., "wilcoxon",
        "t-test", "logreg").
    min_cells : int, default 10
        Minimum number of cells required in *each* condition within a cluster.
        Clusters not meeting this threshold are skipped.
    compute_pct : bool, default True
        If True, add detection-rate columns:
        - `pct_target`: fraction of target-condition cells with expression
          > `expr_threshold`
        - `pct_reference`: fraction of reference-condition cells with expression
          > `expr_threshold`
    expr_layer : str or None, default None
        Which matrix to use for detection-rate computation:
        - None: use `adata_c.X`
        - str : use `adata_c.layers[expr_layer]`
        This does not change the DEG test itself (handled by Scanpy).
    expr_threshold : float, default 0.0
        Expression threshold for defining a gene as "expressed" when computing
        detection rates. A gene is counted as expressed in a cell if its value
        is strictly greater than this threshold.

    Returns
    -------
    pandas.DataFrame
        Concatenated results across clusters with one row per ranked gene per
        cluster. Always includes:
        - `gene` : str
        - `logfc` : float
        - `pvals_adj` : float
        - `cluster` : str
        If `compute_pct=True`, also includes:
        - `pct_target` : float in [0, 1]
        - `pct_reference` : float in [0, 1]

    Notes
    -----
    - Detection rates are computed over all genes in the subset cluster matrix
      and then indexed to the ranked genes returned by Scanpy.
    - If `expr_layer` is provided, it must exist in `adata.layers`.
    - For sparse matrices, detection rates are computed efficiently without
      densifying.
    """

    def _get_X(a):
        if expr_layer is None:
            return a.X
        if expr_layer not in a.layers:
            raise KeyError(f"expr_layer='{expr_layer}' not found in adata.layers")
        return a.layers[expr_layer]

    def _pct_expressing(X, idx, thr):
        # X: cells x genes, idx: boolean mask over cells
        Xg = X[idx]
        if sparse.issparse(Xg):
            # (Xg > thr) yields sparse bool matrix
            frac = (Xg > thr).mean(axis=0)
            return np.asarray(frac).ravel()
        return (Xg > thr).mean(axis=0)

    all_degs = []
    clusters = adata.obs[cluster_key].unique()

    for cluster in clusters:
        adata_c = adata[adata.obs[cluster_key] == cluster].copy()

        counts = adata_c.obs[condition_key].value_counts()
        ok = (
            target_condition in counts
            and reference_condition in counts
            and counts[target_condition] >= min_cells
            and counts[reference_condition] >= min_cells
        )
        if not ok:
            print(f"Skipping cluster: {cluster} (Insufficient cells)")
            continue

        print(
            f"Processing cluster: {cluster} "
            f"({counts[target_condition]} vs {counts[reference_condition]} cells)"
        )

        sc.tl.rank_genes_groups(
            adata_c,
            groupby=condition_key,
            groups=[target_condition],
            reference=reference_condition,
            method=method,
        )

        result = adata_c.uns["rank_genes_groups"]
        genes = np.array(result["names"][target_condition], dtype=str)

        df = pd.DataFrame(
            {
                "gene": genes,
                "logfc": np.array(result["logfoldchanges"][target_condition]),
                "pvals_adj": np.array(result["pvals_adj"][target_condition]),
                "cluster": cluster,
            }
        )

        if compute_pct:
            X = _get_X(adata_c)
            mask_t = (adata_c.obs[condition_key].to_numpy() == target_condition)
            mask_r = (adata_c.obs[condition_key].to_numpy() == reference_condition)

            pct_t_all = _pct_expressing(X, mask_t, expr_threshold)
            pct_r_all = _pct_expressing(X, mask_r, expr_threshold)

            # map gene -> var index, then pick only DE genes
            gene_to_idx = pd.Index(adata_c.var_names).get_indexer(genes)
            valid = gene_to_idx >= 0
            if not valid.all():
                # shouldn't happen, but keep robust
                df.loc[~valid, "pct_target"] = np.nan
                df.loc[~valid, "pct_reference"] = np.nan
                df.loc[valid, "pct_target"] = pct_t_all[gene_to_idx[valid]]
                df.loc[valid, "pct_reference"] = pct_r_all[gene_to_idx[valid]]
            else:
                df["pct_target"] = pct_t_all[gene_to_idx]
                df["pct_reference"] = pct_r_all[gene_to_idx]

        all_degs.append(df)

    if not all_degs:
        return pd.DataFrame()

    return pd.concat(all_degs, ignore_index=True)

# def compute_nested_deg_df(
#     adata, 
#     cluster_key, 
#     condition_key, 
#     target_condition, 
#     reference_condition,
#     method='wilcoxon', 
#     min_cells=10
# ):
#     """
#     Computes DEGs between two conditions within each cluster.
     
#     Parameters:
#     -----------
#     adata : AnnData
#     cluster_key : str
#         The column in adata.obs defining clusters (e.g., 'C0', 'C1').
#     condition_key : str
#         The column in adata.obs defining conditions (e.g., 'Status').
#     target_condition : str
#         The 'Disease' or 'Adult' group.
#     reference_condition : str
#         The 'Normal' or 'Fetal' group.
#     """
    
#     all_degs = []
#     clusters = adata.obs[cluster_key].unique()

#     for cluster in clusters:
#         # 1. Subset to the specific cluster
#         adata_c = adata[adata.obs[cluster_key] == cluster].copy()
        
#         # 2. Check if both conditions exist and have enough cells
#         counts = adata_c.obs[condition_key].value_counts()
        
#         if (target_condition in counts and reference_condition in counts and 
#             counts[target_condition] >= min_cells and 
#             counts[reference_condition] >= min_cells):
            
#             print(f"Processing cluster: {cluster} ({counts[target_condition]} vs {counts[reference_condition]} cells)")
            
#             # 3. Run DEG for this cluster
#             sc.tl.rank_genes_groups(
#                 adata_c, 
#                 groupby=condition_key, 
#                 groups=[target_condition], 
#                 reference=reference_condition, 
#                 method=method
#             )
            
#             # 4. Extract results
#             result = adata_c.uns['rank_genes_groups']
#             df = pd.DataFrame({
#                 'gene': result['names'][target_condition],
#                 'logfc': result['logfoldchanges'][target_condition],
#                 'pvals_adj': result['pvals_adj'][target_condition],
#                 'cluster': cluster
#             })
#             all_degs.append(df)
#         else:
#             print(f"Skipping cluster: {cluster} (Insufficient cells)")

#     if len(all_degs) == 0:
#         return pd.DataFrame()

#     return pd.concat(all_degs).reset_index(drop=True)


def clip_logfc_in_nested_deg_df(df, logfc_col='logfc', cluster_col='cluster', quantile=0.95):
    """
    按 Cluster 对 logfc 进行分位数裁剪 (Clipping)
    """
    
    def _apply_clip(group):
        lfc = group[logfc_col]
        
        # --- 计算 Max Clip (正数部分) ---
        max_val = lfc.max()
        if max_val <= 1:
            max_clip = 1  # 设置为1，因为原值都不超过1，实际上不会发生裁剪
        else:
            pos_vals = lfc[lfc > 0]
            # 如果存在正值则计算95分位数，否则默认1
            max_clip = pos_vals.quantile(quantile) if not pos_vals.empty else 1
            
        # --- 计算 Min Clip (负数部分) ---
        min_val = lfc.min()
        if min_val >= -1:
            min_clip = -1 # 设置为-1，因为原值都比-1大，实际上不会发生裁剪
        else:
            neg_vals = lfc[lfc < 0]
            # 如果存在负值则计算5分位数，否则默认-1
            min_clip = neg_vals.quantile(1-quantile) if not neg_vals.empty else -1
            
        # 执行裁剪并返回
        group[logfc_col] = lfc.clip(lower=min_clip, upper=max_clip)
        return group

    # 使用 groupby 并在组内应用逻辑
    # group_keys=False 确保返回的索引与原数据框一致
    return df.groupby(cluster_col, group_keys=False).apply(_apply_clip)

import pandas as pd
import re

def generate_highlight_dict(
    deg_df, 
    strategies=['topn'], 
    cluster_key='cluster', 
    top_n=5, 
    k=3, 
    ktimes_poscut=1.0, 
    ktimes_negcut=-1.0, 
    manual_genes=None,
    exclude_genes=None,
    exclude_regex=None
):
    """
    根据多种策略生成每个 Cluster 需要 highlight 的基因字典，支持正则表达式排除。
    
    正则表达式示例:
    -----------
    1. 排除线粒体基因 (以 MT- 开头): r'^MT-'
    2. 排除核糖体基因 (以 RPS 或 RPL 开头): r'^RP[SL]'
    3. 排除特定模式的基因 (如 AC 后面跟数字，以 .1 结尾): r'^AC\d+\.1$'
    4. 排除所有以 Gm 开头的基因: r'^Gm'
    5. 同时排除多种 (使用 | 分隔): r'^MT-|^RP[SL]|^Gm'

    参数:
    -----------
    deg_df : pd.DataFrame
        差异分析结果，包含 gene, logfc, cluster 等列。
    exclude_regex : list of str
        正则表达式列表。任何匹配其中一个正则的基因都将被排除。
    """
    
    # 初始化字典
    highlight_dict = {cluster: set() for cluster in deg_df[cluster_key].unique()}
    clusters = list(highlight_dict.keys())

    # --- 1. 选取逻辑 (Population Phase) ---

    if 'manual' in strategies and manual_genes:
        manual_set = set(manual_genes)
        for cluster in clusters:
            subset_genes = set(deg_df[deg_df[cluster_key] == cluster]['gene'])
            highlight_dict[cluster].update(manual_set.intersection(subset_genes))

    if 'ktimes' in strategies:
        # 严格依据阈值判断共同响应
        up_df = deg_df[deg_df['logfc'] >= ktimes_poscut]
        up_counts = up_df['gene'].value_counts()
        ktimes_up_genes = set(up_counts[up_counts >= k].index)
        
        down_df = deg_df[deg_df['logfc'] <= ktimes_negcut]
        down_counts = down_df['gene'].value_counts()
        ktimes_down_genes = set(down_counts[down_counts >= k].index)
        
        for cluster in clusters:
            cluster_subset = deg_df[deg_df[cluster_key] == cluster]
            current_up = set(cluster_subset[cluster_subset['logfc'] >= ktimes_poscut]['gene'])
            highlight_dict[cluster].update(ktimes_up_genes.intersection(current_up))
            
            current_down = set(cluster_subset[cluster_subset['logfc'] <= ktimes_negcut]['gene'])
            highlight_dict[cluster].update(ktimes_down_genes.intersection(current_down))

    if 'topn' in strategies:
        for cluster in clusters:
            subset = deg_df[deg_df[cluster_key] == cluster]
            # 选取 logFC 极值基因
            top_up = subset.nlargest(top_n, 'logfc')['gene'].tolist()
            top_down = subset.nsmallest(top_n, 'logfc')['gene'].tolist()
            highlight_dict[cluster].update(top_up + top_down)

    # --- 2. 排除逻辑 (Exclusion Phase: Regex & List) ---
    
    exclude_set = set(exclude_genes) if exclude_genes else set()
    
    # 预编译正则对象提高效率
    regex_objs = [re.compile(r) for r in exclude_regex] if exclude_regex else []

    filtered_dict = {}
    for cluster, genes in highlight_dict.items():
        valid_genes = []
        for g in genes:
            # 检查是否在排除列表中
            if g in exclude_set:
                continue
            # 检查是否匹配任何正则表达式
            if any(r.search(g) for r in regex_objs):
                continue
            valid_genes.append(g)
            
        filtered_dict[cluster] = valid_genes

    return filtered_dict

    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text

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