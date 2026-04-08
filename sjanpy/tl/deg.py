import re

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats, sparse
from scipy.sparse import issparse
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
