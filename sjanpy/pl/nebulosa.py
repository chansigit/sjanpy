import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.sparse import issparse


def silverman_bandwidth(data):
    """
    根据 Silverman 规则计算带宽：
    h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    """
    n = len(data)
    std = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    return 0.9 * min(std, iqr / 1.34) * n ** (-1/5)


def wkde2d(x, y, w, adjust=1, n=100, lims=None):
    """
    加权二维核密度估计（Weighted KDE）

    参数:
      x, y: 数据点的二维坐标（向量）
      w: 对应的权重（例如基因表达值）
      adjust: 带宽调整因子
      n: 网格大小（在 x 与 y 方向上各取 n 个点）
      lims: [xmin, xmax, ymin, ymax]，若为 None 则根据数据自动计算

    返回:
      gx, gy: 网格在 x 与 y 方向上的坐标数组
      z: 得到的密度矩阵，大小为 (n, n)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)
    if len(x) != len(y) or len(x) != len(w):
        raise ValueError("x, y, and w must have the same length")
    if lims is None:
        lims = [np.min(x), np.max(x), np.min(y), np.max(y)]

    h_x = silverman_bandwidth(x) * adjust
    h_y = silverman_bandwidth(y) * adjust

    gx = np.linspace(lims[0], lims[1], n)
    gy = np.linspace(lims[2], lims[3], n)

    # 构建网格上每个点与数据点距离（标准化后的差值）
    ax = (gx[:, None] - x[None, :]) / h_x  # shape: (n, N)
    ay = (gy[:, None] - y[None, :]) / h_y  # shape: (n, N)

    pdf_ax = norm.pdf(ax)
    pdf_ay = norm.pdf(ay)

    # 构建权重矩阵，每行复制 w
    w_mat = np.tile(w, (n, 1))

    # 分别乘以核函数值
    A = pdf_ax * w_mat
    B = pdf_ay * w_mat

    # 计算密度矩阵，对每个网格点 (gx, gy)，汇总所有数据点的加权贡献
    z = np.dot(A, B.T) / (np.sum(w) * h_x * h_y)
    return gx, gy, z


def wkde3d(x, y, z, w, adjust=1, n=30, lims=None):
    """
    加权三维核密度估计

    参数:
      x, y, z: 数据点的三维坐标（向量，长度 N）
      w: 对应的权重（例如基因表达值）
      adjust: 带宽调整因子
      n: 网格数（每个维度上生成 n 个网格点）
      lims: [xmin, xmax, ymin, ymax, zmin, zmax]；若为 None，则自动根据数据范围确定

    返回:
      gx, gy, gz: 三个方向上网格的坐标数组
      Z: 得到的密度值三维数组，形状为 (n, n, n)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)

    if lims is None:
        lims = [np.min(x), np.max(x),
                np.min(y), np.max(y),
                np.min(z), np.max(z)]

    h_x = silverman_bandwidth(x) * adjust
    h_y = silverman_bandwidth(y) * adjust
    h_z = silverman_bandwidth(z) * adjust

    gx = np.linspace(lims[0], lims[1], n)
    gy = np.linspace(lims[2], lims[3], n)
    gz = np.linspace(lims[4], lims[5], n)

    # 对每个维度计算正态核函数值
    pdf_x = norm.pdf((gx[:, None] - x[None, :]) / h_x)  # shape: (n, N)
    pdf_y = norm.pdf((gy[:, None] - y[None, :]) / h_y)  # shape: (n, N)
    pdf_z = norm.pdf((gz[:, None] - z[None, :]) / h_z)  # shape: (n, N)

    # 利用 einsum 对三个维度进行乘积，并对所有数据点求和
    Z = np.einsum('xi,yi,zi,i->xyz', pdf_x, pdf_y, pdf_z, w)

    # 归一化：除以总权重及各方向带宽
    Z = Z / (np.sum(w) * h_x * h_y * h_z)

    return gx, gy, gz, Z


def nebulosa_density(adata, coord_key, gene, adjust=1, n=100, lims=None, cmap='viridis', show=False):
    """
    在 AnnData 对象中，根据 obsm 中的二维坐标和基因表达值计算加权二维 KDE，
    并绘制散点图（颜色表示密度值）。

    参数:
      adata: AnnData 对象
      coord_key: 用于提取散点坐标的 obsm 键（例如 "X_umap"），要求形状为 (n_cells, 2)
      gene: 用于加权的基因名称（必须存在于 adata.var_names 中）
      adjust: 带宽调整因子（默认 1）
      n: 网格大小（默认 100）
      lims: [xmin, xmax, ymin, ymax]，如果为 None，则自动计算数据范围
      cmap: 绘图使用的颜色图（默认 'viridis'）
      show: 是否调用 plt.show() 显示图形（默认 False）

    返回:
      densities (ndarray) if show=False, else (fig, ax)
    """
    # 提取二维坐标
    if coord_key not in adata.obsm.keys():
        raise KeyError(f"{coord_key} 不存在于 adata.obsm 中")
    coords = adata.obsm[coord_key]
    if coords.shape[1] < 2:
        raise ValueError("提取的坐标必须至少包含两列")
    # 只取前两列
    coords = coords[:, :2]
    x = coords[:, 0]
    y = coords[:, 1]

    # 检查基因是否存在
    if gene not in adata.var_names:
        raise KeyError(f"基因 {gene} 不存在于 adata.var_names 中")
    # 提取对应的基因表达值，确保是一维数组
    expr = adata[:, gene].X
    if issparse(expr):
        expr = expr.toarray().flatten()
    else:
        expr = np.array(expr).flatten()

    # 计算加权 KDE
    gx, gy, z = wkde2d(x, y, expr, adjust=adjust, n=n, lims=lims)

    # 将每个原始数据点映射到对应网格上，得到密度值
    ix = np.digitize(x, gx) - 1
    iy = np.digitize(y, gy) - 1
    ix = np.clip(ix, 0, len(gx) - 1)
    iy = np.clip(iy, 0, len(gy) - 1)
    densities = z[ix, iy]
    if not show:
        return densities

    # 绘制散点图，颜色映射密度值
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=densities, cmap=cmap, s=30)
    plt.colorbar(sc, ax=ax, label='Weighted KDE Density', shrink=0.3)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'Weighted KDE Density for {gene}')
    plt.grid(False)
    plt.show()
    return fig, ax
