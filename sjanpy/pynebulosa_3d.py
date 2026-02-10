import numpy as np
import plotly.express as px
from scipy.stats import norm

def silverman_bandwidth(data):
    """
    根据 Silverman 规则计算带宽：
    h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    """
    n = len(data)
    std = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    return 0.9 * min(std, iqr / 1.34) * n ** (-1/5)

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

# =======================
# 模拟三维数据
# =======================
np.random.seed(42)
N = 500

# 生成两簇数据：一簇在 (-2, -2, -2)，另一簇在 (2, 2, 2)
x = np.concatenate([np.random.normal(loc=-2, scale=1.0, size=N//2),
                    np.random.normal(loc=2, scale=1.0, size=N//2)])
y = np.concatenate([np.random.normal(loc=-2, scale=1.0, size=N//2),
                    np.random.normal(loc=2, scale=1.0, size=N//2)])
z = np.concatenate([np.random.normal(loc=-2, scale=1.0, size=N//2),
                    np.random.normal(loc=2, scale=1.0, size=N//2)])
# 模拟表达值（权重）：用离原点距离的指数函数，离中心近的点权重较高
w = np.exp(-((x**2 + y**2 + z**2) / 15))

# =======================
# 计算加权三维 KDE
# =======================
gx, gy, gz, Z = wkde3d(x, y, z, w, adjust=1, n=30)

# 将每个原始数据点映射到对应网格上，获取密度值
ix = np.digitize(x, gx) - 1
iy = np.digitize(y, gy) - 1
iz = np.digitize(z, gz) - 1
ix = np.clip(ix, 0, len(gx)-1)
iy = np.clip(iy, 0, len(gy)-1)
iz = np.clip(iz, 0, len(gz)-1)
densities = Z[ix, iy, iz]

# =======================
# 使用 Plotly 绘制交互式 3D 散点图
# =======================
fig = px.scatter_3d(
    x=x, y=y, z=z,
    color=densities,
    color_continuous_scale='viridis',
    title='3D Weighted KDE on Simulated Data',
    labels={'x': 'X', 'y': 'Y', 'z': 'Z', 'color': 'Density'},
)
fig.show()


# =======================
# 3D 可视化：用颜色显示每个点的密度
# =======================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=densities, cmap='viridis', s=30)
fig.colorbar(sc, ax=ax, label='Weighted KDE Density')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Weighted KDE on Simulated Data')
plt.show()

