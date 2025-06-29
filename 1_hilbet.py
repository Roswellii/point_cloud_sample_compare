import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert 排序函数（可根据2D投影实现）
def hilbert_sort_2d(points):
    proj_points = points[:, :2]  # 使用 X 和 Y
    min_coords = np.min(proj_points, axis=0)
    max_coords = np.max(proj_points, axis=0)
    normalized = (proj_points - min_coords) / (max_coords - min_coords + 1e-8)  # 防止除以0
    normalized = (normalized * 1023).astype(int)  # 映射到 10-bit 网格空间

    hilbert_curve = HilbertCurve(p=10, n=2)
    distances = [hilbert_curve.distance_from_point(coord.tolist()) for coord in normalized]
    sorted_idx = np.argsort(distances)
    return sorted_idx

# 加载点云
pcd_path = "/Users/captain/workspace/MAPS_sample/data/0fc8a9b2a41944a5bd2aac364b19b4ec/lidar.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# 原始顺序颜色映射
colormap = plt.get_cmap("turbo")
colors_orig = colormap(np.linspace(0, 1, len(points)))[:, :3]

# 显示原始顺序的点云
pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(points)
pcd_orig.colors = o3d.utility.Vector3dVector(colors_orig)
o3d.visualization.draw_geometries([pcd_orig], window_name="Original Order Visualization")

# Hilbert 排序
sorted_idx = hilbert_sort_2d(points)
points_hilbert = points[sorted_idx]
colors_hilbert = colormap(np.linspace(0, 1, len(points)))[:, :3]

# 显示 Hilbert 顺序的点云
pcd_hilbert = o3d.geometry.PointCloud()
pcd_hilbert.points = o3d.utility.Vector3dVector(points_hilbert)
pcd_hilbert.colors = o3d.utility.Vector3dVector(colors_hilbert)
o3d.visualization.draw_geometries([pcd_hilbert], window_name="Hilbert Order Visualization")
