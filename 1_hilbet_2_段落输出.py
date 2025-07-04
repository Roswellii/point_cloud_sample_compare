import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert 排序函数
def hilbert_sort_2d(points):
    proj_points = points[:, :2]
    min_coords = np.min(proj_points, axis=0)
    max_coords = np.max(proj_points, axis=0)
    normalized = (proj_points - min_coords) / (max_coords - min_coords + 1e-8)
    normalized = (normalized * 1023).astype(int)
    hilbert_curve = HilbertCurve(p=10, n=2)
    distances = [hilbert_curve.distance_from_point(coord.tolist()) for coord in normalized]
    sorted_idx = np.argsort(distances)
    return sorted_idx

# 加载点云
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# Hilbert 排序
sorted_idx = hilbert_sort_2d(points)
points_hilbert = points[sorted_idx]

# 统一渐变色（Turbo colormap）
colormap = plt.get_cmap("turbo")
colors = colormap(np.linspace(0, 1, len(points)))[:, :3]

pcd_hilbert = o3d.geometry.PointCloud()
pcd_hilbert.points = o3d.utility.Vector3dVector(points_hilbert)
pcd_hilbert.colors = o3d.utility.Vector3dVector(colors[sorted_idx])


o3d.visualization.draw_geometries(
)
