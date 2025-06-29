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
pcd_path = "/Users/captain/workspace/MAPS_sample/data/752f05ba342c450bb9caadc099dbf192/lidar.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# Hilbert 排序
sorted_idx = hilbert_sort_2d(points)
points_hilbert = points[sorted_idx]

# 统一渐变色（Turbo colormap）
colormap = plt.get_cmap("turbo")
colors = colormap(np.linspace(0, 1, len(points)))[:, :3]


# 原始点云（原始顺序 + 渐变色）
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(points)
pcd_raw.colors = o3d.utility.Vector3dVector(colors)

line_indices_raw = [[i, i + 1] for i in range(min(999, len(points) - 1))]
lines_raw = o3d.geometry.LineSet()
lines_raw.points = o3d.utility.Vector3dVector(points)
lines_raw.lines = o3d.utility.Vector2iVector(line_indices_raw)
lines_raw.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(line_indices_raw))

# Hilbert 排序后点云（重新排序 + 对应颜色）
pcd_hilbert = o3d.geometry.PointCloud()
pcd_hilbert.points = o3d.utility.Vector3dVector(points_hilbert)
pcd_hilbert.colors = o3d.utility.Vector3dVector(colors[sorted_idx])

line_indices_hilbert = [[i, i + 1] for i in range(min(99999,len(points_hilbert) - 1))]
lines_hilbert = o3d.geometry.LineSet()
lines_hilbert.points = o3d.utility.Vector3dVector(points_hilbert)
lines_hilbert.lines = o3d.utility.Vector2iVector(line_indices_hilbert)
lines_hilbert.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(line_indices_hilbert))

# 分两个窗口显示
print("显示原始点云...")
o3d.visualization.draw_geometries(
    [pcd_raw, lines_raw],
    window_name="Original Point Cloud with Turbo Color"
)

print("显示 Hilbert 排序点云...")
o3d.visualization.draw_geometries(
    [pcd_hilbert, lines_hilbert],
    window_name="Hilbert Sorted Point Cloud with Turbo Color"
)
