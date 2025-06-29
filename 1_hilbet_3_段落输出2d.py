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

# 创建某一段的连线
def create_line_segment(start, end, max_len):
    line_indices = []
    for i in range(start, min(end, max_len - 1)):
        line_indices.append([i, i + 1])
    return line_indices

# 加载点云
pcd_path = "./data/752f05ba342c450bb9caadc099dbf192/lidar.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

# Hilbert 排序
sorted_idx = hilbert_sort_2d(points)
points_hilbert = points[sorted_idx]

# 统一渐变色（Turbo colormap）
colormap = plt.get_cmap("turbo")
colors = colormap(np.linspace(0, 1, len(points)))[:, :3]

# Hilbert 排序点云（全局，只画一次）
pcd_hilbert = o3d.geometry.PointCloud()
pcd_hilbert.points = o3d.utility.Vector3dVector(points_hilbert)
pcd_hilbert.colors = o3d.utility.Vector3dVector(colors[sorted_idx])

# 指定显示的区间
ranges = [(0, 1000), (4000, 5000)]

# 每个区间单独显示
for idx, (start, end) in enumerate(ranges):
    line_indices = create_line_segment(start, end, len(points_hilbert))
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points_hilbert)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    lines.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(line_indices))

    print(f"显示 Hilbert 排序点云的区间 {start}-{end}...")
    o3d.visualization.draw_geometries(
        [pcd_hilbert, lines],
        window_name=f"Hilbert Segment {start}-{end}"
    )
