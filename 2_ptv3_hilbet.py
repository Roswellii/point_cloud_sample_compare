# process_existing_pcd_fixed.py

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import os


# ==============================================================================
# 代码来源: hilbert.py (Numpy版本)
# 核心Hilbert曲线编码/解码算法
# ==============================================================================

def right_shift(binary, k=1, axis=-1):
    """向右二进制移位 (Numpy实现)"""
    if binary.shape[axis] <= k:
        return np.zeros_like(binary)

    pad_width = [(0, 0)] * binary.ndim
    pad_width[axis] = (k, 0)

    slicing = [slice(None)] * binary.ndim
    slicing[axis] = slice(None, -k)

    shifted = np.pad(
        binary[tuple(slicing)], pad_width, mode="constant", constant_values=0
    )
    return shifted


def binary2gray(binary, axis=-1):
    """二进制转格雷码 (Numpy实现)"""
    shifted = right_shift(binary, k=1, axis=axis)
    gray = np.logical_xor(binary, shifted)
    return gray


def gray2binary(gray, axis=-1):
    """格雷码转二进制 (Numpy实现)"""
    num_bits = gray.shape[axis]
    power_of_2 = 2 ** (int(np.ceil(np.log2(num_bits))) - 1)

    shift = power_of_2
    while shift > 0:
        gray = np.logical_xor(gray, right_shift(gray, shift, axis=axis))
        shift //= 2
    return gray


def hilbert_encode_(locs, num_dims, num_bits):
    """將超立方體中的位置編碼為希爾伯特整數 (Numpy實現)"""
    if locs.shape[-1] != num_dims:
        raise ValueError(f"位置的最后一个维度是 {locs.shape[-1]}, 但 num_dims={num_dims}。它们需要相等。")
    if num_dims * num_bits > 64:
        raise ValueError(
            f"num_dims={num_dims} 和 num_bits={num_bits} 总共需要 {num_dims * num_bits} 位，无法编码为int64。")

    locs_uint8 = locs.astype(np.int64).view(np.uint8).reshape((-1, num_dims, 8))[:, :, ::-1]
    bitpack_mask_rev = (1 << np.arange(8, dtype=np.uint8))[::-1]
    gray = (np.bitwise_and(locs_uint8[..., np.newaxis], bitpack_mask_rev) != 0).astype(np.uint8)
    gray = gray.reshape(gray.shape[:-2] + (-1,))[..., -num_bits:]

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit].astype(bool)
            gray[:, 0, bit + 1:] = np.logical_xor(gray[:, 0, bit + 1:], mask[:, np.newaxis])
            to_flip = np.logical_and(
                np.logical_not(mask[:, np.newaxis]),
                np.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:])
            )
            gray[:, dim, bit + 1:] = np.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = np.logical_xor(gray[:, 0, bit + 1:], to_flip)

    gray = np.swapaxes(gray, 1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = gray2binary(gray)

    extra_dims = 64 - num_bits * num_dims
    padded = np.pad(hh_bin, ((0, 0), (extra_dims, 0)), 'constant')

    bitpack_mask = 1 << np.arange(8, dtype=np.uint8)
    hh_uint8 = np.sum(
        padded[:, ::-1].reshape((-1, 8, 8)) * bitpack_mask,
        axis=2
    ).astype(np.uint8)

    hh_uint64 = hh_uint8.view(np.int64).squeeze()
    return hh_uint64


# ==============================================================================
# 代码来源: default.py (部分) (Numpy版本)
# ==============================================================================

def hilbert_encode(grid_coord: np.ndarray, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def encode(grid_coord, batch=None, depth=16, order="hilbert"):
    if order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError

    if batch is not None:
        batch = batch.astype(np.int64)
        code = batch << (depth * 3) | code
    return code


# ==============================================================================
# 代码来源: structure.py (Numpy版本)
# ==============================================================================

class Point(dict):
    """Pointcept 的 Point 结构 (Numpy实现)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def serialization(self, order="hilbert", depth=None):
        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = np.trunc(
                self["coord"] / self["grid_size"]
            ).astype(np.int64)

        if depth is None:
            depth = int(self["grid_coord"].max() + 1).bit_length()
        self["serialized_depth"] = depth
        assert depth <= 16, "深度应小于等于16"

        order_list = [order]
        #  vvvvvvvvvvvvvvvvvvv   这里是修复的地方  vvvvvvvvvvvvvvvvvv
        #  将 'order_=o' 改为 'order=o'
        code = [
            encode(self["grid_coord"], self["batch"], depth, order=o) for o in order_list
        ]
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        code = np.stack(code)
        order_indices = np.argsort(code)

        self["serialized_code"] = code
        self["serialized_order"] = order_indices


# ==============================================================================
# 主执行逻辑: 读取PCD，排序，然后保存
# ==============================================================================

def main():
    """主函数"""
    # --- 用户配置 ---
    # 请将此路径更改为您要处理的PCD文件的实际路径
    # 例如: "C:/Users/YourUser/Desktop/my_cloud.pcd" 或 "/home/user/data/cloud.pcd"
    pcd_input_path = "data/0fc8a9b2a41944a5bd2aac364b19b4ec/lidar.pcd"
    # ----------------

    # 检查输入路径是否为示例路径
    if "path/to/your/point_cloud.pcd" in pcd_input_path:
        print("错误: 请在脚本中修改 'pcd_input_path' 变量，使其指向您自己的PCD文件。")
        sys.exit(1)

    # 1. 读取现有的PCD文件
    print(f"正在尝试从 '{pcd_input_path}' 读取点云...")
    if not os.path.exists(pcd_input_path):
        print(f"错误: 文件不存在 -> '{pcd_input_path}'")
        sys.exit(1)

    try:
        pcd = o3d.io.read_point_cloud(pcd_input_path)
        if not pcd.has_points():
            print(f"错误: 文件 '{pcd_input_path}' 已加载，但不包含任何点。")
            sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件时发生未知错误 '{pcd_input_path}'。")
        print(f"详细错误: {e}")
        sys.exit(1)

    print(f"成功从 '{pcd_input_path}' 加载了 {len(pcd.points)} 个点。")

    # 2. 准备数据
    coords_np = np.asarray(pcd.points)
    coords_min = coords_np.min(axis=0)
    coords_max = coords_np.max(axis=0)
    # 处理分母为0的情况
    scale = coords_max - coords_min
    scale[scale == 0] = 1.0
    coords_normalized = (coords_np - coords_min) / scale
    batch_np = np.zeros(coords_np.shape[0], dtype=np.int64)

    # 3. 初始化Point对象并执行序列化
    depth = 10
    grid_size = 1.0 / (2 ** depth)

    point_data = Point(
        coord=coords_normalized,
        batch=batch_np,
        grid_size=grid_size,
    )

    print(f"正在使用 Hilbert 曲线进行排序... (depth={depth})")
    point_data.serialization(order="hilbert", depth=depth)

    # 4. 获取排序后的索引并重新排列点
    hilbert_order_indices = point_data["serialized_order"][0]
    sorted_coords_np = coords_np[hilbert_order_indices]

    # 5. 为排序后的点创建颜色图以进行可视化
    print("正在为排序后的点生成颜色图...")
    cmap = plt.get_cmap("viridis")
    colors_sorted = cmap(np.linspace(0, 1, len(sorted_coords_np)))[:, :3]

    # 6. 创建一个新的Open3D点云对象并保存
    sorted_pcd = o3d.geometry.PointCloud()
    sorted_pcd.points = o3d.utility.Vector3dVector(sorted_coords_np)
    sorted_pcd.colors = o3d.utility.Vector3dVector(colors_sorted)

    # 定义输出文件路径
    input_dir, input_filename = os.path.split(pcd_input_path)
    input_name, input_ext = os.path.splitext(input_filename)
    pcd_output_path = os.path.join(input_dir, f"{input_name}_hilbert_sorted.pcd")

    o3d.io.write_point_cloud(pcd_output_path, sorted_pcd)
    print(f"排序完成！结果已保存到 '{pcd_output_path}'。")

    # 7. 使用o3d进行可视化
    print("按 'q' 关闭可视化窗口。")
    o3d.visualization.draw_geometries([sorted_pcd], window_name="Hilbert Sorted Point Cloud (Numpy)")


if __name__ == "__main__":
    main()