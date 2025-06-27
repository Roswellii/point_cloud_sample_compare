import open3d as o3d
import numpy as np
import random
import matplotlib.pyplot as plt


# ==============================================================================
# 核心函數和輔助函數 (無變化)
# ==============================================================================

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def same_shuffl(arr1, arr2):
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]


def shuffle_idx(idx):
    """
    打亂索引數組。
    """
    np.random.shuffle(idx)
    return idx


# ==============================================================================
# polar_samplr 函數已更新，與訓練腳本的核心採樣策略完全一致
# ==============================================================================
def polar_samplr(mid_xyz, mid_gt, grid, fixed_volume_space=False):
    """
    核心採樣邏輯已與 semkitti_trainset.py 中的版本同步。
    """
    if mid_gt is None:
        mid_gt = np.zeros(len(mid_xyz))
    xyz_pol = cart2polar(mid_xyz)
    if fixed_volume_space:
        max_volume_space, min_volume_space = [50, np.pi, 1.5], [0, -np.pi, -3]
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)
    else:
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = max(np.percentile(xyz_pol[:, 0], 0, axis=0), 3)
        max_bound_p = np.max(xyz_pol[:, 1], axis=0)
        min_bound_p = np.min(xyz_pol[:, 1], axis=0)
        max_bound_z = min(np.max(xyz_pol[:, 2], axis=0), 1.5)
        min_bound_z = max(np.min(xyz_pol[:, 2], axis=0), -3)
        max_bound = np.concatenate(([max_bound_r], [max_bound_p], [max_bound_z]))
        min_bound = np.concatenate(([min_bound_r], [min_bound_p], [min_bound_z]))
    cur_grid_size = np.asarray(grid)
    crop_range = max_bound - min_bound
    intervals = crop_range / (cur_grid_size - 1)
    if (intervals == 0).any():
        intervals[intervals == 0] = 1e-6
    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int32)
    keys, revers, counts = np.unique(grid_ind, return_inverse=True, return_counts=True, axis=0)
    idx = np.argsort(revers)
    mid_xyz = mid_xyz[idx]
    slic = counts.cumsum()
    slic = np.insert(slic, 0, 0)
    left_xyz = np.zeros([0, 3])
    right_xyz = np.zeros([0, 3])
    target_count = len(mid_xyz) // 4

    # ========================= 此處為更新的核心邏輯 =========================
    # 識別稀疏網格 (點數<4)
    small = counts[counts < 4]
    # 計算需要從密集網格中採樣的總點數
    new_nums = target_count - sum(small)
    # 計算密集網格的數量
    new_grid = len(counts) - len(small)

    # 計算並隨機分配每個密集網格的採樣配額
    sample_list = []
    if new_grid > 0:
        for i in range(new_grid):
            # 平均分配配額
            curr = new_nums // new_grid
            sample_list.append(curr)
            new_nums -= curr
            new_grid -= 1
        sample_list = np.array(sample_list)
        # 打亂配額列表，實現隨機分配
        sample_list = shuffle_idx(sample_list)

    # 遍歷所有網格，執行分層採樣
    large_cell_idx = 0
    for i in range(len(counts)):
        select_xyz = mid_xyz[slic[i]:slic[i + 1]]
        select_xyz, _ = same_shuffl(select_xyz, select_xyz)
        nubs = counts[i]
        if nubs >= 4:  # 如果是密集網格
            # 從打亂的配額列表中獲取採樣數量
            downs_n = sample_list[large_cell_idx]
            large_cell_idx += 1
            left_xyz = np.concatenate((left_xyz, select_xyz[0:downs_n]), axis=0)
            right_xyz = np.concatenate((right_xyz, select_xyz[downs_n:]), axis=0)
        else:  # 如果是稀疏網格
            # 保留所有點
            left_xyz = np.concatenate((left_xyz, select_xyz), axis=0)
    # ========================= 核心邏輯更新結束 =========================

    # 執行與訓練腳本完全相同的最終平衡邏輯
    supp = target_count - len(left_xyz)
    if supp > 0 and len(right_xyz) > supp:
        right_xyz, _ = same_shuffl(right_xyz, right_xyz)
        left_xyz = np.concatenate((left_xyz, right_xyz[0:supp]))
    elif supp < 0:
        left_xyz, _ = same_shuffl(left_xyz, left_xyz)
        left_xyz = left_xyz[:target_count]

    left_xyz, _ = same_shuffl(left_xyz, left_xyz)
    return left_xyz, min_bound, max_bound


def draw_grid_on_cartesian_plot(ax, grid_params, grid_min_bound, grid_max_bound):
    rho_ticks = np.linspace(grid_min_bound[0], grid_max_bound[0], grid_params[0])
    phi_ticks = np.linspace(grid_min_bound[1], grid_max_bound[1], grid_params[1])
    theta_for_circle = np.linspace(grid_min_bound[1], grid_max_bound[1], 200)
    for rho in rho_ticks:
        x_coords = rho * np.cos(theta_for_circle)
        y_coords = rho * np.sin(theta_for_circle)
        ax.plot(x_coords, y_coords, color='white', linewidth=0.5, alpha=0.5)
    for phi in phi_ticks:
        x_start = grid_min_bound[0] * np.cos(phi)
        y_start = grid_min_bound[0] * np.sin(phi)
        x_end = grid_max_bound[0] * np.cos(phi)
        y_end = grid_max_bound[0] * np.sin(phi)
        ax.plot([x_start, x_end], [y_start, y_end], color='white', linewidth=0.5, alpha=0.5)


# ==============================================================================
# 主執行函數 (無變化)
# ==============================================================================
def main(pcd_file, num_points_to_sample, grid):
    """
    主函數，協調整個加載、採樣和分步可視化流程。
    """
    print(f"1. 從 '{pcd_file}' 加載點雲...")
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    if len(points) < num_points_to_sample:
        print(f"錯誤: 點雲中的點數 ({len(points)}) 少于要採樣的點數 ({num_points_to_sample}).")
        return

    print(f"2. 隨機選擇中心點，並提取 {num_points_to_sample} 個近鄰點作為輸入...")
    center_idx = random.randint(0, len(points) - 1)
    center_point = points[center_idx]
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(center_point, num_points_to_sample)
    initial_chunk_pts = points[idx, :]
    print(f"   已提取 {len(initial_chunk_pts)} 個點。")

    print(f"3. 執行與訓練腳本一致的Polar採樣...")
    polar_sampled_pts, min_b, max_b = polar_samplr(initial_chunk_pts, mid_gt=None, grid=grid)
    print(f"   Polar採樣完成，得到 {len(polar_sampled_pts)} 個點。")

    print(f"4. 執行純隨機採樣...")
    num_to_sample_randomly = len(polar_sampled_pts)
    indices = np.arange(len(initial_chunk_pts))
    np.random.shuffle(indices)
    random_indices = indices[:num_to_sample_randomly]
    random_sampled_pts = initial_chunk_pts[random_indices]
    print(f"   純隨機採樣完成，得到 {len(random_sampled_pts)} 個點。")

    # --- 第一次可視化：採樣前 ---
    print("\n5. 生成採樣前可視化... (關閉此窗口後將繼續)")
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_facecolor('black')
    draw_grid_on_cartesian_plot(ax1, grid, min_b, max_b)
    ax1.scatter(initial_chunk_pts[:, 0], initial_chunk_pts[:, 1], s=1, color='gray',
                label=f'採樣前 ({len(initial_chunk_pts)}點)')
    ax1.scatter(center_point[0], center_point[1], marker='*', s=200, color='yellow', label='隨機中心點',
                edgecolors='black', zorder=5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Before Sampling (True XY Coordinates)', color='white')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()

    # --- 第二次可視化：Polar採樣后 ---
    print("\n6. 生成Polar採樣后可視化... (關閉此窗口後將繼續)")
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_facecolor('black')
    draw_grid_on_cartesian_plot(ax2, grid, min_b, max_b)
    ax2.scatter(polar_sampled_pts[:, 0], polar_sampled_pts[:, 1], s=10, color='red',
                label=f'Polar採樣后 ({len(polar_sampled_pts)}點)')
    ax2.scatter(center_point[0], center_point[1], marker='*', s=200, color='yellow', label='隨機中心點',
                edgecolors='black', zorder=5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('After Polar Sampling (True XY Coordinates)', color='white')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()

    # --- 第三次可視化：純隨機採樣后 ---
    print("\n7. 生成純隨機採樣后可視化...")
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.set_facecolor('black')
    draw_grid_on_cartesian_plot(ax3, grid, min_b, max_b)
    ax3.scatter(random_sampled_pts[:, 0], random_sampled_pts[:, 1], s=10, color='cyan',
                label=f'純隨機採樣 ({len(random_sampled_pts)}點)')
    ax3.scatter(center_point[0], center_point[1], marker='*', s=200, color='yellow', label='隨機中心點',
                edgecolors='black', zorder=5)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('After Random Sampling (True XY Coordinates)', color='white')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()

    print("\n可視化流程結束。")


if __name__ == "__main__":
    # 直接在代碼中指定參數
    pcd_file = "/Users/captain/workspace/MAPS_sample/data/0fc8a9b2a41944a5bd2aac364b19b4ec/lidar.pcd"  # 請替換為你的點雲文件路徑
    num_points_to_sample = 20000
    grid = [32, 64, 16]

    main(pcd_file, num_points_to_sample, grid)