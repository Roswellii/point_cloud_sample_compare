import numpy as np
import open3d as o3d
import random
import os
import matplotlib.pyplot as plt


# ==============================================================================
# GridSample 類別的完整實作 (與前一版本相同)
# ==============================================================================
class GridSample(object):
    def __init__(
            self,
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            keys=("coord", "color", "normal"),
            return_inverse=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        for key in self.keys:
            if key not in data_dict:
                pass

        scaled_coord = data_dict["coord"] / self.grid_size
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        if self.mode == "train":
            idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            for key in self.keys:
                if key in data_dict:
                    data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":
            print("本範例僅展示 'train' 模式。")
            idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            for key in self.keys:
                if key in data_dict:
                    data_dict[key] = data_dict[key][idx_unique]
            return data_dict

    @staticmethod
    def ravel_hash_vec(arr):
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1
        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


# ==============================================================================
# 輔助函數：將點雲投影到 2D 圖像
# ==============================================================================
def project_to_2d_image(points, x_range, y_range, image_size=(512, 512)):
    """
    將 3D 點雲投影到 XY 平面生成 2D 圖像。
    """
    image = np.zeros(image_size, dtype=np.uint8)
    x_coords = ((points[:, 0] - x_range[0]) / (x_range[1] - x_range[0])) * (image_size[0] - 1)
    y_coords = ((points[:, 1] - y_range[0]) / (y_range[1] - y_range[0])) * (image_size[1] - 1)
    x_coords = np.clip(x_coords, 0, image_size[0] - 1).astype(int)
    y_coords = np.clip(y_coords, 0, image_size[1] - 1).astype(int)
    image[y_coords, x_coords] = 255
    return image


# ==============================================================================
# 主執行函數
# ==============================================================================
def main():
    """
    主執行函數，演示如何讀取PCD，應用GridSample，並生成兩張獨立的2D比較圖。
    """
    # 1. 指定要讀取的 PCD 檔案路徑
    pcd_file_path = "./data/0fc8a9b2a41944a5bd2aac364b19b4ec/lidar.pcd"

    # 2. 讀取指定的 PCD 檔案
    if not os.path.exists(pcd_file_path):
        print(f"錯誤：找不到檔案 '{pcd_file_path}'")
        return

    print(f"\n正在讀取指定的檔案: '{pcd_file_path}'...")
    try:
        pcd_loaded = o3d.io.read_point_cloud(pcd_file_path)
        if not pcd_loaded.has_points():
            print(f"錯誤：檔案 '{pcd_file_path}' 中不包含任何點。")
            return
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    original_points = np.asarray(pcd_loaded.points)
    print(f"原始點雲已載入，包含 {len(original_points)} 個點。")

    # 3. 準備 data_dict
    data_dict = {'coord': original_points}

    # 4. 初始化並應用 GridSample
    grid_size_to_use = 0.05
    print(f"\n正在應用 GridSample (grid_size={grid_size_to_use}, mode='train')...")
    grid_sampler = GridSample(grid_size=grid_size_to_use, mode='train', keys=('coord', 'color'))
    downsampled_data = grid_sampler(data_dict)
    downsampled_points = downsampled_data['coord']
    print(f"下採樣完成，剩餘 {len(downsampled_points)} 個點。")

    # 5. 生成兩張 2D 圖像並分別顯示
    print("\n正在生成 2D 投影圖...")

    # 為了確保兩張圖的比例尺相同，計算原始點雲的全局X,Y範圍
    x_range = (original_points[:, 0].min(), original_points[:, 0].max())
    y_range = (original_points[:, 1].min(), original_points[:, 1].max())

    # 生成原始點雲的 2D 圖像
    original_image = project_to_2d_image(original_points, x_range, y_range)

    # 生成過濾後點雲的 2D 圖像
    downsampled_image = project_to_2d_image(downsampled_points, x_range, y_range)

    # --- 繪製第一張圖：原始點雲 ---
    plt.figure(1, figsize=(20, 20))
    plt.imshow(original_image, cmap='gray')
    plt.title(f'原始點雲 (2D 俯視圖)\n{len(original_points)} 個點')
    plt.xlabel('X 軸')
    plt.ylabel('Y 軸')

    # --- 繪製第二張圖：過濾後點雲 ---
    plt.figure(2, figsize=(20, 20))
    plt.imshow(downsampled_image, cmap='gray')
    plt.title(f'GridSample 過濾後 (2D 俯視圖)\n{len(downsampled_points)} 個點')
    plt.xlabel('X 軸')
    plt.ylabel('Y 軸')

    # --- 顯示所有建立的圖形視窗 ---
    plt.show()


if __name__ == "__main__":
    main()