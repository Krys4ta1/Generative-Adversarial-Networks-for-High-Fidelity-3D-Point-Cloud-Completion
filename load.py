import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import open3d as o3d


def load_point_clouds_from_folder(folder_path, voxel_size=0.05, num_points=4096):
    point_clouds = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            point_cloud = np.loadtxt(file_path, delimiter=',', usecols=(0, 1, 2))
            normals = np.loadtxt(file_path, delimiter=',', usecols=(3, 4, 5))

            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)

            downsampled = o3d_point_cloud.voxel_down_sample(voxel_size=voxel_size)
            print(f"Original points: {len(point_cloud)}, Downsampled points: {len(downsampled.points)}")

            downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))

            downsampled_array = np.asarray(downsampled.points)
            downsampled_normals = np.asarray(downsampled.normals)
            print(f"Downsampled points: {downsampled_array.shape[0]}, Normals: {downsampled_normals.shape[0]}")

            if downsampled_array.shape[0] < num_points:
                padding = np.zeros((num_points - downsampled_array.shape[0], 3))
                downsampled_array = np.vstack((downsampled_array, padding))
                normals_padding = np.zeros((num_points - downsampled_normals.shape[0], 3))
                downsampled_normals = np.vstack((downsampled_normals, normals_padding))
            elif downsampled_array.shape[0] > num_points:
                downsampled_array = downsampled_array[:num_points]
                downsampled_normals = downsampled_normals[:num_points]
            assert downsampled_array.shape[0] == downsampled_normals.shape[0], \
                f"Point cloud size ({downsampled_array.shape[0]}) and normals size ({downsampled_normals.shape[0]}) don't match."

            point_cloud_with_normals = np.hstack((downsampled_array, downsampled_normals))

            point_clouds.append(point_cloud_with_normals)

    return point_clouds


def load_data(folder_path, batch_size=8, voxel_size=0.05, num_points=1024):
    point_clouds = load_point_clouds_from_folder(folder_path, voxel_size=voxel_size, num_points=num_points)
    point_clouds = np.array(point_clouds, dtype=np.float32)  # 转换为numpy数组

    point_clouds_tensor = torch.tensor(point_clouds, dtype=torch.float32)

    dataset = TensorDataset(point_clouds_tensor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return data_loader


def save_point_clouds(point_clouds, output_folder, prefix="downsampled"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, point_cloud in enumerate(point_clouds):
        file_name = f"{prefix}_{idx+1}.txt"
        file_path = os.path.join(output_folder, file_name)
        np.savetxt(file_path, point_cloud, delimiter=',', fmt='%.6f')


if __name__ == "__main__":
    folder_path = os.path.join(os.getcwd(), 'chair')
    output_folder = os.path.join(os.getcwd(), 'downsampled')

    batch_size = 24
    voxel_size = 0.025
    data_loader = load_data(folder_path, batch_size=batch_size, voxel_size=voxel_size, num_points=4096)

    for batch in data_loader:
        print(f"Batch shape: {batch[0].shape}")
        save_point_clouds(batch[0].numpy(), output_folder)
        break
