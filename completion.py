import torch
import torch.nn as nn
import numpy as np
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def knn_filtering(point_cloud, k=3, threshold=5.0):

    xyz = point_cloud[:, :3]
    N = xyz.shape[0]

    distance_matrix = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)

    sorted_distances = np.sort(distance_matrix, axis=1)[:, 1:k + 1]

    mean_distances = np.mean(sorted_distances, axis=1)

    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    mask = mean_distances < (global_mean + threshold * global_std)

    return point_cloud[mask]


def farthest_point_sampling(points, num_samples):
    N = points.shape[0]
    indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distances = torch.ones(N, device=points.device) * float('inf')
    farthest = torch.randint(0, N, (1,), device=points.device).item()

    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.argmax(distances).item()

    sampled_points = points[indices]
    return sampled_points, indices


def custom_downsampling(point_cloud, num_samples=4096):

    xyz = torch.tensor(point_cloud[:, :3], dtype=torch.float32)
    normals = torch.tensor(point_cloud[:, 3:], dtype=torch.float32)

    sampled_xyz, indices = farthest_point_sampling(xyz, num_samples)

    sampled_normals = normals[indices]

    downsampled_pc = torch.cat([sampled_xyz, sampled_normals], dim=1).numpy()

    return downsampled_pc


def voxel_downsampling(point_cloud, num_samples=4096, k_neighbors=16):
    filtered_pc = knn_filtering(point_cloud, k=10, threshold=20.0)

    downsampled_pc = custom_downsampling(filtered_pc, num_samples=num_samples)

    return downsampled_pc


class Encoder(nn.Module):
    def __init__(self, input_dim=6, feature_dim=64):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.ReLU()
        )
        self.global_fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        x = self.mlp(x)
        global_feat = self.global_fc(x.mean(dim=1))
        return global_feat


input_dim = 64
output_dim = 6
num_points = 4096

generator = Generator(input_dim, output_dim, num_points).to(device)
generator.load_state_dict(torch.load("./pth/desk/best_generator.pth", map_location=device))
generator.eval()

encoder = Encoder(input_dim=6, feature_dim=64).to(device)
encoder.eval()


def load_incomplete_point_cloud(file_path, target_points=4096):
    data = np.loadtxt(file_path, delimiter=",")
    num_points = data.shape[0]

    if num_points > target_points:
        indices = np.random.choice(num_points, target_points, replace=False)
        data = data[indices]
    elif num_points < target_points:
        extra_points = target_points - num_points
        indices = np.random.choice(num_points, extra_points, replace=True)
        data = np.vstack((data, data[indices]))

    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)


def save_point_cloud(file_path, point_cloud):
    np.savetxt(file_path, point_cloud, delimiter=",")
    print(f"Saved completed point cloud: {file_path}")


input_file = "3-3(对比桌子_缺失).txt"
output_file = "3-3(对比桌子_本方法).txt"

incomplete_pc = load_incomplete_point_cloud(input_file, target_points=num_points).to(device)

latent_code = encoder(incomplete_pc)
print("Latent code shape:", latent_code.shape)

generated_pc = generator(latent_code)
print("Generated point cloud shape:", generated_pc.shape)

fused_pc = torch.cat([incomplete_pc, generated_pc], dim=1)
print("Fused point cloud shape:", fused_pc.shape)

fused_pc_np = fused_pc.squeeze(0).detach().cpu().numpy()
print("Fused point cloud shape (after squeeze and detach):", fused_pc_np.shape)
downsampled_pc = voxel_downsampling(fused_pc_np, num_samples=4096)
print("Downsampled point cloud shape:", downsampled_pc.shape)

save_point_cloud(output_file, downsampled_pc)

print("Point cloud completion done!")
