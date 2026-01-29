import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import os


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_points):
        super(Generator, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 4096)
        self.fc6 = nn.Linear(4096, num_points * output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.view(-1, self.num_points, 6)

        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_points):
        super(Discriminator, self).__init__()
        self.num_points = num_points
        # 定义卷积层
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x[:, :, :3]
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, 2)[0]

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, D_output_real, D_output_fake):
        D_loss_real = -torch.mean(torch.log(D_output_real + 1e-8))
        D_loss_fake = -torch.mean(torch.log(1 - D_output_fake + 1e-8))
        D_loss = D_loss_real + D_loss_fake
        return D_loss


class GeneratorLoss(nn.Module):
    def __init__(self,
                 adversarial_weight=0.6,
                 normal_weight=0.05,
                 data_weight=0.25,
                 dist_weight=0.1,
                 min_dist=0.04,
                 max_dist=0.1):
        super(GeneratorLoss, self).__init__()
        self.adversarial_weight = adversarial_weight
        self.normal_weight = normal_weight
        self.data_weight = data_weight
        self.dist_weight = dist_weight
        self.min_dist = min_dist
        self.max_dist = max_dist

    def compute_local_distance_loss(self, fake_data):
        coords = fake_data[..., :3]
        B, N, _ = coords.shape

        xx = (coords ** 2).sum(dim=2, keepdim=True)
        yy = xx.transpose(1, 2)
        xy = coords @ coords.transpose(1, 2)
        dist_matrix = torch.sqrt(torch.clamp(xx + yy - 2 * xy, min=0.0))

        _, knn100 = dist_matrix.topk(100, dim=2, largest=False)

        batch_idx = torch.arange(B, device=coords.device).view(B, 1, 1).expand(B, N, 100)
        point_idx = torch.arange(N, device=coords.device).view(1, N, 1).expand(B, N, 100)
        dist100 = dist_matrix[batch_idx, point_idx, knn100]
        _, knn10 = dist100.topk(10, dim=2, largest=False)
        knn10_idx = torch.gather(knn100, 2, knn10)

        batch_idx2 = batch_idx[:, :, :10]
        point_idx2 = point_idx[:, :, :10]
        dist10 = dist_matrix[batch_idx2, point_idx2, knn10_idx]
        _, knn1 = dist10.topk(1, dim=2, largest=False)
        knn1_idx = torch.gather(knn10_idx, 2, knn1)

        batch_idx_flat = torch.arange(B, device=coords.device).view(B, 1).expand(B, N)
        nearest_idx = knn1_idx.squeeze(2)
        closest_pts = coords[batch_idx_flat, nearest_idx]

        dists = torch.norm(coords - closest_pts, p=2, dim=2)  # (B, N)
        dists = torch.clamp(dists, min=self.min_dist, max=self.max_dist)
        loss = ((self.min_dist - dists).clamp(min=0)**2 + (dists - self.max_dist).clamp(min=0)**2).mean()
        return loss

    def forward(self, D_output_fake, fake_data, real_data):
        adv_loss = -torch.mean(torch.log(D_output_fake + 1e-8))
        real_normals = real_data[..., 3:]
        fake_normals = fake_data[..., 3:]
        normal_loss = self.normal_weight * torch.mean((real_normals - fake_normals) ** 2)
        data_loss = self.data_weight * torch.mean((real_data[..., :3] - fake_data[..., :3]) ** 2)
        dist_loss = self.dist_weight * self.compute_local_distance_loss(fake_data)
        return self.adversarial_weight * adv_loss + normal_loss + data_loss + dist_loss


if __name__ == "__main__":

    input_dim = 128
    output_dim = 6
    num_points = 4096

    generator = Generator(input_dim, output_dim, num_points)
    discriminator = Discriminator(3, num_points)

    dummy_input = torch.randn(1, input_dim)
    output = generator(dummy_input)

    noise = torch.randn(8, input_dim)
    generated_point_clouds = generator(noise)
    print(f"Generated point clouds shape: {generated_point_clouds.shape}")

    real_point_clouds = torch.randn(8, num_points, output_dim)
    fake_point_clouds = generated_point_clouds.detach()
    real_output = discriminator(real_point_clouds)
    fake_output = discriminator(fake_point_clouds)

    print(f"Discriminator output for real point clouds: {real_output.shape}")
    print(f"Discriminator output for fake point clouds: {fake_output.shape}")

