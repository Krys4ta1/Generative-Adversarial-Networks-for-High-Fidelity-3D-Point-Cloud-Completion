# Generative-Adversarial-Networks-for-High-Fidelity-3D-Point-Cloud-Completion
PyTorch implementation of a GAN-based method for high-fidelity 3D point cloud completion. 

This repository accompanies the manuscript submitted to *The Visual Computer*.

---

## Overview

This repository provides a PyTorch implementation of a generative adversarial network (GAN) for high-fidelity 3D point cloud completion from partial inputs.

The code is developed to support the experiments presented in the manuscript entitled  
“Generative Adversarial Networks for High-Fidelity 3D Point Cloud Completion”,  
which is currently under submission to *The Visual Computer*.

The proposed framework consists of an encoder, a generator, and a discriminator, and aims to reconstruct dense and geometrically consistent point clouds from incomplete observations. Experiments are conducted on the ModelNet40 dataset to evaluate the effectiveness of the proposed method.

---

## Requirements

To run this project, please install the following Python packages. It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

```bash
numpy>=1.24
torch>=2.1
torchvision>=0.16       # only if using torchvision
open3d>=0.17
matplotlib>=3.7
torchviz>=0.1.1
```

**Notes:**

- Python standard libraries such as `os` and `multiprocessing` are already included.

- Versions indicated are recommended; you may adjust if needed.

---

## Usage

This section describes how to train the model and perform 3D point cloud completion using the provided code.

### 1. Loading and Preprocessing Point Clouds (`load.py`)

The `load.py` script is responsible for:

- Loading 3D point cloud files from a specified folder (expects `.txt` files with columns `x,y,z,nx,ny,nz`).
- Performing voxel-based downsampling to reduce point cloud density.
- Estimating normals for each point.
- Ensuring each point cloud has a fixed number of points (`num_points`) by padding or trimming.
- Returning a PyTorch `DataLoader` for batch processing in training.

#### Example Usage

```python
from load import load_data

# Set paths to your dataset folder
folder_path = "chair"          # folder containing your .txt point cloud files
batch_size = 24                # number of point clouds per batch
voxel_size = 0.025             # voxel grid size for downsampling
num_points = 4096              # fixed number of points per cloud

# Load data and prepare DataLoader
data_loader = load_data(
    folder_path,
    batch_size=batch_size,
    voxel_size=voxel_size,
    num_points=num_points
)

# Iterate over batches
for batch in data_loader:
    print(f"Batch shape: {batch[0].shape}")  # (batch_size, num_points, 6)
```

**Saving Downsampled Point Clouds**

```python
from load import save_point_clouds

output_folder = "downsampled"
for batch in data_loader:
    save_point_clouds(batch[0].numpy(), output_folder)
    break  # save the first batch only as an example
```

**Notes:**

- Default parameters are set based on our experiments in the paper, but you can and should adjust `batch_size`, `voxel_size`, and `num_points` depending on your dataset and hardware.

- The script uses **Open3D** for point cloud processing and **PyTorch DataLoader** for batching.

- Standard Python libraries such as os and multiprocessing are already included.

### 2. Generator and Discriminator Models (`model.py`)

The `model.py` script defines the neural network architectures and loss functions for the GAN-based 3D point cloud completion:

- **Generator**: Takes a random noise vector and generates 3D point clouds with positions and normals `(x, y, z, nx, ny, nz)`.
- **Discriminator**: Evaluates whether a given point cloud is real or generated.
- **GeneratorLoss**: Combines adversarial loss, normal consistency loss, data reconstruction loss, and local distance constraints.
- **DiscriminatorLoss**: Standard GAN discriminator loss.

#### Example Usage

```python
import torch
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss

# Set model parameters
input_dim = 128          # Dimension of random noise vector
output_dim = 6           # Output dimension per point (x, y, z, nx, ny, nz)
num_points = 4096        # Number of points per point cloud

# Instantiate models
generator = Generator(input_dim, output_dim, num_points)
discriminator = Discriminator(3, num_points)  # only xyz coordinates used in discriminator

# Create dummy input to test generator
noise = torch.randn(8, input_dim)  # batch_size=8
generated_point_clouds = generator(noise)
print(f"Generated point clouds shape: {generated_point_clouds.shape}")  # (8, num_points, 6)

# Create dummy real data
real_point_clouds = torch.randn(8, num_points, output_dim)
fake_point_clouds = generated_point_clouds.detach()

# Discriminator forward pass
real_output = discriminator(real_point_clouds)
fake_output = discriminator(fake_point_clouds)
print(f"Discriminator output for real point clouds: {real_output.shape}")
print(f"Discriminator output for fake point clouds: {fake_output.shape}")

# Instantiate loss functions
G_loss = GeneratorLoss()
D_loss = DiscriminatorLoss()

# Compute losses
loss_D = D_loss(real_output, fake_output)
loss_G = G_loss(fake_output, fake_point_clouds, real_point_clouds)
print(f"Discriminator loss: {loss_D.item()}, Generator loss: {loss_G.item()}")
```

**Notes:**

- Default parameters (`input_dim`, `num_points`, `output_dim`) are based on our experiments in the paper, but you can and should adjust them depending on your dataset and hardware.

- GeneratorLoss allows tuning weights for adversarial, normal, data, and distance losses (`adversarial_weight`, `normal_weight`, `data_weight`, `dist_weight`) as well as minimum and maximum distance thresholds (`min_dist`, `max_dist`).

- PyTorch standard libraries such as `torch`, `torch.nn`, and `torch.nn.functional` are required.

### 3. Training Script (train.py)

The `train.py` script orchestrates the training of the GAN-based 3D point cloud completion:

- Loads point cloud datasets using load.py.

- Initializes the Generator and Discriminator models from model.py.

- Uses GeneratorLoss and DiscriminatorLoss for training.

- Supports adjustable learning rates, dynamic training frequency, gradient clipping, and checkpointing.

- Saves best models and sample generated point clouds during training.

#### Example Usage

```bash
# Run training script (make sure Python environment has required packages)
python train.py
```

Or from a Python script:

```python
import train

# On Windows, multiprocessing requires:
import multiprocessing
multiprocessing.freeze_support()

# Start training
train.main()
```

**Notes:**

- **Dataset and Parameters**

  - folder_path points to the dataset folder (e.g., "chair", "desk").

  - batch_size, voxel_size, and num_points are adjustable based on dataset size and hardware.

  - Input noise dimension input_dim and output dimension output_dim can also be modified.

  - Learning rates g_lr (generator) and d_lr (discriminator) are provided as defaults but can be tuned.

- **Checkpointing**

  - Checkpoints are saved in checkpoints/.
  
  - Both model weights (.pth) and sample point clouds (.txt) are stored periodically and for the best generator.

- **Training Control**

  - Dynamic adjustment of generator/discriminator training frequency is implemented.

  - Discriminator learning rate decays periodically.

  - Training can terminate early if generator loss drops below a threshold.

- **Hardware**

  - Supports GPU if available; otherwise runs on CPU.

  - Uses cuDNN benchmark for performance optimization.

- All default parameters are chosen based on experiments in the paper. Users are encouraged to tune them according to their datasets and hardware capabilities.

### 4. Point Cloud Completion Script (completion.py)

The `completion.py` script performs 3D point cloud completion using the trained Generator model:

- Loads an incomplete point cloud from a file.

- Encodes it into a latent feature vector using a simple `Encoder`.

- Generates a completed point cloud using the pre-trained `Generator`.

- Optionally fuses the original and generated point clouds.

- Performs KNN-based outlier filtering and farthest point sampling (FPS) for uniformity.

- Saves the final completed point cloud to a file.

#### Example Usage

```bash
# Run completion script
python completion.py
```

Or from Python:

```pyhton
from completion import load_incomplete_point_cloud, save_point_cloud, generator, encoder, voxel_downsampling

# Load incomplete point cloud
incomplete_pc = load_incomplete_point_cloud("input_point_cloud.txt", target_points=4096)

# Encode and generate completed point cloud
latent_code = encoder(incomplete_pc)
generated_pc = generator(latent_code)

# Fuse with original and downsample
fused_pc = torch.cat([incomplete_pc, generated_pc], dim=1)
fused_pc_np = fused_pc.squeeze(0).detach().cpu().numpy()
completed_pc = voxel_downsampling(fused_pc_np, num_samples=4096)

# Save final completed point cloud
save_point_cloud("completed_point_cloud.txt", completed_pc)
```

**Notes:**

- **Device**: Automatically uses GPU if available; otherwise falls back to CPU.

- **Filtering & Downsampling:**

  - `knn_filtering` removes outliers.

  - `farthest_point_sampling` (FPS) ensures uniform sampling.

  - `voxel_downsampling` combines these steps to produce a clean, fixed-size point cloud.

- **Encoder**:

  - `Encoder` extracts a global feature from the incomplete point cloud.

  - `Generator` acts as a decoder to reconstruct the full point cloud.

  - Default feature dimension is `64`; output dimension is `6` (xyz + normals).

- **Input/Output**:

  - Supports `.txt` files with columns `x,y,z,nx,ny,nz`.

  - Target number of points can be changed (`num_samples`).

- **Pre-trained Model**:

  - Load your pre-trained generator weights (e.g., `"./pth/desk/best_generator.pth"`).

- Default parameters such as `num_points`, `input_dim`, and filtering thresholds are provided but can be adjusted for different datasets.

---

## Dataset

The experiments in this work are based on the **ModelNet40** dataset.

Due to the inherent randomness of Generative Adversarial Network training, the generated point clouds may vary between training runs. Therefore, exact replication of every individual point in the generated point clouds is not expected.

To facilitate reproduction and comparison, we provide **example point cloud files for the chair category** in the repository. These include:
- the incomplete input point cloud (`*partial_chair.txt`),
- the generated point cloud from the GAN (`*generated_chair.txt`),
- and the final completed point cloud after fusion and downsampling (`*completed_chair.txt`).

These files demonstrate the workflow of our method and can be directly used to visualize or evaluate chair point cloud completion without retraining the model.

---

## Citation

If you use this code in your research, please consider citing our submitted manuscript:

@article{Zhao2026GAN3DPCC,
title={Generative Adversarial Networks for High-Fidelity 3D Point Cloud Completion},
author={Di Zhao and Sizhe Mao and Others},
journal={The Visual Computer (submitted)},
year={2026}
}

**Note:** This repository is directly related to the submitted manuscript to *The Visual Computer*.

---

## License

This repository is released under the **MIT License**.

You are free to use, modify, and distribute this code for academic and non-commercial purposes. 
For commercial use, please contact the authors.

See the full license in the [LICENSE](./LICENSE) file.
