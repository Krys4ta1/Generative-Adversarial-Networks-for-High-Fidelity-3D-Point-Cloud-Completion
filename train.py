import os
import torch
import torch.optim as optim
from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from load import load_data
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    folder_path = os.path.join(os.getcwd(), 'desk')
    batch_size = 24
    voxel_size = 0.0325
    num_points = 4096
    data_loader = load_data(folder_path, batch_size=batch_size, voxel_size=voxel_size, num_points=num_points)

    input_dim = 64
    output_dim = 6
    generator = Generator(input_dim, output_dim, num_points).to(device)
    discriminator = Discriminator(3, num_points).to(device)

    g_loss_fn = GeneratorLoss()
    d_loss_fn = DiscriminatorLoss()

    g_lr = 0.001
    d_lr = 0.0001
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

    def adjust_discriminator_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.99, decay_every=10,
                                           min_lr=0.00001):
        if epoch % decay_every == 0:
            new_lr = initial_lr * (decay_rate ** (epoch // decay_every))
            if new_lr < min_lr:
                new_lr = min_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Updated discriminator learning rate to {new_lr:.6f}")

    def adjust_training_frequency(d_loss, g_loss, d_step_interval=1, g_step_interval=1):
        if g_loss > d_loss:
            g_step_interval = 1
            d_step_interval = 2
        elif g_loss < 3 * d_loss:
            g_step_interval = 2
            d_step_interval = 1
        return d_step_interval, g_step_interval

    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_list, g_loss_list, d_loss_list = [], [], []

    num_epochs = 200
    best_g_loss = float('inf')
    best_epoch = 0
    d_step_interval = 1
    g_step_interval = 1
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for batch_idx, (real_data,) in enumerate(data_loader):

            real_data = real_data.to(device, non_blocking=True)

            noise = torch.randn(batch_size, input_dim, device=device)
            fake_data = generator(noise)

            D_output_real = discriminator(real_data)
            D_output_fake = discriminator(fake_data.detach())

            d_loss = d_loss_fn(D_output_real, D_output_fake)

            if batch_idx % d_step_interval == 0:
                d_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                # print(f"[Epoch {epoch} | Batch {batch_idx}] D_Grad norm (clipped): {D_norm:.4f}")
                d_optimizer.step()

            D_output_fake = discriminator(fake_data)

            g_loss = g_loss_fn(D_output_fake, fake_data, real_data)

            if batch_idx % g_step_interval == 0:
                g_optimizer.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
                # print(f"[Epoch {epoch} | Batch {batch_idx}] G_Grad norm (clipped): {G_norm:.4f}")
                g_optimizer.step()

        epoch_list.append(epoch + 1)
        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())

        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        if epoch > 40 and g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            best_epoch = epoch
            best_pc = fake_data.detach().cpu().numpy()

            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'best_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'best_discriminator.pth'))

            best_file = os.path.join(checkpoint_dir, 'generated_sample_best_epoch.txt')
            with open(best_file, 'w') as f:
                for pt in best_pc[0]:
                    f.write(','.join(map(str, pt)) + '\n')
            print(f"[Best] Epoch {epoch}: G Loss={best_g_loss:.4f} → samples saved to {best_file}")

        if (epoch + 1) % 20 == 0:
            snap = epoch + 1
            gen_snap = os.path.join(checkpoint_dir, f'generator_epoch_{snap}.pth')
            dis_snap = os.path.join(checkpoint_dir, f'discriminator_epoch_{snap}.pth')
            torch.save(generator.state_dict(), gen_snap)
            torch.save(discriminator.state_dict(), dis_snap)
            noise = torch.randn(1, input_dim, device=device)
            pc_snap = generator(noise).detach().cpu().numpy()
            snap_file = os.path.join(checkpoint_dir, f'generated_sample_epoch_{snap}.txt')
            with open(snap_file, 'w') as f:
                for pt in pc_snap[0]:
                    f.write(','.join(map(str, pt)) + '\n')
            print(f"[Snapshot] Saved epoch {snap} models and sample to {checkpoint_dir}")

        if g_loss.item() < 0.1:
            print(f"Generator loss dropped below 0.01, stopping training.")
            break

        d_step_interval, g_step_interval = adjust_training_frequency(d_loss.item(), g_loss.item())

        adjust_discriminator_learning_rate(d_optimizer, epoch, d_lr)

    print(f"Best Generator at Epoch {best_epoch} with G Loss: {best_g_loss:.4f} saved!")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
