import numpy as np
import pandas as pd
import os

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
import random
from pathlib import Path
from mtcnn import MTCNN
import json
import pickle
from datetime import datetime
import lpips
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config('../config.yaml')

class FaceDataset(Dataset):
    """Custom Dataset for Kaggle Human Faces Dataset"""

    def __init__(self, image_dir, num_images=7000, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.endswith(('.jpg', '.png', '.jpeg'))]

        if len(self.image_paths) > num_images:
            self.image_paths = random.sample(self.image_paths, num_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def calculate_jnd_map(img, kernel_size=16, critical_point_ratio=0.15):
    """
    Calculate JND map based on kernel size and critical point
    """
    k_sqrt = int(np.sqrt(kernel_size))

    # Convert to numpy and ensure grayscale for JND calculation
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop image to be divisible by kernel size
    h, w = img.shape
    h_crop = (h // k_sqrt) * k_sqrt
    w_crop = (w // k_sqrt) * k_sqrt
    img_crop = img[:h_crop, :w_crop]

    # Calculate critical point
    critical_point = int(kernel_size * critical_point_ratio)

    # Initialize JND map
    jnd_map = np.zeros_like(img_crop, dtype=np.float32)

    for i in range(0, h_crop, k_sqrt):
        for j in range(0, w_crop, k_sqrt):
            patch = img_crop[i:i + k_sqrt, j:j + k_sqrt]

            # Calculate patch complexity
            patch_variance = np.var(patch)
            patch_gradient = np.sum(np.abs(np.gradient(patch)))

            # Simulate reconstruction error based on patch complexity and critical point
            complexity_factor = (patch_variance + patch_gradient) / 1000
            critical_factor = 1.0 / (critical_point + 1)

            # JND value: higher for complex patches with low critical points
            reconstruction_error = complexity_factor * critical_factor * 20
            jnd_map[i:i + k_sqrt, j:j + k_sqrt] = reconstruction_error

    return jnd_map


def create_jnd_weight_map(original_batch, reconstructed_batch,
                          kernel_size=16, critical_point_ratio=0.15,
                          high_weight_multiplier=5.0, smooth_sigma=1.0):
    """
    Create weight maps based on JND values

    Args:
        original_batch: Original images tensor [B, C, H, W]
        reconstructed_batch: Reconstructed images tensor [B, C, H, W]
        kernel_size: Size of kernel for JND calculation
        critical_point_ratio: Ratio for critical point calculation
        high_weight_multiplier: Multiplier for high JND regions
        smooth_sigma: Gaussian smoothing sigma for weight maps

    Returns:
        weight_maps: Weight maps tensor [B, H, W]
    """
    from scipy.ndimage import gaussian_filter
    batch_size = original_batch.shape[0]
    weight_maps = []

    for i in range(batch_size):
        # Convert tensors to numpy arrays (0-1 range)
        orig_np = original_batch[i].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = reconstructed_batch[i].detach().cpu().permute(1, 2, 0).numpy()

        # Denormalize from [-1, 1] to [0, 1]
        orig_np = (orig_np + 1.0) / 2.0
        recon_np = (recon_np + 1.0) / 2.0

        # Clip to valid range
        orig_np = np.clip(orig_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)

        # Calculate JND maps for both images
        jnd_map_orig = calculate_jnd_map(orig_np, kernel_size, critical_point_ratio)
        jnd_map_recon = calculate_jnd_map(recon_np, kernel_size, critical_point_ratio)

        # Create weight map based on difference in JND values
        jnd_diff = np.abs(jnd_map_orig - jnd_map_recon)

        # Normalize JND difference and create weight map
        jnd_diff_normalized = (jnd_diff - jnd_diff.min()) / (jnd_diff.max() - jnd_diff.min() + 1e-8)
        weight_map = 1.0 + (high_weight_multiplier - 1.0) * jnd_diff_normalized

        # Smooth the weight map
        if smooth_sigma > 0:
            weight_map = gaussian_filter(weight_map, sigma=smooth_sigma)

        weight_maps.append(torch.from_numpy(weight_map).float())

    return torch.stack(weight_maps).to(original_batch.device)


def detect_faces_mtcnn(image, mtcnn_detector):
    """Use MTCNN for face detection - FACE REGION ONLY"""
    image_np = image.cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 0.5 + 0.5) * 255
    image_np = image_np.astype(np.uint8)

    result = mtcnn_detector.detect_faces(image_np)
    weight_map = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    for face in result:
        x, y, w, h = face['box']
        weight_map[y:y + h, x:x + w] = 3.0

    return torch.from_numpy(weight_map).to(image.device)


def weighted_mse_loss(input, target, weight_map):
    """Weighted MSE loss with proper dimension handling"""
    weight_map = weight_map.unsqueeze(1)
    weight_map = weight_map.expand(-1, input.shape[1], -1, -1)

    squared_diff = (input - target) ** 2
    weighted_squared_diff = weight_map * squared_diff

    loss = weighted_squared_diff.sum() / weight_map.sum()
    return loss


def weighted_lpips_loss(lpips_fn, input, target, weight_map):
    """Weighted LPIPS loss with region emphasis"""
    lpips_losses = []

    for i in range(input.shape[0]):
        lpips_val = lpips_fn(input[i:i + 1], target[i:i + 1])
        avg_weight = weight_map[i].mean()
        weighted_lpips = lpips_val * avg_weight
        lpips_losses.append(weighted_lpips)

    return torch.stack(lpips_losses).mean()


def combined_loss_with_jnd(lpips_fn, input, target, weight_map=None,
                           lpips_weight=1.0, mse_weight=1.0,
                           use_face_weighting=False, use_jnd_weighting=False,
                           kernel_size=16, critical_point_ratio=0.15,
                           jnd_weight_multiplier=5.0):
    """
    Combined LPIPS + MSE loss with optional face and JND weighting
    """
    if use_jnd_weighting:
        jnd_weight_map = create_jnd_weight_map(
            target, input,
            kernel_size=kernel_size,
            critical_point_ratio=critical_point_ratio,
            high_weight_multiplier=jnd_weight_multiplier
        )

        if use_face_weighting and weight_map is not None:
            combined_weight_map = weight_map * jnd_weight_map
        else:
            combined_weight_map = jnd_weight_map
    else:
        combined_weight_map = weight_map

    if combined_weight_map is not None:
        lpips_loss = weighted_lpips_loss(lpips_fn, input, target, combined_weight_map)
        mse_loss = weighted_mse_loss(input, target, combined_weight_map)
    else:
        lpips_loss = lpips_fn(input, target).mean()
        mse_loss = F.mse_loss(input, target, reduction='mean')

    total_loss = lpips_weight * lpips_loss + mse_weight * mse_loss

    return total_loss, lpips_loss, mse_loss


def save_training_state(save_path, epoch, vae, optimizer, scheduler,
                        epoch_losses, lpips_losses, mse_losses, kl_losses,
                        val_losses, val_lpips_losses, val_mse_losses, learning_rates,
                        best_val_loss, best_epoch, training_config):
    """Save complete training state for resuming"""
    state_dict = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch_losses': epoch_losses,
        'lpips_losses': lpips_losses,
        'mse_losses': mse_losses,
        'kl_losses': kl_losses,
        'val_losses': val_losses,
        'val_lpips_losses': val_lpips_losses,
        'val_mse_losses': val_mse_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'training_config': training_config,
        'timestamp': datetime.now().isoformat()
    }

    state_file = os.path.join(save_path, f"training_state_epoch_{epoch}.pkl")
    with open(state_file, 'wb') as f:
        pickle.dump(state_dict, f)

    json_state = {
        'epoch': epoch,
        'epoch_losses': epoch_losses,
        'lpips_losses': lpips_losses,
        'mse_losses': mse_losses,
        'kl_losses': kl_losses,
        'val_losses': val_losses,
        'val_lpips_losses': val_lpips_losses,
        'val_mse_losses': val_mse_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'training_config': training_config,
        'timestamp': datetime.now().isoformat()
    }

    json_file = os.path.join(save_path, f"training_state_epoch_{epoch}.json")
    with open(json_file, 'w') as f:
        json.dump(json_state, f, indent=2)

    print(f"✓ Training state saved: {state_file}")
    return state_file


def load_training_state(state_file):
    """Load training state from file"""
    with open(state_file, 'rb') as f:
        state_dict = pickle.load(f)

    print(f"✓ Training state loaded from: {state_file}")
    print(f"  - Epoch: {state_dict['epoch']}")
    print(f"  - Best validation loss: {state_dict['best_val_loss']:.6f} at epoch {state_dict['best_epoch']}")
    print(f"  - Timestamp: {state_dict['timestamp']}")

    return state_dict


def visualize_jnd_weights(original, reconstructed, kernel_size=16, save_path=None):
    """Visualize JND weight maps for debugging"""
    with torch.no_grad():
        orig = original[0:1]
        recon = reconstructed[0:1]

        jnd_weights = create_jnd_weight_map(orig, recon, kernel_size=kernel_size)

        orig_np = (orig[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        recon_np = (recon[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        weight_np = jnd_weights[0].cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(recon_np)
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')

        axes[2].imshow(np.abs(orig_np - recon_np))
        axes[2].set_title('Difference')
        axes[2].axis('off')

        im = axes[3].imshow(weight_np, cmap='hot')
        axes[3].set_title('JND Weight Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def train_vae_with_jnd_weighting(vae, dataloader, val_dataloader, num_epochs, device,
                                 lpips_weight=1.0, mse_weight=1.0,
                                 use_face_weighting=False, use_jnd_weighting=True,
                                 kernel_size=16, critical_point_ratio=0.15,
                                 jnd_weight_multiplier=5.0,
                                 mtcnn_detector=None, save_path="jnd_weighted_vae",
                                 resume_from_state=None):
    """
    Train VAE with JND-based pixel weighting
    """
    training_config = {
        'num_epochs': num_epochs,
        'lpips_weight': lpips_weight,
        'mse_weight': mse_weight,
        'use_face_weighting': use_face_weighting,
        'use_jnd_weighting': use_jnd_weighting,
        'kernel_size': kernel_size,
        'critical_point_ratio': critical_point_ratio,
        'jnd_weight_multiplier': jnd_weight_multiplier,
        'device': str(device),
        'batch_size': dataloader.batch_size,
        'dataset_size': len(dataloader.dataset),
        'val_dataset_size': len(val_dataloader.dataset)
    }

    print(f"JND-WEIGHTED TRAINING: Training decoder for {num_epochs} epochs")
    print(f"Loss weights - LPIPS: {lpips_weight}, MSE: {mse_weight}")
    print(f"Face weighting: {use_face_weighting}")
    print(f"JND weighting: {use_jnd_weighting} (kernel_size: {kernel_size}, "
          f"critical_point_ratio: {critical_point_ratio}, multiplier: {jnd_weight_multiplier})")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    for param in vae.encoder.parameters():
        param.requires_grad = False

    for param in vae.decoder.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vae.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

    os.makedirs(save_path, exist_ok=True)

    trainable_params_list = [p for p in vae.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params_list, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    start_epoch = 0
    epoch_losses = []
    lpips_losses = []
    mse_losses = []
    kl_losses = []
    val_losses = []
    val_lpips_losses = []
    val_mse_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    best_epoch = 0

    if resume_from_state:
        print(f"\nResuming training from: {resume_from_state}")
        state_dict = load_training_state(resume_from_state)

        vae.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

        start_epoch = state_dict['epoch']
        epoch_losses = state_dict['epoch_losses']
        lpips_losses = state_dict['lpips_losses']
        mse_losses = state_dict['mse_losses']
        kl_losses = state_dict['kl_losses']
        val_losses = state_dict['val_losses']
        val_lpips_losses = state_dict['val_lpips_losses']
        val_mse_losses = state_dict['val_mse_losses']
        learning_rates = state_dict['learning_rates']
        best_val_loss = state_dict['best_val_loss']
        best_epoch = state_dict['best_epoch']

        print(f"Resumed from epoch {start_epoch}, continuing to epoch {num_epochs}")

    print(f"\nStarting training with JND-based weighting...")
    print(f"Using Cosine Annealing LR scheduler: {1e-4} -> {1e-6}")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        total_lpips_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0

        vae.train()
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            with torch.no_grad():
                encoded = vae.encode(images)
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.sample()
                else:
                    latents = encoded

            recon_images = vae.decode(latents).sample

            face_weight_maps = None
            if use_face_weighting and mtcnn_detector:
                face_weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector)
                                                for img in images]).to(device)

            loss, lpips_loss, mse_loss = combined_loss_with_jnd(
                lpips_fn, recon_images, images, face_weight_maps,
                lpips_weight, mse_weight,
                use_face_weighting, use_jnd_weighting,
                kernel_size, critical_point_ratio, jnd_weight_multiplier
            )

            kl_loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_([p for p in vae.parameters() if p.requires_grad], 1.0)

            optimizer.step()

            total_loss += loss.item()
            total_lpips_loss += lpips_loss.item()
            total_mse_loss += mse_loss.item()
            total_kl_loss += kl_loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f} LPIPS: {lpips_loss.item():.6f} MSE: {mse_loss.item():.6f} "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if epoch == start_epoch and batch_idx == 0 and use_jnd_weighting:
                visualize_jnd_weights(images, recon_images,
                                      kernel_size=kernel_size,
                                      save_path=os.path.join(save_path, f"jnd_weights_epoch_{epoch + 1}.png"))

        avg_loss = total_loss / len(dataloader)
        avg_lpips_loss = total_lpips_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)

        epoch_losses.append(avg_loss)
        lpips_losses.append(avg_lpips_loss)
        mse_losses.append(avg_mse_loss)
        kl_losses.append(avg_kl_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        vae.eval()
        val_loss = 0
        val_lpips_loss = 0
        val_mse_loss = 0

        with torch.no_grad():
            for images in val_dataloader:
                images = images.to(device)

                encoded = vae.encode(images)
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.sample()
                else:
                    latents = encoded

                recon_images = vae.decode(latents).sample

                face_weight_maps = None
                if use_face_weighting and mtcnn_detector:
                    face_weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector)
                                                    for img in images]).to(device)

                v_loss, v_lpips_loss, v_mse_loss = combined_loss_with_jnd(
                    lpips_fn, recon_images, images, face_weight_maps,
                    lpips_weight, mse_weight,
                    use_face_weighting, use_jnd_weighting,
                    kernel_size, critical_point_ratio, jnd_weight_multiplier
                )

                val_loss += v_loss.item()
                val_lpips_loss += v_lpips_loss.item()
                val_mse_loss += v_mse_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_lpips_loss = val_lpips_loss / len(val_dataloader)
        avg_val_mse_loss = val_mse_loss / len(val_dataloader)

        val_losses.append(avg_val_loss)
        val_lpips_losses.append(avg_val_lpips_loss)
        val_mse_losses.append(avg_val_mse_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            vae.save_pretrained(os.path.join(save_path, "best_model"))
            print(f"✓ Saved best model at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            save_training_state(
                save_path, epoch + 1, vae, optimizer, scheduler,
                epoch_losses, lpips_losses, mse_losses, kl_losses,
                val_losses, val_lpips_losses, val_mse_losses, learning_rates,
                best_val_loss, best_epoch, training_config
            )

        if (epoch + 1) % 10 == 0:
            vae.save_pretrained(os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

        if (epoch + 1) % 10 == 0:
            plot_training_curves(epoch_losses, lpips_losses, mse_losses, kl_losses,
                                 val_losses, val_lpips_losses, val_mse_losses,
                                 learning_rates,
                                 save_path=os.path.join(save_path, f"training_curves_epoch_{epoch + 1}.png"))

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f} LPIPS: {avg_lpips_loss:.6f} MSE: {avg_mse_loss:.6f} "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    save_training_state(
        save_path, num_epochs, vae, optimizer, scheduler,
        epoch_losses, lpips_losses, mse_losses, kl_losses,
        val_losses, val_lpips_losses, val_mse_losses, learning_rates,
        best_val_loss, best_epoch, training_config
    )

    vae.save_pretrained(os.path.join(save_path, "final_model"))
    print(f"Saved final model at epoch {num_epochs}")

    summary_path = os.path.join(save_path, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== JND-WEIGHTED TRAINING SUMMARY ===\n\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"LPIPS Weight: {lpips_weight}\n")
        f.write(f"MSE Weight: {mse_weight}\n")
        f.write(f"Face Weighting: {use_face_weighting}\n")
        f.write(f"JND Weighting: {use_jnd_weighting}\n")
        f.write(f"Kernel Size: {kernel_size}\n")
        f.write(f"Critical Point Ratio: {critical_point_ratio}\n")
        f.write(f"JND Weight Multiplier: {jnd_weight_multiplier}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}\n")
        f.write(f"Final Learning Rate: {learning_rates[-1]:.2e}\n")
        f.write(f"Trainable Parameters: {trainable_params:,} / {total_params:,}\n\n")
        f.write(f"MODEL LOCATIONS:\n")
        f.write(f"Best Model: best_model/\n")
        f.write(f"Final Model: final_model/\n\n")
        f.write(f"TRAINING STATE FILES:\n")
        for file in os.listdir(save_path):
            if file.startswith("training_state_epoch_"):
                f.write(f"  - {file}\n")

    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    return vae, epoch_losses, lpips_losses, mse_losses, kl_losses, val_losses, val_lpips_losses, val_mse_losses, learning_rates, best_val_loss, best_epoch

def plot_training_curves(epoch_losses, lpips_losses, mse_losses, kl_losses,
                         val_losses, val_lpips_losses, val_mse_losses, learning_rates,
                         save_path="training_curves.png"):
    """Plot training curves with JND weighting information"""
    epochs = range(1, len(epoch_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(epochs, epoch_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'm-', label='Val Loss', linewidth=2)
    ax1.set_title('Total Loss (LPIPS + MSE with JND Weighting)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, lpips_losses, 'c-', label='Train LPIPS Loss', linewidth=2)
    ax2.plot(epochs, val_lpips_losses, 'orange', label='Val LPIPS Loss', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, mse_losses, 'g--', label='Train MSE Loss', linewidth=2, alpha=0.7)
    ax2_twin.plot(epochs, val_mse_losses, 'r--', label='Val MSE Loss', linewidth=2, alpha=0.7)

    ax2.set_title('LPIPS vs MSE Loss Components (JND Weighted)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LPIPS Loss', color='c')
    ax2_twin.set_ylabel('MSE Loss', color='g')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax3.set_title('Cosine Annealing Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(epochs, val_losses, 'm-', label='Total Val Loss', linewidth=2)
    ax4.plot(epochs, val_lpips_losses, 'orange', label='Val LPIPS Loss', linewidth=2, alpha=0.7)
    ax4.plot(epochs, val_mse_losses, 'r-', label='Val MSE Loss', linewidth=2, alpha=0.7)
    ax4.set_title('Validation Loss Components (JND Weighted)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


dataset_dir = config['dataset_dir']
image_size = config['image_size']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = FaceDataset(dataset_dir, num_images=config['num_images'], transform=transform)
train_size = int(config['train_split_ratio'] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation images")

print("\nLoading fresh pretrained VAE...")
vae = AutoencoderKL.from_pretrained(config['pretrained_vae'], low_cpu_mem_usage=True).to(device)

mtcnn = MTCNN()

resume_state_file = None
save_path = config['save_path']

input_resume_path = config['resume_path']

if os.path.exists(input_resume_path):
    state_files = [f for f in os.listdir(input_resume_path) if
                   f.startswith("training_state_epoch_") and f.endswith(".pkl")]
    if state_files:
        latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in state_files])
        resume_state_file = os.path.join(input_resume_path, f"training_state_epoch_{latest_epoch}.pkl")
        print(f"Found training state to resume from: {resume_state_file}")

os.makedirs(save_path, exist_ok=True)

print("\nStarting FULL DECODER training with LPIPS and JND weighting...")
vae_trained, epoch_losses, lpips_losses, mse_losses, kl_losses, val_losses, val_lpips_losses, val_mse_losses, learning_rates, best_val_loss, best_epoch = train_vae_with_jnd_weighting(
    vae, train_loader, val_loader, num_epochs, device,
    lpips_weight=config['lpips_weight'],
    mse_weight=config['mse_weight'],
    use_face_weighting=config['use_face_weighting'],
    use_jnd_weighting=config['use_jnd_weighting'],
    kernel_size=config['kernel_size'],
    critical_point_ratio=config['critical_point_ratio'],
    jnd_weight_multiplier=config['jnd_weight_multiplier'],
    mtcnn_detector=mtcnn, save_path=save_path,
    resume_from_state=resume_state_file
)

plot_training_curves(
    epoch_losses, lpips_losses, mse_losses, kl_losses,
    val_losses, val_lpips_losses, val_mse_losses,
    learning_rates, save_path="/kaggle/working/jnd_weighted_training_final_curves.png"
)

print(f"\nFull decoder training with LPIPS and JND weighting completed!")
print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
print(f"Final learning rate: {learning_rates[-1]:.2e}")
print(f"All models and logs saved in: {save_path}/")
print(f"Training curves saved as: jnd_weighted_training_final_curves.png")