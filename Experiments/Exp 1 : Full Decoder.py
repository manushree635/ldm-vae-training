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


def detect_faces_mtcnn(image, mtcnn_detector):
    """Use MTCNN for more accurate face and landmark detection"""
    image_np = image.cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 0.5 + 0.5) * 255  # Denormalize properly for RGB
    image_np = image_np.astype(np.uint8)

    result = mtcnn_detector.detect_faces(image_np)
    weight_map = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    for face in result:
        x, y, w, h = face['box']
        weight_map[y:y + h, x:x + w] = 5.0

        keypoints = face['keypoints']
        for key, point in keypoints.items():
            cv2.circle(weight_map, point, 8, 15.0, -1)

    return torch.from_numpy(weight_map).to(image.device)


def weighted_mse_loss(input, target, weight_map):
    """Weighted MSE loss with proper dimension handling"""
    weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]
    weight_map = weight_map.expand(-1, input.shape[1], -1, -1)  # [B, C, H, W]

    squared_diff = (input - target) ** 2
    weighted_squared_diff = weight_map * squared_diff

    loss = weighted_squared_diff.sum() / weight_map.sum()
    return loss


def train_vae_freeze_encoder(vae, dataloader, val_dataloader, num_epochs, device, use_weighted_loss=False,
                             mtcnn_detector=None, save_path="freeze_encoder_vae_model"):
    """
    Training with completely frozen encoder and very low learning rate for decoder only
    """
    print(f"FREEZE ENCODER TRAINING: Training decoder only for {num_epochs} epochs with very low LR")

    # Completely freeze encoder
    for param in vae.encoder.parameters():
        param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in vae.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

    # Very low learning rate for decoder-only training
    optimizer = torch.optim.Adam([p for p in vae.parameters() if p.requires_grad], lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)

    # Lower beta for KL divergence since we're not training encoder
    beta = 0.05

    epoch_losses = []
    recon_losses = []
    kl_losses = []
    face_losses = []
    val_losses = []
    val_face_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    # Create directory for saving models and plots
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_face_loss = 0

        vae.train()
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)

            # Forward pass with frozen encoder
            with torch.no_grad():
                encoded = vae.encode(images)
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.sample()
                else:
                    latents = encoded

            # Only train decoder
            recon_images = vae.decode(latents).sample

            if use_weighted_loss and mtcnn_detector:
                weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector) for img in images]).to(device)
                face_loss = weighted_mse_loss(recon_images, images, weight_maps)
                recon_loss = face_loss
            else:
                recon_loss = F.mse_loss(recon_images, images, reduction='mean')
                face_loss = recon_loss

            # No KL loss since encoder is frozen
            kl_loss = torch.tensor(0.0, device=device)
            loss = recon_loss

            optimizer.zero_grad()
            loss.backward()
            # Gentler gradient clipping for very low LR
            torch.nn.utils.clip_grad_norm_([p for p in vae.parameters() if p.requires_grad], 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_face_loss += face_loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f} Recon: {recon_loss.item():.6f} Face: {face_loss.item():.6f} "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_face_loss = total_face_loss / len(dataloader)

        epoch_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        face_losses.append(avg_face_loss)

        # Validation loop
        vae.eval()
        val_loss = 0
        val_face_loss = 0
        with torch.no_grad():
            for images in val_dataloader:
                images = images.to(device)

                # Forward pass with frozen encoder
                encoded = vae.encode(images)
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.sample()
                else:
                    latents = encoded

                recon_images = vae.decode(latents).sample

                if use_weighted_loss and mtcnn_detector:
                    weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector) for img in images]).to(device)
                    v_face_loss = weighted_mse_loss(recon_images, images, weight_maps)
                    v_recon_loss = v_face_loss
                else:
                    v_recon_loss = F.mse_loss(recon_images, images, reduction='mean')
                    v_face_loss = v_recon_loss

                val_loss += v_recon_loss.item()
                val_face_loss += v_face_loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_face_loss = val_face_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_face_losses.append(avg_val_face_loss)

        # Save best weights based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            vae.save_pretrained(os.path.join(save_path, "best_model"))
            print(f"✓ Saved best model at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            vae.save_pretrained(os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}"))
            print(f"Saved checkpoint at epoch {epoch + 1}")

        # Plot and save training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(epoch_losses, recon_losses, kl_losses, face_losses, val_losses, val_face_losses,
                                 save_path=os.path.join(save_path, f"training_curves_epoch_{epoch + 1}.png"))

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.6f} Val Loss: {avg_val_loss:.6f} "
              f"Face Loss: {avg_face_loss:.6f} Val Face Loss: {avg_val_face_loss:.6f}")

    # Save final model
    vae.save_pretrained(os.path.join(save_path, "final_model"))
    print(f"Saved final model at epoch {num_epochs}")

    return vae, epoch_losses, recon_losses, kl_losses, face_losses, val_losses, val_face_losses, best_val_loss, best_epoch


def plot_training_curves(epoch_losses, recon_losses, kl_losses, face_losses, val_losses, val_face_losses,
                         save_path="training_curves.png"):
    """Plot and save training and validation loss curves, including face loss"""
    epochs = range(1, len(epoch_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Total Loss (Train and Validation)
    ax1.plot(epochs, epoch_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'm-', label='Val Loss', linewidth=2)
    ax1.set_title('Total Loss (Frozen Encoder)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction Loss (including Face Loss)
    ax2.plot(epochs, recon_losses, 'r-', label='Train Recon Loss', linewidth=2)
    ax2.plot(epochs, face_losses, 'c-', label='Train Face Loss', linewidth=2)
    ax2.plot(epochs, val_face_losses, 'y-', label='Val Face Loss', linewidth=2)
    ax2.set_title('Reconstruction and Face Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # KL Divergence Loss (should be zero for frozen encoder)
    ax3.plot(epochs, kl_losses, 'g-', label='KL Loss', linewidth=2)
    ax3.set_title('KL Divergence Loss (Frozen Encoder)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # All Losses Combined
    ax4.plot(epochs, epoch_losses, 'b-', label='Total Loss', linewidth=2)
    ax4.plot(epochs, recon_losses, 'r-', label='Reconstruction Loss', linewidth=2)
    ax4.plot(epochs, face_losses, 'c-', label='Face Loss', linewidth=2)
    ax4.plot(epochs, val_losses, 'm-', label='Validation Loss', linewidth=2)
    ax4.plot(epochs, val_face_losses, 'y-', label='Validation Face Loss', linewidth=2)
    ax4.set_title('All Losses (Frozen Encoder Training)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# MAIN TRAINING - FREEZE ENCODER VERSION
dataset_dir = "/kaggle/input/humans-faces/Humans"
image_size = 256
batch_size = 4
num_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
dataset = FaceDataset(dataset_dir, num_images=7000, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation images")

# Load pretrained VAE
print("\nLoading fresh pretrained VAE...")
vae = AutoencoderKL.from_pretrained("EPFL-VILAB/flextok_vae_c8", low_cpu_mem_usage=True).to(device)

# Initialize MTCNN
mtcnn = MTCNN()

# Train with frozen encoder
print("\nStarting FREEZE ENCODER training with very low learning rate...")
vae_trained, epoch_losses, recon_losses, kl_losses, face_losses, val_losses, val_face_losses, best_val_loss, best_epoch = train_vae_freeze_encoder(
    vae, train_loader, val_loader, num_epochs, device,
    use_weighted_loss=True, mtcnn_detector=mtcnn,
    save_path="freeze_encoder_vae_model_decoder_only_chosen"
)

# Plot final results
plot_training_curves(epoch_losses, recon_losses, kl_losses, face_losses, val_losses, val_face_losses,
                     "freeze_encoder_training_curves_final.png")

print(f"\nFreeze encoder training completed! Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")