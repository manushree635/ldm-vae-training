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
import lpips


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


        #Note for Manushree, You can use either of the setting.
        for key, point in keypoints.items():
            cv2.circle(weight_map, point, 8, 15.0, -1)
            box_size=8
            #px, py = point
            #top_left = (px - box_size, py - box_size)
            #bottom_right = (px + box_size, py + box_size)

            #cv2.rectangle(weight_map, top_left, bottom_right, color=15.0, thickness=-1)  # -1 to fill the box

    return torch.from_numpy(weight_map).to(image.device)


def weighted_mse_loss(input, target, weight_map):
    """Weighted MSE loss with proper dimension handling"""
    weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]
    weight_map = weight_map.expand(-1, input.shape[1], -1, -1)  # [B, C, H, W]

    squared_diff = (input - target) ** 2
    weighted_squared_diff = weight_map * squared_diff

    loss = weighted_squared_diff.sum() / weight_map.sum()
    return loss


def weighted_lpips_loss(lpips_fn, input, target, weight_map):
    """Weighted LPIPS loss with face region emphasis"""
    # Get LPIPS loss for each image in batch
    lpips_losses = []

    for i in range(input.shape[0]):
        # Calculate LPIPS for this image
        lpips_val = lpips_fn(input[i:i + 1], target[i:i + 1])

        # Apply face weighting (LPIPS returns a scalar per image)
        # Use average weight as multiplier for this image
        avg_face_weight = weight_map[i].mean()
        weighted_lpips = lpips_val * avg_face_weight
        lpips_losses.append(weighted_lpips)

    return torch.stack(lpips_losses).mean()


def combined_loss(lpips_fn, input, target, weight_map=None,
                  lpips_weight=1.0, mse_weight=1.0, use_face_weighting=False):
    """
    Combined LPIPS + MSE loss with optional face weighting

    Args:
        lpips_fn: LPIPS loss function
        input: Reconstructed images
        target: Original images
        weight_map: Face detection weight maps (optional)
        lpips_weight: Weight for LPIPS loss component (default: 1.0)
        mse_weight: Weight for MSE loss component (default: 1.0)
        use_face_weighting: Whether to apply face weighting
    """
    # LPIPS loss (perceptual)
    if use_face_weighting and weight_map is not None:
        lpips_loss = weighted_lpips_loss(lpips_fn, input, target, weight_map)
    else:
        lpips_loss = lpips_fn(input, target).mean()

    # MSE loss (pixel-wise)
    if use_face_weighting and weight_map is not None:
        mse_loss = weighted_mse_loss(input, target, weight_map)
    else:
        mse_loss = F.mse_loss(input, target, reduction='mean')

    # Combine losses with equal weights
    total_loss = lpips_weight * lpips_loss + mse_weight * mse_loss

    return total_loss, lpips_loss, mse_loss


def get_decoder_layer_blocks(vae):
    """
    Intelligently group decoder layers into blocks based on Conv + activation + normalization patterns.
    This follows the SDXL VAE architecture mentioned in the FlexTok paper.
    """
    decoder_blocks = []
    current_block = []
    block_name = ""

    # Get all named modules in decoder
    decoder_modules = list(vae.decoder.named_modules())

    # Group modules into logical blocks
    for name, module in decoder_modules:
        if len(list(module.children())) == 0:  # Leaf modules only
            if any(param.requires_grad for param in module.parameters()):
                # Start a new block if we encounter a Conv layer
                if 'conv' in name.lower() or 'convolution' in name.lower():
                    if current_block:  # Save previous block
                        decoder_blocks.append((block_name, current_block))
                    current_block = [(name, module)]
                    block_name = name.split('.')[0] if '.' in name else name
                # Add to current block if it's related (norm, activation, etc.)
                elif any(keyword in name.lower() for keyword in
                         ['norm', 'relu', 'silu', 'swish', 'gelu', 'act', 'bn', 'ln']):
                    current_block.append((name, module))
                # Handle other types of layers
                else:
                    if current_block:
                        current_block.append((name, module))
                    else:
                        # Standalone layer becomes its own block
                        decoder_blocks.append((name, [(name, module)]))

    # Add the last block if it exists
    if current_block:
        decoder_blocks.append((block_name, current_block))

    # Reverse to get from output to input (following the original logic)
    decoder_blocks.reverse()

    print(f"\nDecoder architecture grouped into {len(decoder_blocks)} blocks:")
    for i, (block_name, block_layers) in enumerate(decoder_blocks[:15]):  # Show first 15 blocks
        layer_types = [name.split('.')[-1] for name, _ in block_layers]
        print(f"  Block {i + 1} ({block_name}): {len(block_layers)} layers - {', '.join(layer_types)}")
    if len(decoder_blocks) > 15:
        print(f"  ... and {len(decoder_blocks) - 15} more blocks")

    return decoder_blocks


def freeze_all_decoder_layers(vae):
    """Freeze all decoder parameters"""
    for param in vae.decoder.parameters():
        param.requires_grad = False


def unfreeze_decoder_blocks(vae, decoder_blocks, num_blocks_to_unfreeze):
    """Unfreeze the specified number of decoder blocks (closest to output)"""
    unfrozen_count = 0

    # Unfreeze the specified number of blocks from the beginning (closest to output)
    for i in range(min(num_blocks_to_unfreeze, len(decoder_blocks))):
        block_name, block_layers = decoder_blocks[i]
        for layer_name, layer_module in block_layers:
            for param in layer_module.parameters():
                param.requires_grad = True
        unfrozen_count += 1
        print(f"  Unfrozen block {i + 1}: {block_name} ({len(block_layers)} layers)")

    return unfrozen_count


def get_adaptive_lr_schedule_100_epochs():
    """Define the adaptive learning rate schedule for 100 epochs with 10 phases, equally dividing 36 blocks"""
    schedule = [
        (0, 10, 4, 1e-3),  # Phase 1: Epochs 0-9, 4 blocks
        (10, 20, 8, 8e-4),  # Phase 2: Epochs 10-19, 8 blocks
        (20, 30, 12, 6e-4),  # Phase 3: Epochs 20-29, 12 blocks
        (30, 40, 16, 4e-4),  # Phase 4: Epochs 30-39, 16 blocks
        (40, 50, 20, 3e-4),  # Phase 5: Epochs 40-49, 20 blocks
        (50, 60, 24, 2e-4),  # Phase 6: Epochs 50-59, 24 blocks
        (60, 70, 28, 1.5e-4),  # Phase 7: Epochs 60-69, 28 blocks
        (70, 80, 32, 1e-4),  # Phase 8: Epochs 70-79, 32 blocks
        (80, 90, 36, 8e-5),  # Phase 9: Epochs 80-89, 36 blocks
        (90, 100, 36, 5e-5)  # Phase 10: Epochs 90-99, 36 blocks
    ]
    return schedule


def train_vae_progressive_unfreezing_lpips_100_epochs(vae, dataloader, val_dataloader, num_epochs, device,
                                                      lpips_weight=1.0, mse_weight=1.0, use_face_weighting=False,
                                                      mtcnn_detector=None,
                                                      save_path="progressive_unfreezing_vae_lpips_100"):
    """
    Progressive block-by-block unfreezing with LPIPS + MSE loss for 100 epochs

    Args:
        vae: VAE model
        dataloader: Training data loader
        val_dataloader: Validation data loader
        num_epochs: Number of training epochs (should be 100)
        device: Training device
        lpips_weight: Weight for LPIPS loss component (default: 1.0)
        mse_weight: Weight for MSE loss component (default: 1.0)
        use_face_weighting: Whether to apply face-region weighting (default: False)
        mtcnn_detector: MTCNN detector for face weighting (required if use_face_weighting=True)
        save_path: Path to save models and results
    """
    print(f"PROGRESSIVE UNFREEZING (100 EPOCHS): Training decoder for {num_epochs} epochs")
    print(f"Loss weights - LPIPS: {lpips_weight}, MSE: {mse_weight} (EQUAL WEIGHTING)")
    print(f"Face weighting: {use_face_weighting}")

    # Initialize LPIPS loss function
    lpips_fn = lpips.LPIPS(net='alex').to(device)  # 'alex', 'vgg', or 'squeeze'
    lpips_fn.eval()  # LPIPS should be in eval mode

    # Always keep encoder completely frozen
    for param in vae.encoder.parameters():
        param.requires_grad = False

    # Get decoder blocks (intelligent grouping)
    decoder_blocks = get_decoder_layer_blocks(vae)
    total_blocks = len(decoder_blocks)
    print(f"\nTotal decoder blocks: {total_blocks}")

    # Get the unfreezing schedule
    lr_schedule = get_adaptive_lr_schedule_100_epochs()

    # Initialize tracking variables
    epoch_losses = []
    lpips_losses = []
    mse_losses = []
    kl_losses = []
    val_losses = []
    val_lpips_losses = []
    val_mse_losses = []
    learning_rates = []
    unfrozen_blocks_count = []

    # Global best model tracking
    best_val_loss = float('inf')
    best_epoch = 0

    # Phase-wise best model tracking
    phase_best_losses = {}
    phase_best_epochs = {}

    # Create directory for saving models and plots
    os.makedirs(save_path, exist_ok=True)

    # Create phase-specific directories
    for phase_idx in range(len(lr_schedule)):
        phase_dir = os.path.join(save_path, f"phase_{phase_idx + 1}_best")
        os.makedirs(phase_dir, exist_ok=True)

    # Initialize optimizer (will be recreated for each phase)
    optimizer = None
    scheduler = None
    current_phase = -1

    for epoch in range(num_epochs):
        # Check if we need to update the unfreezing schedule
        new_phase = None
        for phase_idx, (start_epoch, end_epoch, blocks, lr) in enumerate(lr_schedule):
            if start_epoch <= epoch < end_epoch:
                new_phase = phase_idx
                break

        # If we're in a new phase, update the unfreezing and optimizer
        if new_phase != current_phase:
            current_phase = new_phase
            start_epoch, end_epoch, blocks_to_unfreeze, learning_rate = lr_schedule[current_phase]

            # Initialize phase-specific best tracking
            phase_best_losses[current_phase] = float('inf')
            phase_best_epochs[current_phase] = 0

            print(f"\n=== PHASE {current_phase + 1} (Epochs {start_epoch + 1}-{end_epoch}) ===")
            print(f"Learning Rate: {learning_rate:.2e}")
            print(f"Blocks to unfreeze: {'All decoder blocks' if blocks_to_unfreeze == -1 else blocks_to_unfreeze}")

            # Freeze all decoder layers first
            freeze_all_decoder_layers(vae)

            # Unfreeze specified number of blocks
            if blocks_to_unfreeze == -1:
                # Unfreeze all decoder blocks
                for param in vae.decoder.parameters():
                    param.requires_grad = True
                print("  Unfrozen all decoder blocks")
                actual_unfrozen = total_blocks
            else:
                actual_unfrozen = unfreeze_decoder_blocks(vae, decoder_blocks, blocks_to_unfreeze)

            unfrozen_blocks_count.append(actual_unfrozen)

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in vae.parameters())
            print(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

            # Create new optimizer for this phase
            trainable_params_list = [p for p in vae.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params_list, lr=learning_rate, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.8, min_lr=1e-6)
        else:
            # Maintain the unfrozen blocks count for non-transition epochs
            if unfrozen_blocks_count:
                unfrozen_blocks_count.append(unfrozen_blocks_count[-1])
            else:
                unfrozen_blocks_count.append(0)

        # Training loop
        total_loss = 0
        total_lpips_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0

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

            # Train unfrozen decoder layers
            recon_images = vae.decode(latents).sample

            # Calculate combined loss with equal weights
            if use_face_weighting and mtcnn_detector:
                weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector) for img in images]).to(device)
                loss, lpips_loss, mse_loss = combined_loss(
                    lpips_fn, recon_images, images, weight_maps,
                    lpips_weight, mse_weight, use_face_weighting=True
                )
            else:
                loss, lpips_loss, mse_loss = combined_loss(
                    lpips_fn, recon_images, images, None,
                    lpips_weight, mse_weight, use_face_weighting=False
                )

            # No KL loss since encoder is frozen
            kl_loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()

            # Adaptive gradient clipping based on learning rate
            clip_value = 1.0 if learning_rate > 1e-4 else 0.5
            torch.nn.utils.clip_grad_norm_([p for p in vae.parameters() if p.requires_grad], clip_value)

            optimizer.step()

            total_loss += loss.item()
            total_lpips_loss += lpips_loss.item()
            total_mse_loss += mse_loss.item()
            total_kl_loss += kl_loss.item()

            if batch_idx % 20 == 0:
                current_blocks = unfrozen_blocks_count[-1] if unfrozen_blocks_count else 0
                print(
                    f"Phase {current_phase + 1} Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f} LPIPS: {lpips_loss.item():.6f} MSE: {mse_loss.item():.6f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} Blocks: {current_blocks}/{total_blocks}")

        avg_loss = total_loss / len(dataloader)
        avg_lpips_loss = total_lpips_loss / len(dataloader)
        avg_mse_loss = total_mse_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)

        epoch_losses.append(avg_loss)
        lpips_losses.append(avg_lpips_loss)
        mse_losses.append(avg_mse_loss)
        kl_losses.append(avg_kl_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation loop
        vae.eval()
        val_loss = 0
        val_lpips_loss = 0
        val_mse_loss = 0

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

                # Calculate validation loss with equal weights
                if use_face_weighting and mtcnn_detector:
                    weight_maps = torch.stack([detect_faces_mtcnn(img, mtcnn_detector) for img in images]).to(device)
                    v_loss, v_lpips_loss, v_mse_loss = combined_loss(
                        lpips_fn, recon_images, images, weight_maps,
                        lpips_weight, mse_weight, use_face_weighting=True
                    )
                else:
                    v_loss, v_lpips_loss, v_mse_loss = combined_loss(
                        lpips_fn, recon_images, images, None,
                        lpips_weight, mse_weight, use_face_weighting=False
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

        # Save best weights based on validation loss (Global best)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            vae.save_pretrained(os.path.join(save_path, "global_best_model"))
            print(f"✓ Saved GLOBAL best model at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")

        # Save phase-specific best model
        if avg_val_loss < phase_best_losses[current_phase]:
            phase_best_losses[current_phase] = avg_val_loss
            phase_best_epochs[current_phase] = epoch + 1

            # Save phase-specific best model
            phase_model_path = os.path.join(save_path, f"phase_{current_phase + 1}_best", "model")
            vae.save_pretrained(phase_model_path)

            # Save phase info
            current_blocks = unfrozen_blocks_count[-1] if unfrozen_blocks_count else 0
            phase_info = {
                'phase': current_phase + 1,
                'epoch': epoch + 1,
                'validation_loss': avg_val_loss,
                'lpips_loss': avg_lpips_loss,
                'mse_loss': avg_mse_loss,
                'val_lpips_loss': avg_val_lpips_loss,
                'val_mse_loss': avg_val_mse_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'unfrozen_blocks': current_blocks,
                'total_blocks': total_blocks,
                'lpips_weight': lpips_weight,
                'mse_weight': mse_weight,
                'phase_description': f"Phase {current_phase + 1}: {current_blocks}/{total_blocks} blocks unfrozen"
            }

            # Save phase info as text file
            with open(os.path.join(save_path, f"phase_{current_phase + 1}_best", "phase_info.txt"), 'w') as f:
                f.write("=== PHASE BEST MODEL INFO (100 EPOCHS EQUAL WEIGHTS) ===\n")
                f.write(f"Phase: {phase_info['phase']}\n")
                f.write(f"Best Epoch: {phase_info['epoch']}\n")
                f.write(f"Validation Loss: {phase_info['validation_loss']:.6f}\n")
                f.write(f"LPIPS Loss: {phase_info['lpips_loss']:.6f}\n")
                f.write(f"MSE Loss: {phase_info['mse_loss']:.6f}\n")
                f.write(f"Val LPIPS Loss: {phase_info['val_lpips_loss']:.6f}\n")
                f.write(f"Val MSE Loss: {phase_info['val_mse_loss']:.6f}\n")
                f.write(f"Learning Rate: {phase_info['learning_rate']:.2e}\n")
                f.write(f"Unfrozen Blocks: {phase_info['unfrozen_blocks']}/{phase_info['total_blocks']}\n")
                f.write(f"LPIPS Weight: {phase_info['lpips_weight']} (EQUAL)\n")
                f.write(f"MSE Weight: {phase_info['mse_weight']} (EQUAL)\n")
                f.write(f"Description: {phase_info['phase_description']}\n")
                f.write(f"Saved at: Epoch {phase_info['epoch']}\n")

            print(
                f"✓ Saved PHASE {current_phase + 1} best model at epoch {epoch + 1} with validation loss: {avg_val_loss:.6f}")

        # Save checkpoint at end of each phase
        if epoch + 1 in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            vae.save_pretrained(os.path.join(save_path, f"checkpoint_phase_{current_phase + 1}_epoch_{epoch + 1}"))
            print(f"Saved phase {current_phase + 1} checkpoint at epoch {epoch + 1}")

            # Save phase summary when phase ends
            phase_summary_path = os.path.join(save_path, f"phase_{current_phase + 1}_summary.txt")
            current_blocks = unfrozen_blocks_count[-1] if unfrozen_blocks_count else 0
            with open(phase_summary_path, 'w') as f:
                f.write(f"=== PHASE {current_phase + 1} SUMMARY (100 EPOCHS EQUAL WEIGHTS) ===\n")
                f.write(f"Epochs: {lr_schedule[current_phase][0] + 1}-{lr_schedule[current_phase][1]}\n")
                f.write(
                    f"Unfrozen Blocks: {lr_schedule[current_phase][2] if lr_schedule[current_phase][2] != -1 else f'All ({total_blocks}) decoder blocks'}\n")
                f.write(f"Learning Rate: {lr_schedule[current_phase][3]:.2e}\n")
                f.write(f"LPIPS Weight: {lpips_weight} (EQUAL)\n")
                f.write(f"MSE Weight: {mse_weight} (EQUAL)\n")
                f.write(f"Best Validation Loss: {phase_best_losses[current_phase]:.6f}\n")
                f.write(f"Best Epoch in Phase: {phase_best_epochs[current_phase]} (Global: {epoch + 1})\n")
                f.write(f"Phase completed at epoch: {epoch + 1}\n")
                f.write(f"Total decoder blocks: {total_blocks}\n")
            print(f"Saved phase {current_phase + 1} summary")

        # Plot training curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_lpips_training_curves_100_epochs(epoch_losses, lpips_losses, mse_losses, kl_losses,
                                                  val_losses, val_lpips_losses, val_mse_losses,
                                                  learning_rates, unfrozen_blocks_count, total_blocks,
                                                  save_path=os.path.join(save_path,
                                                                         f"training_curves_epoch_{epoch + 1}.png"))

        scheduler.step(avg_val_loss)

        current_blocks = unfrozen_blocks_count[-1] if unfrozen_blocks_count else 0
        print(f"Phase {current_phase + 1} Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f} LPIPS: {avg_lpips_loss:.6f} MSE: {avg_mse_loss:.6f} "
              f"Blocks: {current_blocks}/{total_blocks}")

    # Save final model
    vae.save_pretrained(os.path.join(save_path, "final_model"))
    print(f"Saved final model at epoch {num_epochs}")

    # Create comprehensive training summary
    summary_path = os.path.join(save_path, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== PROGRESSIVE UNFREEZING TRAINING SUMMARY (100 EPOCHS EQUAL WEIGHTS) ===\n\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"LPIPS Weight: {lpips_weight} (EQUAL WEIGHTING)\n")
        f.write(f"MSE Weight: {mse_weight} (EQUAL WEIGHTING)\n")
        f.write(f"Face Weighting: {use_face_weighting}\n")
        f.write(f"Total Decoder Blocks: {total_blocks}\n")
        f.write(f"Global Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}\n\n")

        f.write("PHASE-WISE BEST MODELS:\n")
        for phase_idx, (start, end, blocks, lr) in enumerate(lr_schedule):
            f.write(f"\nPhase {phase_idx + 1} (Epochs {start + 1}-{end}):\n")
            f.write(f"  Unfrozen Blocks: {blocks if blocks != -1 else f'All ({total_blocks}) decoder blocks'}\n")
            f.write(f"  Learning Rate: {lr:.2e}\n")
            f.write(f"  Best Val Loss: {phase_best_losses[phase_idx]:.6f}\n")
            f.write(f"  Best Epoch: {phase_best_epochs[phase_idx]}\n")
            f.write(f"  Model saved at: phase_{phase_idx + 1}_best/model/\n")

        f.write(f"\nMODEL LOCATIONS:\n")
        f.write(f"Global Best Model: global_best_model/\n")
        f.write(f"Final Model: final_model/\n")
        for phase_idx in range(len(lr_schedule)):
            f.write(f"Phase {phase_idx + 1} Best: phase_{phase_idx + 1}_best/model/\n")

    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Global best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"\nPhase-wise best models:")
    for phase_idx in range(len(lr_schedule)):
        print(f"  Phase {phase_idx + 1}: {phase_best_losses[phase_idx]:.6f} at epoch {phase_best_epochs[phase_idx]}")

    return vae, epoch_losses, lpips_losses, mse_losses, kl_losses, val_losses, val_lpips_losses, val_mse_losses, learning_rates, unfrozen_blocks_count, best_val_loss, best_epoch, phase_best_losses, phase_best_epochs


def plot_lpips_training_curves_100_epochs(epoch_losses, lpips_losses, mse_losses, kl_losses, val_losses,
                                          val_lpips_losses, val_mse_losses, learning_rates, unfrozen_blocks_count,
                                          total_blocks, save_path="lpips_training_curves_100.png"):
    """Plot training curves with LPIPS loss information for 100 epochs"""
    epochs = range(1, len(epoch_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Total Loss with phase boundaries
    ax1.plot(epochs, epoch_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'm-', label='Val Loss', linewidth=2)

    # Add phase boundaries
    phase_boundaries = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for boundary in phase_boundaries:
        if boundary <= len(epochs):
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    ax1.set_title('Total Loss (LPIPS + MSE, Equal Weights)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LPIPS vs MSE Loss comparison
    ax2.plot(epochs, lpips_losses, 'c-', label='Train LPIPS Loss', linewidth=2)
    ax2.plot(epochs, val_lpips_losses, 'orange', label='Val LPIPS Loss', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, mse_losses, 'g--', label='Train MSE Loss', linewidth=2, alpha=0.7)
    ax2_twin.plot(epochs, val_mse_losses, 'r--', label='Val MSE Loss', linewidth=2, alpha=0.7)

    for boundary in phase_boundaries:
        if boundary <= len(epochs):
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    ax2.set_title('LPIPS vs MSE Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LPIPS Loss', color='c')
    ax2_twin.set_ylabel('MSE Loss', color='g')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Learning Rate Schedule
    ax3.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2, marker='o', markersize=3)
    ax3.set_title('Adaptive Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    for boundary in phase_boundaries:
        if boundary <= len(epochs):
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    # Unfrozen Blocks Count
    ax4.plot(epochs, unfrozen_blocks_count, 'r-', label='Unfrozen Blocks', linewidth=2, marker='s', markersize=3)
    ax4.set_title('Progressive Block Unfreezing')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Number of Unfrozen Blocks')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    for boundary in phase_boundaries:
        if boundary <= len(epochs):
            ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    # Add phase annotations
    phase_labels = [
        'Phase 1\n(4 blocks)', 'Phase 2\n(8 blocks)', 'Phase 3\n(12 blocks)',
        'Phase 4\n(16 blocks)', 'Phase 5\n(20 blocks)', 'Phase 6\n(24 blocks)',
        'Phase 7\n(28 blocks)', 'Phase 8\n(32 blocks)', 'Phase 9\n(36 blocks)',
        'Phase 10\n(36 blocks)'
    ]
    phase_positions = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    for i, (pos, label) in enumerate(zip(phase_positions, phase_labels)):
        if pos <= len(epochs):
            ax4.annotate(label, xy=(pos, max(unfrozen_blocks_count) * 0.8),
                         xytext=(pos, max(unfrozen_blocks_count) * 0.9),
                         ha='center', va='bottom', fontsize=8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# MAIN TRAINING - PROGRESSIVE UNFREEZING WITH LPIPS (100 EPOCHS)
dataset_dir = "/kaggle/input/humans-faces/Humans"
image_size = 256
batch_size = 4
num_epochs = 100  # Set to 100 epochs as requested
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
dataset = FaceDataset(dataset_dir, num_images=4000, transform=transform)
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

# Print decoder architecture for reference
decoder_blocks = get_decoder_layer_blocks(vae)
print(f"\nTotal decoder blocks: {len(decoder_blocks)}")

# Train with progressive unfreezing using LPIPS (100 epochs)
print("\nStarting PROGRESSIVE UNFREEZING training with LPIPS (100 epochs)...")
print("Schedule:")
schedule = get_adaptive_lr_schedule_100_epochs()
for i, (start, end, blocks, lr) in enumerate(schedule):
    print(f"  Phase {i + 1}: Epochs {start + 1}-{end}, Blocks: {'All' if blocks == -1 else blocks}, LR: {lr:.2e}")

# Updated function call for LPIPS training (100 epochs)
vae_trained, epoch_losses, lpips_losses, mse_losses, kl_losses, val_losses, val_lpips_losses, val_mse_losses, learning_rates, unfrozen_blocks_count, best_val_loss, best_epoch, phase_best_losses, phase_best_epochs = train_vae_progressive_unfreezing_lpips_100_epochs(
    vae, train_loader, val_loader, num_epochs, device,
    lpips_weight=1.0, mse_weight=1.0, use_face_weighting=True,
    mtcnn_detector=mtcnn, save_path="progressive_unfreezing_vae_lpips_100"
)

# Plot final results using LPIPS plotting function
plot_lpips_training_curves_100_epochs(
    epoch_losses, lpips_losses, mse_losses, kl_losses,
    val_losses, val_lpips_losses, val_mse_losses,
    learning_rates, unfrozen_blocks_count, len(decoder_blocks),
    save_path="progressive_unfreezing_lpips_100_final_curves.png"
)

print(f"\nProgressive unfreezing training with LPIPS (100 epochs) completed!")
print(f"Global best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
print(f"Final learning rate: {learning_rates[-1]:.2e}")
print(f"Final unfrozen blocks: {unfrozen_blocks_count[-1]}")

print(f"\nPhase-wise best models saved:")
for phase_idx in range(len(get_adaptive_lr_schedule_100_epochs())):
    print(
        f"  Phase {phase_idx + 1}: Val Loss {phase_best_losses[phase_idx]:.6f} at epoch {phase_best_epochs[phase_idx]}")
    print(f"    Location: progressive_unfreezing_vae_lpips_100/phase_{phase_idx + 1}_best/model/")

print(f"\nAll models and logs saved in: progressive_unfreezing_vae_lpips_100/")
print(f"Training curves saved as: progressive_unfreezing_lpips_100_final_curves.png")