# Experiment 1: Frozen Encoder VAE Training with Weighted MSE Loss

This script fine-tunes the decoder of a pretrained VAE (`EPFL-VILAB/flextok_vae_c8`) on the Kaggle Human Faces dataset, keeping the encoder frozen. It uses a weighted MSE loss with MTCNN-based face detection to emphasize facial regions. The training runs for 40 epochs with a very low learning rate (5e-5) and saves the best model based on validation loss. Expect training logs, loss curves (saved every 5 epochs), and model checkpoints in the `freeze_encoder_vae_model` directory. The script outputs the best validation loss and epoch, along with final model artifacts.

**Key Features**:
- Frozen encoder, decoder-only training
- MTCNN face detection for weighted loss
- Comprehensive loss curve visualization

# Experiment 2: Progressive Unfreezing with LPIPS + MSE Loss

This script implements progressive unfreezing of the VAE decoder (`EPFL-VILAB/flextok_vae_c8`) over 100 epochs, using a combined LPIPS and MSE loss (equal weights) on the Kaggle Human Faces dataset. The encoder remains frozen, and decoder blocks are gradually unfrozen in 10 phases with adaptive learning rates. MTCNN-based face weighting is optional. Expect detailed training logs, phase-wise model checkpoints, and loss curves (saved every 10 epochs) in the `progressive_unfreezing_vae_lpips_100` directory. The script provides phase-wise and global best validation losses.

**Key Features**:
- Progressive decoder unfreezing
- Combined LPIPS + MSE loss
- Adaptive learning rate schedule

# Experiment 3: Full Decoder Training with SSIM-Weighted Loss

This script trains the entire VAE decoder (`EPFL-VILAB/flextok_vae_c8`) on the Kaggle Human Faces dataset for 50 epochs, using a combined LPIPS and MSE loss with SSIM-based pixel weighting. The encoder is frozen, and SSIM weighting emphasizes regions with low structural similarity. MTCNN face weighting is configurable via `config.yaml`. Expect training state checkpoints (every 5 epochs), loss curves (every 10 epochs), and SSIM weight visualizations in the `ssim_weighted_vae` directory. The script outputs the best validation loss and model artifacts.

**Key Features**:
- SSIM-based loss weighting
- Configurable via YAML
- Visualization of SSIM weight maps

# Experiment 4: Full Decoder Training with JND-Weighted Loss

This script trains the full VAE decoder (`EPFL-VILAB/flextok_vae_c8`) on the Kaggle Human Faces dataset for 50 epochs, using a combined LPIPS and MSE loss with Just Noticeable Difference (JND) weighting. The encoder is frozen, and JND weighting prioritizes complex image regions. MTCNN face weighting is optional via `config.yaml`. Expect training state checkpoints (every 5 epochs), loss curves (every 10 epochs), and JND weight visualizations in the `jnd_weighted_vae` directory. The script provides the best validation loss and model artifacts.

**Key Features**:
- JND-based loss weighting
- Configurable via YAML
- Visualization of JND weight maps

# Inference: Comprehensive Reconstruction Analysis

This script evaluates a trained VAE model (e.g., from Experiment 2) on a single image, performing reconstruction and computing structural similarity metrics (SSIM, PSNR, etc.) for both the full image and detected face regions using MTCNN. It generates a visualization with original, reconstructed, and difference images, along with face bounding boxes and metrics. Expect output images and metrics saved in the specified directory (e.g., `JND - Part 1`). The script is configured via `config.yaml` and focuses on SSIM and Face-SSIM metrics.

**Key Features**:
- Comprehensive structural similarity analysis
- Face-specific metrics using MTCNN
- Visualization of reconstructions and metrics