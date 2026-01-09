"""
Loss functions and utilities for VAE training with knowledge distillation.
"""

import torch
import torch.nn.functional as F
import lpips
from torchvision import transforms


def compute_kl_loss(encoded_output):
    if hasattr(encoded_output, 'latent_dist'):
        try:
            mean = encoded_output.latent_dist.mean

            kl_loss = 0.5 * torch.mean(mean.pow(2))
            return kl_loss

        except Exception as e:
            print(f" Again!!!!!!!! KL loss computation failed: {e}")
            return torch.tensor(0.0, device=mean.device if 'mean' in locals() else 'cuda')
    else:
        return torch.tensor(0.0, device='cuda')


def compute_distillation_loss(teacher_encoded, student_encoded, temperature=16.0, method='kl'):
    """
    Compute distillation loss between teacher and student encoder outputs
    
    Args:
        teacher_encoded: Teacher encoder output
        student_encoded: Student encoder output  
        temperature: Temperature scaling factor
        method: Distillation method ('kl', 'normalized_kl', 'mse', 'cosine')
    """
    if not (hasattr(teacher_encoded, 'latent_dist') and hasattr(student_encoded, 'latent_dist')):
        device = teacher_encoded.latent_dist.mean.device if hasattr(teacher_encoded, 'latent_dist') else 'cuda'
        return torch.tensor(0.0, device=device)
    
    # print(f"Using distillation method: {method}")
    teacher_mean = teacher_encoded.latent_dist.mean
    teacher_logvar = teacher_encoded.latent_dist.logvar
    student_mean = student_encoded.latent_dist.mean
    student_logvar = student_encoded.latent_dist.logvar
    
    if teacher_mean.shape != student_mean.shape:
        # Always upsample student to teacher size
        student_mean = F.interpolate(student_mean, size=teacher_mean.shape[-2:], mode='bicubic', align_corners=False)
        student_logvar = F.interpolate(student_logvar, size=teacher_mean.shape[-2:], mode='bicubic', align_corners=False)
    
    if method == 'kl':
        return _compute_kl_distillation_loss(teacher_mean, teacher_logvar, student_mean, student_logvar, temperature)
    elif method == 'normalized_kl':
        return _compute_normalized_kl_distillation_loss(teacher_mean, teacher_logvar, student_mean, student_logvar, temperature)
    elif method == 'mse':
        return _compute_mse_distillation_loss(teacher_mean, student_mean, student_logvar, teacher_logvar, temperature)
    elif method == 'cosine':
        return _compute_cosine_distillation_loss(teacher_mean, student_mean, temperature)
    else:
        raise ValueError(f"Unknown distillation method: {method}")


def _compute_kl_distillation_loss(teacher_mean, teacher_logvar, student_mean, student_logvar, temperature):
    """Original KL divergence distillation loss"""
    teacher_logvar = torch.clamp(teacher_logvar, min=-10, max=10)
    student_logvar = torch.clamp(student_logvar, min=-10, max=10)
    
    var_teacher = torch.exp(teacher_logvar) + 1e-8
    var_student = torch.exp(student_logvar) + 1e-8
    
    # KL divergence: KL(student || teacher)
    mean_diff = student_mean - teacher_mean
    kl_div = 0.5 * (teacher_logvar - student_logvar + 
                   (var_student + mean_diff**2) / var_teacher - 1)
    
    kl_div = kl_div / (temperature ** 2)
    kl_div = kl_div * 0.01
    kl_div = torch.clamp(kl_div, min=-10, max=10)
    
    distillation_loss = kl_div.mean()
    
    return distillation_loss


def _compute_normalized_kl_distillation_loss(teacher_mean, teacher_logvar, student_mean, student_logvar, temperature):
    """KL divergence distillation loss with normalized latents (both means and log variances)"""
    teacher_mean_norm = F.normalize(teacher_mean.view(teacher_mean.size(0), -1), p=2, dim=1)
    student_mean_norm = F.normalize(student_mean.view(student_mean.size(0), -1), p=2, dim=1)
    
    teacher_mean_norm = teacher_mean_norm.view(teacher_mean.shape)
    student_mean_norm = student_mean_norm.view(student_mean.shape)
    

    teacher_logvar_flat = teacher_logvar.view(teacher_logvar.size(0), -1)
    student_logvar_flat = student_logvar.view(student_logvar.size(0), -1)
    
    teacher_logvar_std = (teacher_logvar_flat - teacher_logvar_flat.mean(dim=1, keepdim=True)) / (teacher_logvar_flat.std(dim=1, keepdim=True) + 1e-8)
    student_logvar_std = (student_logvar_flat - student_logvar_flat.mean(dim=1, keepdim=True)) / (student_logvar_flat.std(dim=1, keepdim=True) + 1e-8)
    
    teacher_logvar_norm = teacher_logvar_std.view(teacher_logvar.shape)
    student_logvar_norm = student_logvar_std.view(student_logvar.shape)
    
    var_teacher = torch.exp(teacher_logvar_norm) + 1e-8
    var_student = torch.exp(student_logvar_norm) + 1e-8
    
    mean_diff = student_mean_norm - teacher_mean_norm
    kl_div = 0.5 * (teacher_logvar_norm - student_logvar_norm + 
                   (var_student + mean_diff**2) / var_teacher - 1)
    
    distillation_loss = kl_div.mean()
    
    # Additional safety check
    if torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
        print(f"Warning: Normalized KL distillation loss is {distillation_loss}, using MSE fallback")
        distillation_loss = F.mse_loss(student_mean_norm, teacher_mean_norm) / (temperature ** 2)
    
    return distillation_loss


def _compute_mse_distillation_loss(teacher_mean, student_mean, student_logvar, teacher_logvar, temperature):
    """Simplified MSE-based distillation loss"""
    mse_loss = F.mse_loss(student_mean, teacher_mean)
    mse_variance = F.mse_loss(student_logvar, teacher_logvar)
    distillation_loss = mse_loss + mse_variance
    distillation_loss = distillation_loss / (temperature)
    return distillation_loss


def _compute_cosine_distillation_loss(teacher_mean, student_mean, temperature):
    """Cosine similarity-based distillation loss"""
    # Flatten spatial dimensions for cosine similarity computation
    teacher_flat = teacher_mean.view(teacher_mean.size(0), -1)
    student_flat = student_mean.view(student_mean.size(0), -1)
    cosine_sim = F.cosine_similarity(teacher_flat, student_flat, dim=1)
    
    # Convert similarity to loss (1 - cosine_similarity)
    cosine_loss = 1.0 - cosine_sim.mean()
    distillation_loss = cosine_loss 
    # Scale by temperature
    
    return distillation_loss


def weighted_mse_loss(input, target, weight_map):
    """Weighted MSE loss with proper dimension handling"""
    weight_map = weight_map.unsqueeze(1)  # [B, 1, H, W]
    weight_map = weight_map.expand(-1, input.shape[1], -1, -1)  # [B, C, H, W]

    squared_diff = (input - target) ** 2
    weighted_squared_diff = weight_map * squared_diff

    loss = weighted_squared_diff.sum() / weight_map.sum()
    return loss


def weighted_lpips_loss(lpips_fn, input, target, weight_map):
    """Weighted LPIPS loss with face weighting"""
    lpips_losses = []

    for i in range(input.shape[0]):
        lpips_val = lpips_fn(input[i:i + 1], target[i:i + 1])
        avg_face_weight = weight_map[i].mean()
        weighted_lpips = lpips_val * avg_face_weight
        lpips_losses.append(weighted_lpips)

    return torch.stack(lpips_losses).mean()


def combined_loss(lpips_fn, input, target, encoded_output=None, weight_map=None,
                  lpips_weight=1.0, mse_weight=1.0, kl_weight=0.01, use_face_weighting=False,
                  teacher_encoded=None, distillation_weight=0.0001, distillation_method='kl',
                  use_lpips_loss=True, use_mse_loss=True, use_kl_loss=True, use_distillation_loss=True,
                  teacher_vae=None, student_vae=None, images_512=None, images_256=None,
                  use_feature_distillation=False, feature_distillation_weight=0.1, 
                  feature_distillation_type='last_layer', feature_distillation_method='cosine'):
    """
    Combined LPIPS + MSE + KL + Distillation + Feature Distillation loss with configurable components
    
    Args:
        distillation_method: Method for distillation loss ('kl', 'normalized_kl', 'mse', 'cosine')
        feature_distillation_type: Type of features ('last_layer' or 'sampled')
        feature_distillation_method: Method for feature distillation ('cosine', 'mse', 'kl')
    """
    # Initialize losses
    lpips_loss = torch.tensor(0.0, device=input.device)
    mse_loss = torch.tensor(0.0, device=input.device)
    kl_loss = torch.tensor(0.0, device=input.device)
    distillation_loss = torch.tensor(0.0, device=input.device)
    feature_distillation_loss = torch.tensor(0.0, device=input.device)

    # LPIPS loss (perceptual)
    if use_lpips_loss:
        if use_face_weighting and weight_map is not None:
            lpips_loss = weighted_lpips_loss(lpips_fn, input, target, weight_map)
        else:
            lpips_loss = lpips_fn(input, target).mean()

    # MSE loss (pixel-wise)
    if use_mse_loss:
        if use_face_weighting and weight_map is not None:
            mse_loss = weighted_mse_loss(input, target, weight_map)
        else:
            mse_loss = F.mse_loss(input, target, reduction='mean')

    # KL loss (regularization)
    if use_kl_loss and encoded_output is not None:
        kl_loss = compute_kl_loss(encoded_output)

    # Distillation loss (teacher-student)
    if use_distillation_loss and teacher_encoded is not None and encoded_output is not None:
        distillation_loss = compute_distillation_loss(teacher_encoded, encoded_output, method=distillation_method)
        
        # Debug: Print distillation loss if it's unusually high
        if distillation_loss.item() > 1000:
            print(f"Warning: High distillation loss detected: {distillation_loss.item():.6f}")
            print(f"Teacher mean range: [{teacher_encoded.latent_dist.mean.min().item():.4f}, {teacher_encoded.latent_dist.mean.max().item():.4f}]")
            print(f"Student mean range: [{encoded_output.latent_dist.mean.min().item():.4f}, {encoded_output.latent_dist.mean.max().item():.4f}]")
            print(f"Teacher logvar range: [{teacher_encoded.latent_dist.logvar.min().item():.4f}, {teacher_encoded.latent_dist.logvar.max().item():.4f}]")
            print(f"Student logvar range: [{encoded_output.latent_dist.logvar.min().item():.4f}, {encoded_output.latent_dist.logvar.max().item():.4f}]")

    # Feature distillation loss (teacher-student features)
    if use_feature_distillation and teacher_vae is not None and student_vae is not None and images_512 is not None and images_256 is not None:
        feature_distillation_loss = compute_feature_distillation_loss(
            teacher_vae, student_vae, images_512, images_256, 
            feature_type=feature_distillation_type,
            method=feature_distillation_method
        )

    # Combine losses
    total_loss = (lpips_weight * lpips_loss + mse_weight * mse_loss + 
                  kl_weight * kl_loss + distillation_weight * distillation_loss + 
                  feature_distillation_weight * feature_distillation_loss)

    return total_loss, lpips_loss, mse_loss, kl_loss, distillation_loss, feature_distillation_loss


def extract_last_encoder_features(encoder, images, requires_grad=False):
    """
    Extract features from the last encoder layer using hooks
    
    Args:
        encoder: VAE encoder model
        images: Input images [B, C, H, W]
        requires_grad: Whether to allow gradients to flow through (True for student, False for teacher)
    
    Returns:
        features: Last encoder layer features [B, C, H, W]
    """
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output
    
    # Register hook on the last down block
    last_layer = encoder.down_blocks[-1]
    hook = last_layer.register_forward_hook(hook_fn)
    
    # Forward pass (this will trigger the hook)
    # Only use no_grad if gradients are not required (for teacher)
    if requires_grad:
        encoder(images)  # Allow gradients to flow through
    else:
        with torch.no_grad():
            encoder(images)  # No gradients needed for frozen teacher
    
    # Remove hook
    hook.remove()
    
    return features


def extract_sampled_latent_features(encoded_output):
    """
    Extract sampled latent features from encoder output
    
    Args:
        encoded_output: Encoder output with latent_dist
    
    Returns:
        features: Sampled latent features [B, C, H, W]
    """
    if hasattr(encoded_output, 'latent_dist'):
        return encoded_output.latent_dist.sample()
    else:
        return encoded_output


def compute_feature_distillation_loss(teacher_vae, student_vae, images_512, images_256, 
                                    feature_type='last_layer', method='cosine', 
                                    temperature=1.0):
    """
    Compute feature-based distillation loss
    
    Args:
        teacher_vae: Teacher VAE model
        student_vae: Student VAE model
        images_512: Teacher input images [B, C, 512, 512]
        images_256: Student input images [B, C, 256, 256]
        feature_type: Type of features to extract ('last_layer' or 'sampled')
        method: Feature distillation method ('cosine', 'mse', 'kl')
        temperature: Temperature scaling factor
    
    Returns:
        feature_distillation_loss: Computed feature distillation loss
    """
    if feature_type == 'last_layer':
        # Extract features from last encoder layer
        # Teacher is frozen, so no gradients needed; Student needs gradients for backprop
        teacher_features = extract_last_encoder_features(teacher_vae.encoder, images_512, requires_grad=False)
        student_features = extract_last_encoder_features(student_vae.encoder, images_256, requires_grad=True)
       
    elif feature_type == 'sampled':
        # Extract sampled latent features
        # Teacher is frozen, so no gradients needed; Student needs gradients for backprop
        with torch.no_grad():
            teacher_encoded = teacher_vae.encode(images_512)
        student_encoded = student_vae.encode(images_256)  # Allow gradients for student
        teacher_features = extract_sampled_latent_features(teacher_encoded)
        student_features = extract_sampled_latent_features(student_encoded)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    # print(f"Teacher features shape: {teacher_features.shape}")
    # print(f"Student features shape: {student_features.shape}")
    if teacher_features is None or student_features is None:
        return torch.tensor(0.0, device=images_512.device)
    
    # Handle size mismatch - always upsample student to teacher size
    if teacher_features.shape != student_features.shape:
        student_features = F.interpolate(student_features, size=teacher_features.shape[-2:], 
                                       mode='bicubic', align_corners=False)
    
    if method == 'cosine':
        return _compute_feature_cosine_loss(teacher_features, student_features, temperature)
    elif method == 'mse':
        return _compute_feature_mse_loss(teacher_features, student_features, temperature)
    elif method == 'kl':
        return _compute_feature_kl_loss(teacher_features, student_features, temperature)
    else:
        raise ValueError(f"Unknown feature distillation method: {method}")


def _compute_feature_cosine_loss(teacher_features, student_features, temperature):
    """Cosine similarity-based feature distillation loss"""
    # Flatten spatial dimensions for cosine similarity computation
    teacher_flat = teacher_features.view(teacher_features.size(0), -1)
    student_flat = student_features.view(student_features.size(0), -1)
    
    # Normalize features
    teacher_flat = F.normalize(teacher_flat, p=2, dim=1)
    student_flat = F.normalize(student_flat, p=2, dim=1)
    
    cosine_sim = F.cosine_similarity(teacher_flat, student_flat, dim=1)
    
    # Convert similarity to loss (1 - cosine_similarity)
    cosine_loss = 1.0 - cosine_sim.mean()
    
    return cosine_loss / (temperature ** 2)


def _compute_feature_mse_loss(teacher_features, student_features, temperature):
    """MSE-based feature distillation loss"""
    mse_loss = F.mse_loss(student_features, teacher_features)
    return mse_loss / (temperature ** 2)


def _compute_feature_kl_loss(teacher_features, student_features, temperature):
    """KL divergence-based feature distillation loss (channel-wise)"""
    # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W]
    teacher_flat = teacher_features.view(teacher_features.size(0), teacher_features.size(1), -1)
    student_flat = student_features.view(student_features.size(0), student_features.size(1), -1)
    
    # Compute statistics for each channel
    teacher_mean = teacher_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    teacher_var = teacher_flat.var(dim=2, keepdim=True)    # [B, C, 1]
    teacher_logvar = torch.log(teacher_var + 1e-8)
    
    student_mean = student_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
    student_var = student_flat.var(dim=2, keepdim=True)     # [B, C, 1]
    student_logvar = torch.log(student_var + 1e-8)
    
    # KL divergence: KL(student || teacher)
    mean_diff = student_mean - teacher_mean
    kl_div = 0.5 * (teacher_logvar - student_logvar + 
                   (student_var + mean_diff**2) / (teacher_var + 1e-8) - 1)
    
    return kl_div.mean() / (temperature ** 2)


def resize_for_student(images, target_size):
    """Resize images to target size for student models"""
    if images.shape[-1] == target_size and images.shape[-2] == target_size:
        return images
    
    # return F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return transforms.Resize(target_size)(images)


##cos: 0.16, mse: 0.64, normalized_kl: 4, 