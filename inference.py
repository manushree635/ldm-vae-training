import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from diffusers import AutoencoderKL
from mtcnn import MTCNN
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import local_binary_pattern
from scipy.stats import pearsonr
import cv2
import yaml




#FOR THIS ENTIRE CODE, ALL WE CARE ABOUT IS SSIM, FACE-SSIM. (REST ALL, I TRIED EXPERIMENTING)



def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
config = load_config('config.yaml')

# Device setup
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Initialize MTCNN detector
detector = MTCNN()

def calculate_structural_similarity_metrics(img1, img2):
    """
    Calculate comprehensive structural similarity metrics between two images

    Args:
        img1, img2: Images as numpy arrays in range [0, 1] with shape (H, W, C)

    Returns:
        Dictionary containing various similarity metrics
    """
    # Convert to grayscale for some metrics
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray1 = (img1 * 255).astype(np.uint8)
        gray2 = (img2 * 255).astype(np.uint8)

    # Convert to float for calculations
    gray1_float = gray1.astype(np.float64) / 255.0
    gray2_float = gray2.astype(np.float64) / 255.0

    metrics = {}

    # 1. SSIM (Structural Similarity Index) - MOST IMPORTANT
    ssim_score = ssim(gray1_float, gray2_float, data_range=1.0)
    metrics['SSIM'] = ssim_score

    # 2. Multi-scale SSIM
    try:
        ms_ssim_score = ssim(gray1_float, gray2_float, data_range=1.0, multichannel=False,
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        metrics['MS_SSIM'] = ms_ssim_score
    except:
        metrics['MS_SSIM'] = ssim_score

    # 3. PSNR (Peak Signal-to-Noise Ratio)
    psnr_score = psnr(gray1_float, gray2_float, data_range=1.0)
    metrics['PSNR'] = psnr_score

    # 4. Normalized Cross Correlation
    ncc = np.corrcoef(gray1_float.flatten(), gray2_float.flatten())[0, 1]
    metrics['NCC'] = ncc if not np.isnan(ncc) else 0.0

    # 5. Feature-based similarity using Local Binary Patterns
    lbp1 = local_binary_pattern(gray1, P=8, R=1, method='uniform')
    lbp2 = local_binary_pattern(gray2, P=8, R=1, method='uniform')

    # Calculate histogram correlation for LBP
    hist1 = np.histogram(lbp1, bins=10, range=(0, 10))[0]
    hist2 = np.histogram(lbp2, bins=10, range=(0, 10))[0]

    if np.sum(hist1) > 0 and np.sum(hist2) > 0:
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        lbp_correlation = np.corrcoef(hist1_norm, hist2_norm)[0, 1]
        metrics['LBP_Correlation'] = lbp_correlation if not np.isnan(lbp_correlation) else 0.0
    else:
        metrics['LBP_Correlation'] = 0.0

    # 6. Edge-based similarity using Canny edges
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)

    # Calculate edge overlap
    edge_intersection = np.logical_and(edges1 > 0, edges2 > 0)
    edge_union = np.logical_or(edges1 > 0, edges2 > 0)

    if np.sum(edge_union) > 0:
        edge_jaccard = np.sum(edge_intersection) / np.sum(edge_union)
    else:
        edge_jaccard = 1.0

    metrics['Edge_Jaccard'] = edge_jaccard

    # 7. Gradient-based similarity
    grad1_x = cv2.Sobel(gray1_float, cv2.CV_64F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(gray1_float, cv2.CV_64F, 0, 1, ksize=3)
    grad2_x = cv2.Sobel(gray2_float, cv2.CV_64F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(gray2_float, cv2.CV_64F, 0, 1, ksize=3)

    grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
    grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)

    grad_corr = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0, 1]
    metrics['Gradient_Correlation'] = grad_corr if not np.isnan(grad_corr) else 0.0

    # 8. Color similarity (if RGB images)
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        color_mse = np.mean((img1 - img2) ** 2)
        metrics['Color_MSE'] = color_mse

        # Color histogram correlation
        hist1_r = np.histogram(img1[:,:,0], bins=32, range=(0, 1))[0]
        hist1_g = np.histogram(img1[:,:,1], bins=32, range=(0, 1))[0]
        hist1_b = np.histogram(img1[:,:,2], bins=32, range=(0, 1))[0]

        hist2_r = np.histogram(img2[:,:,0], bins=32, range=(0, 1))[0]
        hist2_g = np.histogram(img2[:,:,1], bins=32, range=(0, 1))[0]
        hist2_b = np.histogram(img2[:,:,2], bins=32, range=(0, 1))[0]

        color_corr_r = np.corrcoef(hist1_r, hist2_r)[0, 1] if np.sum(hist1_r) > 0 and np.sum(hist2_r) > 0 else 0
        color_corr_g = np.corrcoef(hist1_g, hist2_g)[0, 1] if np.sum(hist1_g) > 0 and np.sum(hist2_g) > 0 else 0
        color_corr_b = np.corrcoef(hist1_b, hist2_b)[0, 1] if np.sum(hist1_b) > 0 and np.sum(hist2_b) > 0 else 0

        avg_color_corr = np.mean([
            color_corr_r if not np.isnan(color_corr_r) else 0,
            color_corr_g if not np.isnan(color_corr_g) else 0,
            color_corr_b if not np.isnan(color_corr_b) else 0
        ])
        metrics['Color_Histogram_Correlation'] = avg_color_corr

    return metrics

def calculate_face_structural_similarity(original, reconstructed):
    """
    Calculate structural similarity specifically for face regions using MTCNN
    """
    # Detect faces in original image
    faces, confidences, keypoints_list = detect_faces_mtcnn(original)

    if len(faces) == 0:
        print("No faces detected in image")
        return None, None, None

    face_metrics_list = []
    face_regions = []
    face_confidences = []

    for i, (x, y, w, h) in enumerate(faces):
        # Ensure coordinates are within image bounds
        h_img, w_img = original.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        if w <= 10 or h <= 10:  # Skip very small faces
            continue

        # Extract face regions
        orig_face = original[y:y+h, x:x+w]
        recon_face = reconstructed[y:y+h, x:x+w]

        # Calculate structural similarity metrics for this face
        face_metrics = calculate_structural_similarity_metrics(orig_face, recon_face)
        face_metrics_list.append(face_metrics)
        face_regions.append((x, y, w, h))
        face_confidences.append(confidences[i])

        print(f"Face {len(face_metrics_list)} Metrics:")
        for metric, value in face_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    if len(face_metrics_list) == 0:
        print("No valid faces found after coordinate validation")
        return None, None, None

    # Average metrics across all faces
    avg_face_metrics = {}
    for metric in face_metrics_list[0].keys():
        avg_face_metrics[metric] = np.mean([fm[metric] for fm in face_metrics_list])

    print("Average Face Structural Similarity Metrics:")
    for metric, value in avg_face_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return avg_face_metrics, face_regions, face_confidences

def detect_faces_mtcnn(image_array):
    """Detect faces in image using MTCNN and return bounding boxes"""
    # Convert to uint8 for MTCNN
    if image_array.max() <= 1.0:
        img_uint8 = (image_array * 255).astype(np.uint8)
    else:
        img_uint8 = image_array.astype(np.uint8)

    # Detect faces
    result = detector.detect_faces(img_uint8)

    # Extract bounding boxes in (x, y, w, h) format
    faces = []
    confidences = []
    keypoints_list = []

    for face in result:
        x, y, w, h = face['box']
        confidence = face['confidence']
        keypoints = face['keypoints']

        # Ensure positive dimensions
        if w > 0 and h > 0:
            faces.append((x, y, w, h))
            confidences.append(confidence)
            keypoints_list.append(keypoints)

    return faces, confidences, keypoints_list

def visualize_comprehensive_reconstruction(model, image_path, save_path=None, show_plot=True):
    """Visualize reconstruction with comprehensive structural similarity analysis"""

    # Load and preprocess image
    input_tensor = load_and_preprocess_image(image_path)
    if input_tensor is None:
        return

    input_tensor = input_tensor.to(device)

    # Reconstruct image
    model.eval()
    with torch.no_grad():
        try:
            encoded = model.encode(input_tensor)
            if hasattr(encoded, 'latent_dist'):
                latents = encoded.latent_dist.sample()
            else:
                latents = encoded
            reconstructed = model.decode(latents).sample
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            outputs = model(input_tensor, return_dict=True)
            reconstructed = outputs.sample

    # Convert to numpy for visualization
    original = denormalize_image(input_tensor.squeeze(0)).cpu().numpy()
    recon = denormalize_image(reconstructed.squeeze(0)).cpu().numpy()

    # Transpose from CHW to HWC for matplotlib
    original = np.transpose(original, (1, 2, 0))
    recon = np.transpose(recon, (1, 2, 0))

    # Calculate comprehensive structural similarity metrics
    print("=== OVERALL IMAGE STRUCTURAL SIMILARITY ===")
    overall_metrics = calculate_structural_similarity_metrics(original, recon)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\n=== FACE-SPECIFIC STRUCTURAL SIMILARITY ===")
    # Calculate face-specific structural similarity
    face_metrics, face_regions, face_confidences = calculate_face_structural_similarity(original, recon)

    # Create visualization with face boxes and metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Original image with face boxes
    axes[0,0].imshow(original)
    axes[0,0].set_title('Original Image', fontsize=14)
    if face_regions:
        for i, (x, y, w, h) in enumerate(face_regions):
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[0,0].add_patch(rect)
            axes[0,0].text(x, y-5, f'{face_confidences[i]:.2f}',
                          color='red', fontsize=10, weight='bold')
    axes[0,0].axis('off')

    # Reconstructed image with face boxes
    axes[0,1].imshow(recon)
    title = 'Reconstructed Image'
    if face_metrics is not None:
        title += f'\nFace SSIM: {face_metrics["SSIM"]:.4f}'
    axes[0,1].set_title(title, fontsize=14)
    if face_regions:
        for i, (x, y, w, h) in enumerate(face_regions):
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[0,1].add_patch(rect)
            axes[0,1].text(x, y-5, f'{face_confidences[i]:.2f}',
                          color='red', fontsize=10, weight='bold')
    axes[0,1].axis('off')

    # Difference image
    diff_img = np.abs(original - recon)
    axes[1,0].imshow(diff_img)
    axes[1,0].set_title('Absolute Difference', fontsize=14)
    axes[1,0].axis('off')

    # Metrics text
    axes[1,1].axis('off')
    metrics_text = "Overall Structural Similarity:\n\n"
    for metric, value in overall_metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"

    if face_metrics:
        metrics_text += "\n\nFace Structural Similarity:\n\n"
        for metric, value in face_metrics.items():
            metrics_text += f"{metric}: {value:.4f}\n"

    axes[1,1].text(0.05, 0.95, metrics_text, transform=axes[1,1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    if show_plot:
        plt.show()

    return overall_metrics, face_metrics, face_regions, face_confidences

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def denormalize_image(tensor):
    """Convert tensor from [-1, 1] back to [0, 1] for visualization"""
    return (tensor + 1.0) / 2.0

def load_model(model_path):
    """Load your saved VAE model"""
    try:
        vae = AutoencoderKL.from_pretrained(model_path, low_cpu_mem_usage=True)
        vae = vae.to(device)
        print(f"Model loaded successfully from {model_path}")
        return vae
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Loading pretrained model instead...")
        vae = AutoencoderKL.from_pretrained(config['pretrained_vae'], low_cpu_mem_usage=True)
        vae = vae.to(device)
        return vae

# Usage example
if __name__ == "__main__":
    # Load your model
    vae_model = load_model(config['model_path'])

    # Test with face image
    image_path = config['image_path']
    save_path = config['save_path']
    if os.path.exists(image_path):
        overall_metrics, face_metrics, face_regions, face_confidences = visualize_comprehensive_reconstruction(
            vae_model, image_path, save_path
        )

        print(f"\n=== SUMMARY ===")
        print(f"Overall SSIM: {overall_metrics['SSIM']:.6f}")
        print(f"Overall PSNR: {overall_metrics['PSNR']:.6f}")
        if face_metrics is not None:
            print(f"Face SSIM: {face_metrics['SSIM']:.6f}")
            print(f"Face PSNR: {face_metrics['PSNR']:.6f}")
            print(f"Number of faces detected: {len(face_regions)}")