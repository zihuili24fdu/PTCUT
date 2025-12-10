"""Evaluation script for generated images using multi-process acceleration.
Calculates PSNR, SSIM, Pearson correlation, and FID metrics.
"""
import os
import argparse
import numpy as np
import cv2
import pandas as pd
import json
import time
import warnings
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import linalg

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = structural_similarity(img1_gray, img2_gray, full=True)
    return ssim_value

def calculate_pearsonr(img1, img2):
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr

# ==================== FID Calculation ====================

class InceptionV3FeatureExtractor(nn.Module):
    """Inception V3 model for FID calculation (2048-dim features)"""
    def __init__(self):
        super().__init__()
        try:
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        except:
            inception = models.inception_v3(pretrained=True)
        
        inception.eval()
        
        # Build feature extraction blocks (up to final pooling, 2048-dim)
        self.block = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x):
        # Resize to 299x299
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1]
        x = 2 * x - 1
        # Extract features
        x = self.block(x)
        return x.squeeze(-1).squeeze(-1)

def get_inception_model():
    """Get Inception V3 model for feature extraction"""
    model = InceptionV3FeatureExtractor()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def extract_features_from_images(image_paths, model, batch_size=32):
    """Extract features from images using Inception V3"""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    features = []
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor)
            features.append(batch_features.cpu().numpy())
    
    if features:
        return np.concatenate(features, axis=0)
    return np.array([])

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussian distributions"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(real_features, fake_features):
    """Calculate FID score"""
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score

# ==================== Per-image Evaluation ====================

def evaluate_single_pair(pair):
    """Evaluate single image pair"""
    try:
        real_img = cv2.imread(pair['real_path'])
        fake_img = cv2.imread(pair['fake_path'])
        
        if real_img is None or fake_img is None or real_img.shape != fake_img.shape:
            return None
        
        return {
            'image_name': pair['name'],
            'real_path': pair['real_path'],
            'fake_path': pair['fake_path'],
            'psnr': calculate_psnr(real_img, fake_img),
            'ssim': calculate_ssim(real_img, fake_img),
            'pearson': calculate_pearsonr(real_img, fake_img)
        }
    except:
        return None

def find_image_pairs(results_dir):
    """Find all real_B and fake_B image pairs"""
    images_dir = os.path.join(results_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    pairs = []
    fake_dir = os.path.join(images_dir, 'fake_B')
    real_dir = os.path.join(images_dir, 'real_B')
    
    # Check subdirectory structure
    if os.path.exists(fake_dir) and os.path.exists(real_dir):
        fake_images = sorted([f for f in os.listdir(fake_dir) if f.endswith('.png')])
        for fake_img in fake_images:
            real_path = os.path.join(real_dir, fake_img)
            if os.path.exists(real_path):
                pairs.append({
                    'name': fake_img.replace('.png', ''),
                    'real_path': real_path,
                    'fake_path': os.path.join(fake_dir, fake_img)
                })
    else:
        # Flat structure (old format)
        fake_images = sorted([f for f in os.listdir(images_dir) if f.endswith('_fake_B.png')])
        for fake_img in fake_images:
            base_name = fake_img.replace('_fake_B.png', '')
            real_path = os.path.join(images_dir, base_name + '_real_B.png')
            if os.path.exists(real_path):
                pairs.append({
                    'name': base_name,
                    'real_path': real_path,
                    'fake_path': os.path.join(images_dir, fake_img)
                })
    
    return pairs

def evaluate_images(pairs, num_workers=None):
    """Evaluate all image pairs using multi-processing"""
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 32)
    
    print(f"Evaluating {len(pairs)} images ({num_workers} processes)...\n")
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(evaluate_single_pair, pair): pair for pair in pairs}
        for future in tqdm(as_completed(futures), total=len(pairs), desc="Progress"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    total_time = time.time() - start_time
    print(f"\nCompleted: {len(results)} images in {total_time:.2f}s ({len(results)/total_time:.2f} img/s)\n")
    
    return results

def calculate_statistics(df, fid_score=None):
    """Calculate statistics"""
    stats = {'count': len(df), 'metrics': {}}
    for metric in ['psnr', 'ssim', 'pearson']:
        values = df[metric].values
        stats['metrics'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    if fid_score is not None:
        stats['fid'] = float(fid_score)
    return stats

def save_results(df, stats, output_dir):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed scores
    df.to_csv(os.path.join(output_dir, 'detailed_scores.csv'), index=False, float_format='%.6f')
    
    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to: {output_dir}")

def print_statistics(stats):
    """Print statistics"""
    print(f"\nStatistics ({stats['count']} images):")
    for metric in ['psnr', 'ssim', 'pearson']:
        m = stats['metrics'][metric]
        print(f"  {metric.upper():8} - Mean: {m['mean']:7.4f}  Std: {m['std']:7.4f}  "
              f"Min: {m['min']:7.4f}  Max: {m['max']:7.4f}")
    if 'fid' in stats:
        print(f"  FID      - Score: {stats['fid']:.4f}  (lower is better)")

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated images (multi-process)')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--calculate_fid', action='store_true', 
                        help='Calculate FID score (requires GPU, slower)')
    parser.add_argument('--fid_batch_size', type=int, default=32,
                        help='Batch size for FID calculation')
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'evaluation')
    
    print(f"Input: {args.results_dir}")
    print(f"Output: {args.output_dir}")
    if args.calculate_fid:
        print(f"FID calculation: Enabled")
    print()
    
    # Find and evaluate pairs
    pairs = find_image_pairs(args.results_dir)
    if len(pairs) == 0:
        print("Error: No image pairs found")
        return
    
    print(f"Found {len(pairs)} image pairs")
    results = evaluate_images(pairs, num_workers=args.num_workers)
    
    if len(results) == 0:
        print("Error: No images evaluated successfully")
        return
    
    # Calculate FID if requested
    fid_score = None
    if args.calculate_fid:
        print("\nCalculating FID score...")
        try:
            model = get_inception_model()
            real_paths = [r['real_path'] for r in results]
            fake_paths = [r['fake_path'] for r in results]
            
            print("Extracting features from real images...")
            real_features = extract_features_from_images(real_paths, model, args.fid_batch_size)
            
            print("Extracting features from fake images...")
            fake_features = extract_features_from_images(fake_paths, model, args.fid_batch_size)
            
            if len(real_features) > 0 and len(fake_features) > 0:
                fid_score = calculate_fid(real_features, fake_features)
                print(f"FID Score: {fid_score:.4f}")
            else:
                print("Warning: Could not extract features for FID calculation")
        except Exception as e:
            print(f"Error calculating FID: {e}")
    
    # Calculate and save statistics
    df = pd.DataFrame(results)
    stats = calculate_statistics(df, fid_score)
    save_results(df, stats, args.output_dir)
    print_statistics(stats)
    print()

if __name__ == '__main__':
    main()

# Usage examples:
# Basic evaluation (PSNR, SSIM, Pearson):
# python evaluate.py --results_dir /home/lzh/myCode/awesome-virtual-staining/datasets/pix2pix/test_latest

# With FID calculation:
# python evaluate.py --results_dir /home/lzh/myCode/awesome-virtual-staining/datasets/pix2pix/test_latest --calculate_fid

# With custom workers and batch size:
# python evaluate.py --results_dir /home/lzh/myCode/awesome-virtual-staining/datasets/pix2pix/test_latest --num_workers 16 --calculate_fid --fid_batch_size 64