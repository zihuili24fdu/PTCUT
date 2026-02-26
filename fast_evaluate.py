"""Fast evaluation script for virtual staining models.
Performs inference and evaluation without saving images for maximum speed.
"""
import os
import argparse
import time
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import OrderedDict

# Import test options and model
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

# Import evaluation metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
import cv2
import warnings
warnings.filterwarnings('ignore')

# FID computation
import torch.nn as nn
from torchvision import models, transforms
from scipy import linalg

class InceptionV3FeatureExtractor(nn.Module):
    """Inception V3 for FID calculation"""
    def __init__(self):
        super().__init__()
        try:
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        except:
            inception = models.inception_v3(pretrained=True)
        
        inception.eval()
        self.block = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = 2 * x - 1
        x = self.block(x)
        return x.squeeze(-1).squeeze(-1)

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array (0-255 uint8)"""
    img = tensor.cpu().float().numpy()
    if img.ndim == 4:
        img = img[0]  # Remove batch dimension
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img + 1) / 2.0 * 255.0  # [-1, 1] -> [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    return peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
    ssim_value, _ = structural_similarity(img1_gray, img2_gray, full=True)
    return ssim_value

def calculate_pearson(img1, img2):
    """Calculate Pearson correlation"""
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
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
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def evaluate_model_in_memory(model, val_dataset, num_test=1000, compute_fid=False):
    """
    专为集成到 train.py 中设计的内存级快速评估接口。
    与 fast_evaluate() 不同，本函数不重新加载模型，
    而是直接使用已在 GPU 上的训练模型进行推理，避免 OOM 和重复加载开销。
    """
    print(f"\n[评估模式] 开始在内存中进行快速评估 (最大数量: {num_test})...")

    was_training = getattr(model, 'isTrain', True)
    model.eval()

    inception_model = None
    if compute_fid:
        print("加载 Inception V3 用于计算 FID...")
        inception_model = InceptionV3FeatureExtractor()
        if torch.cuda.is_available():
            inception_model = inception_model.cuda()

    psnr_scores, ssim_scores, pearson_scores = [], [], []
    real_features, fake_features = [], []

    start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataset, total=min(num_test, len(val_dataset)), desc="评估进度")):
            if i >= num_test:
                break

            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()
            real_B_np = tensor_to_numpy(visuals['real_B'])
            fake_B_np = tensor_to_numpy(visuals['fake_B'])

            try:
                psnr_scores.append(calculate_psnr(real_B_np, fake_B_np))
                ssim_scores.append(calculate_ssim(real_B_np, fake_B_np))
                pearson_scores.append(calculate_pearson(real_B_np, fake_B_np))
            except Exception:
                pass

            if compute_fid and inception_model is not None:
                real_B_norm = (visuals['real_B'] + 1) / 2.0
                fake_B_norm = (visuals['fake_B'] + 1) / 2.0
                real_features.append(inception_model(real_B_norm).cpu().numpy())
                fake_features.append(inception_model(fake_B_norm).cpu().numpy())

    total_time = time.time() - start_time

    if inception_model is not None:
        del inception_model
        torch.cuda.empty_cache()

    if was_training:
        if hasattr(model, 'train') and callable(getattr(model, 'train')):
            model.train()
        else:
            # BaseModel 不一定暴露 .train()，手动恢复内部各子网络为训练模式
            for name in getattr(model, 'model_names', []):
                if isinstance(name, str):
                    net = getattr(model, 'net' + name, None)
                    if net is not None:
                        net.train()

    result_str = (
        f"Evaluated {len(psnr_scores)} images in {total_time:.2f}s "
        f"({len(psnr_scores)/max(total_time, 1e-9):.2f} img/s)\n"
    )

    def format_metric(name, scores):
        if not scores:
            return ""
        return (
            f"{name.upper():8} - Mean: {np.mean(scores):7.4f}  "
            f"Std: {np.std(scores):7.4f}  "
            f"Min: {np.min(scores):7.4f}  "
            f"Max: {np.max(scores):7.4f}\n"
        )

    result_str += format_metric('psnr', psnr_scores)
    result_str += format_metric('ssim', ssim_scores)
    result_str += format_metric('pearson', pearson_scores)

    if compute_fid and len(real_features) > 0:
        fid_score = calculate_fid(
            np.concatenate(real_features, axis=0),
            np.concatenate(fake_features, axis=0)
        )
        result_str += f"FID      - Score: {fid_score:.4f}  (lower is better)\n"

    return result_str


def fast_evaluate(opt):
    """Fast evaluation without saving images"""
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {opt.gpu_ids}")
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    # Load model and dataset
    print(f"Loading model: {opt.name}")
    dataset = create_dataset(opt)
    model = create_model(opt)
    
    first_data = next(iter(dataset))
    
    # Clear cache again before initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.data_dependent_initialize(first_data)
    model.setup(opt)
    model.parallelize()
    if opt.eval:
        model.eval()
    
    # Initialize FID model if needed
    inception_model = None
    if opt.calculate_fid:
        print("Loading Inception V3 for FID calculation...")
        inception_model = InceptionV3FeatureExtractor()
        if torch.cuda.is_available():
            inception_model = inception_model.cuda()
    
    # Storage for results
    psnr_scores = []
    ssim_scores = []
    pearson_scores = []
    real_features = []
    fake_features = []
    
    print(f"\nEvaluating {min(opt.num_test, len(dataset))} images...")
    print("=" * 80)
    
    start_time = time.time()
    inference_times = []
    
    # Process images
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, total=min(opt.num_test, len(dataset)), desc="Processing")):
            if i >= opt.num_test:
                break
            
            # Inference
            t_start = time.time()
            model.set_input(data)
            model.test()
            inference_times.append(time.time() - t_start)
            
            # Get results
            visuals = model.get_current_visuals()
            
            # Convert to numpy for metric calculation
            real_B_np = tensor_to_numpy(visuals['real_B'])
            fake_B_np = tensor_to_numpy(visuals['fake_B'])
            
            # Calculate per-image metrics
            try:
                psnr_scores.append(calculate_psnr(real_B_np, fake_B_np))
                ssim_scores.append(calculate_ssim(real_B_np, fake_B_np))
                pearson_scores.append(calculate_pearson(real_B_np, fake_B_np))
            except Exception as e:
                print(f"\nWarning: Failed to calculate metrics for image {i}: {e}")
                continue
            
            # Extract features for FID
            if opt.calculate_fid and inception_model is not None:
                # Normalize to [0, 1] for Inception
                real_B_norm = (visuals['real_B'] + 1) / 2.0
                fake_B_norm = (visuals['fake_B'] + 1) / 2.0
                
                real_feat = inception_model(real_B_norm).cpu().numpy()
                fake_feat = inception_model(fake_B_norm).cpu().numpy()
                
                real_features.append(real_feat)
                fake_features.append(fake_feat)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    print(f"\n{'=' * 80}")
    print("📊 EVALUATION RESULTS")
    print(f"{'=' * 80}\n")
    
    stats = {
        'count': len(psnr_scores),
        'total_time': total_time,
        'avg_inference_time': np.mean(inference_times),
        'throughput': len(psnr_scores) / total_time,
        'metrics': {}
    }
    
    print(f"Total images: {stats['count']}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average inference time: {stats['avg_inference_time']*1000:.2f}ms")
    print(f"Throughput: {stats['throughput']:.2f} img/s")
    print()
    
    for metric_name, scores in [('psnr', psnr_scores), ('ssim', ssim_scores), ('pearson', pearson_scores)]:
        if len(scores) > 0:
            stats['metrics'][metric_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }
            m = stats['metrics'][metric_name]
            print(f"{metric_name.upper():8} - Mean: {m['mean']:7.4f}  Std: {m['std']:7.4f}  "
                  f"Min: {m['min']:7.4f}  Max: {m['max']:7.4f}")
    
    # Calculate FID
    if opt.calculate_fid and len(real_features) > 0:
        print("\nCalculating FID score...")
        real_features_np = np.concatenate(real_features, axis=0)
        fake_features_np = np.concatenate(fake_features, axis=0)
        fid_score = calculate_fid(real_features_np, fake_features_np)
        stats['fid'] = float(fid_score)
        print(f"FID      - Score: {fid_score:.4f}  (lower is better)")
    
    # Save statistics to file if output_dir is specified
    if hasattr(opt, 'output_dir') and opt.output_dir:
        os.makedirs(opt.output_dir, exist_ok=True)
        stats_file = os.path.join(opt.output_dir, 'fast_eval_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n📁 Statistics saved to: {stats_file}")
    
    print(f"\n{'=' * 80}\n")
    
    return stats

if __name__ == '__main__':
    import sys
    
    # Parse custom arguments first
    parser = argparse.ArgumentParser(description='Fast evaluation without saving images')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--name', type=str, required=True, help='Model name')
    parser.add_argument('--model', type=str, required=True, help='Model type')
    parser.add_argument('--epoch', type=str, default='latest', help='Which epoch to load')
    parser.add_argument('--num_test', type=int, default=5000, help='Number of test images')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids')
    parser.add_argument('--preprocess', type=str, default='none', help='Preprocessing')
    parser.add_argument('--load_size', type=int, default=2048, help='Scale images to this size')
    parser.add_argument('--crop_size', type=int, default=1024, help='Crop to this size')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID score')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for statistics')
    
    args = parser.parse_args()
    
    # Modify sys.argv for TestOptions
    sys.argv = [
        'fast_evaluate.py',
        '--dataroot', args.dataroot,
        '--name', args.name,
        '--model', args.model,
        '--epoch', args.epoch,
        '--num_test', str(args.num_test),
        '--gpu_ids', args.gpu_ids,
        '--preprocess', args.preprocess,
        '--load_size', str(args.load_size),
        '--crop_size', str(args.crop_size),
        '--eval',
        '--num_threads', '0',
        '--batch_size', '1',
        '--serial_batches',
        '--no_flip'
    ]
    
    # Parse with TestOptions
    opt = TestOptions().parse()
    opt.calculate_fid = args.calculate_fid
    opt.output_dir = args.output_dir
    
    # Run fast evaluation
    stats = fast_evaluate(opt)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Basic fast evaluation (PSNR, SSIM, Pearson only):
# python fast_evaluate.py  --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class  --name gnb_cut  --model cut --gpu_ids 0 --preprocess resize_and_crop

# python fast_evaluate.py  --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class  --name gnb_cyclegan  --model cycle_gan --gpu_ids 0 --preprocess resize_and_crop

# python fast_evaluate.py  --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class  --name gnb_pix2pix  --model pix2pix --gpu_ids 0 --preprocess resize_and_crop

# With FID calculation:
# python fast_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --calculate_fid

# Save statistics to file:
# python fast_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --calculate_fid \
#   --output_dir ./fast_eval_results

# Test more images:
# python fast_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --num_test 1000 \
#   --calculate_fid

# Full example:
# python fast_evaluate.py \
#   --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 \
#   --name pix2pix \
#   --model pix2pix \
#   --epoch latest \
#   --num_test 500 \
#   --gpu_ids 0 \
#   --preprocess none \
#   --calculate_fid \
#   --output_dir ./results/fast_eval
