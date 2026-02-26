"""Test and Evaluation script for image-to-image translation.
Supports inference, optional multi-threaded image saving, and in-line metric evaluation.

Usage examples:
1. Only evaluate metrics on test set (Fastest):
   python test.py --dataroot /path/to/data --name model_name --model cut --phase test --calc_metrics

2. Save images AND evaluate metrics on validation set:
   python test.py --dataroot /path/to/data --name model_name --model ptcut --phase val --calc_metrics --save_images

3. Calculate FID and metrics, save images:
   python test.py --dataroot /path/to/data --name model_name --model pix2pix --calc_metrics --calc_fid --save_images
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import cv2
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 1. ==============================================================================
# 预解析自定义参数 (避免和 TestOptions 冲突)
# ==============================================================================
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument('--save_images', action='store_true', help='Toggle to save inference images to disk')
custom_parser.add_argument('--calc_metrics', action='store_true', help='Calculate PSNR, SSIM, Pearson')
custom_parser.add_argument('--calc_fid', action='store_true', help='Calculate FID score')

# 提取自定义参数，剩下的传给 TestOptions
custom_args, remaining_argv = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv

# 导入CUT框架依赖
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# 导入评估指标依赖
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
import torch.nn as nn
from torchvision import models
from scipy import linalg

# 2. ==============================================================================
# 评估指标辅助函数与类
# ==============================================================================
class InceptionV3FeatureExtractor(nn.Module):
    """Inception V3 for FID calculation"""
    def __init__(self):
        super().__init__()
        try:
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        except:
            inception = models.inception_v3(pretrained=True)
        
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
    img = tensor.cpu().float().numpy()
    if img.ndim == 4: img = img[0]
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2.0 * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
    ssim_value, _ = structural_similarity(img1_gray, img2_gray, full=True)
    return ssim_value

def calculate_pearson(img1, img2):
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr

def calculate_fid(real_features, fake_features, eps=1e-6):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# 异步存图任务
def save_images_task(webpage, visuals_cpu, img_path, width):
    save_images(webpage, visuals_cpu, img_path, width=width)

# 3. ==============================================================================
# 主推理流程
# ==============================================================================
if __name__ == '__main__':
    opt = TestOptions().parse()  # 解析标准 CUT 参数
    
    # 强制将预解析的参数注入到 opt 中
    opt.save_images = custom_args.save_images
    opt.calc_metrics = custom_args.calc_metrics
    opt.calc_fid = custom_args.calc_fid
    
    # 测试时的硬编码参数 (确保一对一推理，禁止打乱)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    print(f"\n{'='*80}")
    print(f"Model: {opt.name} | Dataset Phase: {opt.phase} | Max Images: {opt.num_test}")
    print(f"Tasks -> Save Images: {opt.save_images} | Calc Metrics: {opt.calc_metrics} | Calc FID: {opt.calc_fid}")
    print(f"{'='*80}\n")
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    
    # 初始化模型
    first_data = next(iter(dataset))
    model.data_dependent_initialize(first_data)
    model.setup(opt)
    model.parallelize()
    if opt.eval:
        model.eval()
    
    # 存图与 HTML 配置
    webpage = None
    executor = None
    futures = []
    if opt.save_images:
        web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
        os.makedirs(web_dir, exist_ok=True)
        webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
        executor = ThreadPoolExecutor(max_workers=16) # 适当降低线程数防止CPU计算指标时卡顿
        print(f"📁 Images will be saved to: {web_dir}\n")
    
    # 指标计算配置
    inception_model = None
    psnr_scores, ssim_scores, pearson_scores = [], [], []
    real_features, fake_features = [], []
    
    if opt.calc_fid:
        print("⏳ Loading Inception V3 for FID calculation...")
        inception_model = InceptionV3FeatureExtractor()
        if torch.cuda.is_available():
            inception_model = inception_model.cuda()
    
    max_test = len(dataset) if opt.num_test == -1 else min(opt.num_test, len(dataset))
    inference_times = []
    start_time = time.time()
    
    missing_real_B_warned = False
    
    # ==============================================================================
    # 核心测试循环
    # ==============================================================================
    try:
        with torch.no_grad():
            for i, data in enumerate(dataset):
                if i >= max_test:
                    break
                
                # 推理
                t_infer = time.time()
                model.set_input(data)
                model.test()
                inference_times.append(time.time() - t_infer)
                
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()
                
                # ---------------------------------------------------------
                # 任务 1: 保存图片 (多线程异步)
                # ---------------------------------------------------------
                if opt.save_images:
                    visuals_cpu = OrderedDict()
                    for label, tensor in visuals.items():
                        visuals_cpu[label] = tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
                    future = executor.submit(save_images_task, webpage, visuals_cpu, img_path, opt.display_winsize)
                    futures.append(future)
                
                # ---------------------------------------------------------
                # 任务 2: 计算指标 (主线程同步计算)
                # ---------------------------------------------------------
                if opt.calc_metrics or opt.calc_fid:
                    if 'real_B' in visuals and 'fake_B' in visuals:
                        if opt.calc_metrics:
                            real_B_np = tensor_to_numpy(visuals['real_B'])
                            fake_B_np = tensor_to_numpy(visuals['fake_B'])
                            
                            psnr_scores.append(calculate_psnr(real_B_np, fake_B_np))
                            ssim_scores.append(calculate_ssim(real_B_np, fake_B_np))
                            pearson_scores.append(calculate_pearson(real_B_np, fake_B_np))
                            
                        if opt.calc_fid and inception_model is not None:
                            real_B_norm = (visuals['real_B'] + 1) / 2.0
                            fake_B_norm = (visuals['fake_B'] + 1) / 2.0
                            real_features.append(inception_model(real_B_norm).cpu().numpy())
                            fake_features.append(inception_model(fake_B_norm).cpu().numpy())
                    else:
                        if not missing_real_B_warned:
                            print("\n⚠️ WARNING: Cannot calculate metrics because 'real_B' is missing in visuals.")
                            print("   Ensure you are using a paired dataset mode (e.g. unaligned) for testing!\n")
                            missing_real_B_warned = True

                # ---------------------------------------------------------
                # 进度打印
                # ---------------------------------------------------------
                if (i + 1) % 50 == 0 or i == max_test - 1:
                    elapsed = time.time() - start_time
                    actual_speed = elapsed / (i + 1)
                    eta = actual_speed * (max_test - i - 1)
                    
                    status = f"Progress: {i+1}/{max_test} | Speed: {actual_speed:.3f}s/img | ETA: {eta:.1f}s"
                    if opt.save_images:
                        saved_count = sum(1 for f in futures if f.done())
                        status += f" | Saved: {saved_count}/{i+1}"
                    if opt.calc_metrics and len(psnr_scores) > 0:
                        status += f" | Avg PSNR: {np.mean(psnr_scores):.2f}"
                        
                    print(status)
                    
    finally:
        if opt.save_images:
            print("\nWaiting for image save tasks to complete...")
            executor.shutdown(wait=True)
            webpage.save()
            print(f"✅ Images successfully saved to: {web_dir}")

    total_time = time.time() - start_time
    
    # ==============================================================================
    # 汇总统计与输出
    # ==============================================================================
    print(f"\n{'='*80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total processed : {len(inference_times)} images")
    print(f"Total time      : {total_time:.2f} s")
    print(f"Avg inference   : {np.mean(inference_times)*1000:.2f} ms/img")
    print(f"Overall speed   : {len(inference_times)/total_time:.2f} img/s")
    print("-" * 80)
    
    stats = {}
    if opt.calc_metrics and len(psnr_scores) > 0:
        for name, scores in [('PSNR', psnr_scores), ('SSIM', ssim_scores), ('PEARSON', pearson_scores)]:
            stats[name] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
            print(f"{name:7} - Mean: {np.mean(scores):7.4f} | Std: {np.std(scores):7.4f}")
            
    if opt.calc_fid and len(real_features) > 0:
        print("\nCalculating final FID score... (This may take a moment)")
        fid_score = calculate_fid(np.concatenate(real_features, axis=0), np.concatenate(fake_features, axis=0))
        stats['FID'] = float(fid_score)
        print(f"FID     - Score: {fid_score:.4f}  (lower is better)")
        
    # 保存 JSON 日志 (存放在与结果或检查点同级的目录中)
    if stats:
        out_dir = opt.results_dir if opt.results_dir else './results'
        os.makedirs(os.path.join(out_dir, opt.name), exist_ok=True)
        stats_file = os.path.join(out_dir, opt.name, f'eval_stats_{opt.phase}_{opt.epoch}.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\n📝 Metric statistics saved to: {stats_file}")
        
    print(f"{'='*80}\n")