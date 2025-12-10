"""
FID (Fréchet Inception Distance) 图像质量评估工具
用于评估非配对数据集的生成图像质量
FID值越小表示生成图像分布越接近真实图像分布
"""

import os
import sys
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ============== 配置参数 ==============
# cut white2he white2masson
REAL_PATH = "/home/lzh/myCode/contrastive-unpaired-translation/datasets/new53_white2masson/test_latest/images/real_B"
FAKE_PATH = "/home/lzh/myCode/contrastive-unpaired-translation/datasets/new53_white2masson/test_latest/images/fake_B"
# CycleGAN white2he white2masson
REAL_PATH = "/home/lzh/myCode/pytorch-CycleGAN-and-pix2pix/datasets/new53_white2masson/test_latest/images_organized/real_B"
FAKE_PATH = "/home/lzh/myCode/pytorch-CycleGAN-and-pix2pix/datasets/new53_white2masson/test_latest/images_organized/fake_B"
# my white2he
REAL_PATH = "/home/lzh/myCode/awesome-virtual-staining/datasets/TTC1/test_latest/images/real_B"  
FAKE_PATH = "/home/lzh/myCode/awesome-virtual-staining/datasets/TTC1/test_latest/images/fake_B"  
# cut white2he
REAL_PATH = "/home/lzh/myCode/contrastive-unpaired-translation/datasets/mydatasets_CUT/test_latest/images/real_B" 
FAKE_PATH = "/home/lzh/myCode/contrastive-unpaired-translation/datasets/mydatasets_CUT/test_latest/images/fake_B" 
BATCH_SIZE = 50  # 批处理大小
NUM_WORKERS = 4  # 数据加载线程数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 使用的设备
EXTS = ".png,.jpg,.jpeg,.tif,.tiff"  # 匹配的图像扩展名
DIMS = 2048  # InceptionV3特征维度 (64, 192, 768, 2048)
# ======================================

# 尝试导入依赖
try:
    from scipy import linalg
except ImportError:
    raise RuntimeError("❌ 请安装 scipy: pip install scipy")

try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("❌ 请安装 tqdm: pip install tqdm")

try:
    from torchvision import models, transforms
except ImportError:
    raise RuntimeError("❌ 请安装 torchvision: pip install torchvision")

warnings.filterwarnings('ignore')


class InceptionV3Feature(nn.Module):
    """
    使用预训练的InceptionV3提取特征
    支持多个池化层输出
    """
    
    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        
        # 加载预训练的InceptionV3
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()
        
        # 构建特征提取块
        self.blocks = nn.ModuleList()
        
        # Block 0: 到第一个池化层
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        
        # Block 1: 到第二个池化层
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1))
        
        # Block 2: 到辅助分类器
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2))
        
        # Block 3: 到最终池化层 (2048维特征)
        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3))
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """前向传播"""
        # 调整输入大小到 299x299
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 归一化到 [-1, 1]
        if self.normalize_input:
            x = 2 * x - 1
        
        # 逐块提取特征
        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
        
        return outputs


class ImageDataset(Dataset):
    """图像数据集加载器"""
    
    def __init__(self, image_dir: Path, exts: set[str], transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 收集所有图像文件
        self.image_files = sorted([
            f for f in image_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in exts
        ])
        
        if not self.image_files:
            raise ValueError(f"❌ 在 {image_dir} 中未找到任何图像文件")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            raise RuntimeError(f"❌ 读取图像失败: {img_path}, 错误: {e}")


def get_activations(image_dir: Path, model: nn.Module, batch_size: int, 
                   dims: int, num_workers: int, device: str, exts: set[str]):
    """
    计算图像的Inception特征激活值
    
    Args:
        image_dir: 图像目录
        model: InceptionV3模型
        batch_size: 批大小
        dims: 特征维度
        num_workers: 数据加载线程数
        device: 计算设备
        exts: 图像扩展名集合
    
    Returns:
        activations: (N, dims) 的特征数组
    """
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和加载器
    dataset = ImageDataset(image_dir, exts, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"📁 找到 {len(dataset)} 张图像")
    
    # 提取特征
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="📊 提取特征", unit="batch"):
            batch = batch.to(device)
            features = model(batch)[0]
            
            # 展平特征
            if features.size(2) != 1 or features.size(3) != 1:
                features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
            
            features = features.squeeze(3).squeeze(2)
            activations.append(features.cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    
    # 验证维度
    if activations.shape[1] != dims:
        raise ValueError(f"❌ 特征维度不匹配: 期望 {dims}, 得到 {activations.shape[1]}")
    
    return activations


def calculate_activation_statistics(activations: np.ndarray):
    """
    计算激活值的均值和协方差矩阵
    
    Args:
        activations: (N, dims) 的特征数组
    
    Returns:
        mu: 均值向量
        sigma: 协方差矩阵
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                               mu2: np.ndarray, sigma2: np.ndarray, 
                               eps=1e-6):
    """
    计算两个多元高斯分布之间的Fréchet距离
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: 第一个分布的均值
        sigma1: 第一个分布的协方差矩阵
        mu2: 第二个分布的均值
        sigma2: 第二个分布的协方差矩阵
        eps: 数值稳定性的小常数
    
    Returns:
        fid_score: FID分数
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "均值向量维度不一致"
    assert sigma1.shape == sigma2.shape, "协方差矩阵维度不一致"
    
    # 计算均值差的平方
    diff = mu1 - mu2
    
    # 计算协方差矩阵的乘积平方根
    # 使用数值稳定的方式
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理数值误差导致的虚部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"⚠️  协方差矩阵乘积平方根包含显著虚部 (max={m})")
        covmean = covmean.real
    
    # 计算FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return float(fid)


def compute_fid(real_path: Path, fake_path: Path, batch_size: int, 
               device: str, dims: int, num_workers: int, exts: set[str]):
    """
    计算FID分数
    
    Args:
        real_path: 真实图像目录
        fake_path: 生成图像目录
        batch_size: 批大小
        device: 计算设备
        dims: 特征维度
        num_workers: 线程数
        exts: 图像扩展名集合
    
    Returns:
        fid_score: FID分数
    """
    # 确定使用的block索引
    block_idx_map = {64: 0, 192: 1, 768: 2, 2048: 3}
    if dims not in block_idx_map:
        raise ValueError(f"❌ 不支持的特征维度: {dims}, 支持的维度: {list(block_idx_map.keys())}")
    
    block_idx = block_idx_map[dims]
    
    print(f"🔧 初始化 InceptionV3 模型 (特征维度: {dims})...")
    model = InceptionV3Feature(output_blocks=[block_idx], resize_input=True, normalize_input=True)
    model = model.to(device)
    
    print(f"💻 使用设备: {device}")
    
    # 提取真实图像特征
    print("\n📸 处理真实图像...")
    real_activations = get_activations(
        real_path, model, batch_size, dims, num_workers, device, exts
    )
    
    # 提取生成图像特征
    print("\n🎨 处理生成图像...")
    fake_activations = get_activations(
        fake_path, model, batch_size, dims, num_workers, device, exts
    )
    
    # 计算统计量
    print("\n📊 计算统计量...")
    mu_real, sigma_real = calculate_activation_statistics(real_activations)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_activations)
    
    # 计算FID
    print("🧮 计算 FID 分数...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score


def main():
    """主函数"""
    # 验证路径
    real_path = Path(REAL_PATH)
    fake_path = Path(FAKE_PATH)
    
    if not real_path.exists():
        print(f"❌ 真实图像路径不存在: {real_path}")
        print(f"💡 请将 REAL_PATH 变量设置为正确的路径")
        return
    
    if not fake_path.exists():
        print(f"❌ 生成图像路径不存在: {fake_path}")
        print(f"💡 请将 FAKE_PATH 变量设置为正确的路径")
        return
    
    if not real_path.is_dir() or not fake_path.is_dir():
        print("❌ REAL_PATH 和 FAKE_PATH 必须都是目录")
        return
    
    # 解析扩展名
    exts = {f".{e.strip().lower()}" for e in EXTS.replace('.', '').split(',') if e.strip()}
    
    print("=" * 60)
    print("🎯 FID (Fréchet Inception Distance) 评估")
    print("=" * 60)
    print(f"📁 真实图像目录: {real_path}")
    print(f"📁 生成图像目录: {fake_path}")
    print(f"🔢 批大小: {BATCH_SIZE}")
    print(f"🧵 工作线程数: {NUM_WORKERS}")
    print(f"📐 特征维度: {DIMS}")
    print("=" * 60)
    
    try:
        fid_score = compute_fid(
            real_path=real_path,
            fake_path=fake_path,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            dims=DIMS,
            num_workers=NUM_WORKERS,
            exts=exts
        )
        
        print("\n" + "=" * 60)
        print(f"✨ FID 分数: {fid_score:.4f}")
        print("=" * 60)
        print("\n💡 FID 解读:")
        print("  • FID 值越小越好")
        print("  • FID < 50: 优秀")
        print("  • FID 50-100: 良好")
        print("  • FID 100-200: 一般")
        print("  • FID > 200: 需要改进")
        print("\n✨ 评估完成！")
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

