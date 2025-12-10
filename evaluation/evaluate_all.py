"""
图像质量评估全流程工具
集成 PSNR/SSIM 计算、统计分析和可视化功能
一步到位完成从评估到可视化的全部流程
"""

import csv
import math
import os
import functools
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# ============== 配置参数 ==============
# 路径配置
REF_PATH = "/home/lzh/myCode/awesome-virtual-staining/datasets/ttc1/test_latest/images/real_B"
PRED_PATH = "/home/lzh/myCode/awesome-virtual-staining/datasets/ttc1/test_latest/images/fake_B"
OUTPUT_DIR = "/home/lzh/myCode/awesome-virtual-staining/datasets/ttc1/test_latest"
# 评估参数
GRAYSCALE = False  # 是否以灰度模式评估
RESIZE = True  # 若尺寸不同是否自动缩放
EXTS = ".png,.jpg,.jpeg,.tif,.tiff"  # 匹配的图像扩展名
DATA_RANGE = None  # 像素动态范围（None为自动推断）
THREADS = os.cpu_count() or 4  # 使用的线程数

# 输出配置
SAVE_CSV = True  # 是否保存详细CSV结果
SAVE_SUMMARY = True  # 是否保存统计摘要
ENABLE_PLOT = True  # 是否生成可视化图表
SHOW_PLOTS = False  # 是否显示图表（False则只保存）
# ======================================

# 尝试导入依赖
try:
    from skimage.metrics import structural_similarity as ssim
except Exception as e:
    raise RuntimeError("❌ 请安装 scikit-image: pip install scikit-image") from e

try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("❌ 请安装 tqdm: pip install tqdm")

try:
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use('Agg')  # 不显示图形界面
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️  matplotlib 未安装，将跳过可视化功能")


# ==================== PSNR/SSIM 计算模块 ====================

def compute_psnr(ref_img: np.ndarray, pred_img: np.ndarray, data_range: float) -> float:
    """计算PSNR值"""
    diff = ref_img.astype(np.float32) - pred_img.astype(np.float32)
    mse = np.mean(diff ** 2, dtype=np.float64)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(ref_img: np.ndarray, pred_img: np.ndarray, data_range: float) -> float:
    """计算SSIM值"""
    try:
        return ssim(
            ref_img, pred_img, data_range=data_range,
            channel_axis=-1 if ref_img.ndim == 3 else None,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
    except TypeError:
        return ssim(
            ref_img, pred_img, data_range=data_range,
            multichannel=(ref_img.ndim == 3), gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False,
        )


def read_image(path: Path, grayscale: bool) -> np.ndarray:
    """读取图像"""
    if grayscale:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def prepare_pair(
    ref_path: Path, pred_path: Path, grayscale: bool, resize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """准备图像对，必要时进行尺寸调整"""
    ref = read_image(ref_path, grayscale)
    pred = read_image(pred_path, grayscale)
    if ref.shape != pred.shape:
        if not resize:
            raise ValueError(
                f"尺寸不一致: {ref_path.name} {ref.shape} vs {pred_path.name} {pred.shape}"
            )
        h, w = ref.shape[:2]
        interp = cv2.INTER_AREA if pred.shape[1] * pred.shape[0] >= w * h else cv2.INTER_CUBIC
        pred = cv2.resize(pred, (w, h), interpolation=interp)
    return ref, pred


def determine_data_range(img: np.ndarray, user_range: Optional[float]) -> float:
    """确定数据范围"""
    if user_range is not None:
        return float(user_range)
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    max_val = float(np.max(img))
    min_val = float(np.min(img))
    return max(1e-6, max_val - min_val)


def is_image_file(p: Path, exts: set[str]) -> bool:
    """判断是否为图像文件"""
    return p.is_file() and p.suffix.lower() in exts


def collect_pairs(ref_dir: Path, pred_dir: Path, exts: set[str]) -> list[tuple[Path, Path]]:
    """收集匹配的图像对"""
    ref_files = [p for p in ref_dir.iterdir() if is_image_file(p, exts)]
    pred_files = [p for p in pred_dir.iterdir() if is_image_file(p, exts)]
    ref_map = {p.name: p for p in ref_files}
    pred_map = {p.name: p for p in pred_files}
    common_filenames = sorted(set(ref_map.keys()) & set(pred_map.keys()))
    return [(ref_map[name], pred_map[name]) for name in common_filenames]


def process_one_pair(pair: tuple[Path, Path], grayscale: bool, resize: bool, data_range: Optional[float]) -> dict:
    """处理单对图像的函数"""
    ref_path, pred_path = pair
    try:
        ref_img, pred_img = prepare_pair(ref_path, pred_path, grayscale, resize)
        dr = determine_data_range(ref_img, data_range)
        psnr_val = compute_psnr(ref_img, pred_img, dr)
        ssim_val = compute_ssim(ref_img, pred_img, dr)
        h, w = ref_img.shape[:2]
        return {
            'filename': ref_path.name,
            'width': w,
            'height': h,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'success': True
        }
    except Exception as e:
        return {
            'filename': ref_path.name,
            'width': None,
            'height': None,
            'psnr': None,
            'ssim': None,
            'success': False,
            'error': f"{type(e).__name__}: {e}"
        }


# ==================== 统计分析模块 ====================

class MetricsAnalyzer:
    """评估指标分析器"""
    
    def __init__(self, results: List[dict]):
        self.results = results
        self.valid_results = [r for r in results if r['success']]
        # 提取PSNR和SSIM数组
        if self.valid_results:
            self.psnr_array = np.array([r['psnr'] for r in self.valid_results])
            self.ssim_array = np.array([r['ssim'] for r in self.valid_results])
        else:
            self.psnr_array = np.array([])
            self.ssim_array = np.array([])
    
    def basic_statistics(self) -> Dict:
        """计算基本统计信息"""
        if len(self.psnr_array) == 0:
            return {}
        
        stats_dict = {
            'count': len(self.psnr_array),
            'psnr': {
                'mean': float(np.mean(self.psnr_array)),
                'std': float(np.std(self.psnr_array)),
                'min': float(np.min(self.psnr_array)),
                'max': float(np.max(self.psnr_array)),
                'median': float(np.median(self.psnr_array)),
                'q25': float(np.percentile(self.psnr_array, 25)),
                'q75': float(np.percentile(self.psnr_array, 75))
            },
            'ssim': {
                'mean': float(np.mean(self.ssim_array)),
                'std': float(np.std(self.ssim_array)),
                'min': float(np.min(self.ssim_array)),
                'max': float(np.max(self.ssim_array)),
                'median': float(np.median(self.ssim_array)),
                'q25': float(np.percentile(self.ssim_array, 25)),
                'q75': float(np.percentile(self.ssim_array, 75))
            }
        }
        
        return stats_dict
    
    def quality_distribution(self) -> Dict:
        """分析质量分布"""
        if len(self.psnr_array) == 0:
            return {}
        
        # PSNR质量分级
        psnr_excellent = np.sum(self.psnr_array >= 40)
        psnr_good = np.sum((self.psnr_array >= 30) & (self.psnr_array < 40))
        psnr_fair = np.sum((self.psnr_array >= 20) & (self.psnr_array < 30))
        psnr_poor = np.sum(self.psnr_array < 20)
        
        # SSIM质量分级
        ssim_excellent = np.sum(self.ssim_array >= 0.9)
        ssim_good = np.sum((self.ssim_array >= 0.8) & (self.ssim_array < 0.9))
        ssim_fair = np.sum((self.ssim_array >= 0.7) & (self.ssim_array < 0.8))
        ssim_poor = np.sum(self.ssim_array < 0.7)
        
        total = len(self.psnr_array)
        
        return {
            'psnr_distribution': {
                'excellent (≥40dB)': {'count': psnr_excellent, 'percentage': psnr_excellent/total*100},
                'good (30-40dB)': {'count': psnr_good, 'percentage': psnr_good/total*100},
                'fair (20-30dB)': {'count': psnr_fair, 'percentage': psnr_fair/total*100},
                'poor (<20dB)': {'count': psnr_poor, 'percentage': psnr_poor/total*100}
            },
            'ssim_distribution': {
                'excellent (≥0.9)': {'count': ssim_excellent, 'percentage': ssim_excellent/total*100},
                'good (0.8-0.9)': {'count': ssim_good, 'percentage': ssim_good/total*100},
                'fair (0.7-0.8)': {'count': ssim_fair, 'percentage': ssim_fair/total*100},
                'poor (<0.7)': {'count': ssim_poor, 'percentage': ssim_poor/total*100}
            }
        }
    
    def print_statistics(self, stats_dict: Dict):
        """打印统计信息"""
        if not stats_dict:
            print("❌ 没有有效数据可分析")
            return
        
        print("\n" + "="*60)
        print("📊 PSNR/SSIM 统计分析报告")
        print("="*60)
        
        print(f"\n📈 数据概览:")
        print(f"   总记录数: {stats_dict['count']}")
        if len(self.results) - stats_dict['count'] > 0:
            print(f"   失败记录: {len(self.results) - stats_dict['count']}")
        
        print(f"\n🎯 PSNR 统计 (dB):")
        psnr = stats_dict['psnr']
        print(f"   平均值: {psnr['mean']:.4f}")
        print(f"   标准差: {psnr['std']:.4f}")
        print(f"   中位数: {psnr['median']:.4f}")
        print(f"   最小值: {psnr['min']:.4f}")
        print(f"   最大值: {psnr['max']:.4f}")
        print(f"   25分位数: {psnr['q25']:.4f}")
        print(f"   75分位数: {psnr['q75']:.4f}")
        
        print(f"\n🎯 SSIM 统计:")
        ssim_stats = stats_dict['ssim']
        print(f"   平均值: {ssim_stats['mean']:.6f}")
        print(f"   标准差: {ssim_stats['std']:.6f}")
        print(f"   中位数: {ssim_stats['median']:.6f}")
        print(f"   最小值: {ssim_stats['min']:.6f}")
        print(f"   最大值: {ssim_stats['max']:.6f}")
        print(f"   25分位数: {ssim_stats['q25']:.6f}")
        print(f"   75分位数: {ssim_stats['q75']:.6f}")
    
    def print_quality_distribution(self, dist_dict: Dict):
        """打印质量分布"""
        if not dist_dict:
            return
        
        print(f"\n📊 质量分布分析:")
        print(f"\n🎯 PSNR 质量分布:")
        for level, info in dist_dict['psnr_distribution'].items():
            print(f"   {level}: {info['count']} 张图像 ({info['percentage']:.1f}%)")
        
        print(f"\n🎯 SSIM 质量分布:")
        for level, info in dist_dict['ssim_distribution'].items():
            print(f"   {level}: {info['count']} 张图像 ({info['percentage']:.1f}%)")

# ==================== 输出模块 ====================

def save_csv_results(results: List[dict], output_path: Path):
    """保存详细的CSV结果"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "width", "height", "psnr", "ssim"])
        for res in results:
            if res['success']:
                writer.writerow([
                    res['filename'], 
                    res['width'], 
                    res['height'], 
                    f"{res['psnr']:.6f}", 
                    f"{res['ssim']:.6f}"
                ])
            else:
                writer.writerow([
                    res['filename'], 
                    "-", 
                    "-", 
                    "ERROR", 
                    res.get('error', 'Unknown error')
                ])
    print(f"📄 详细结果已保存至: {output_path}")


def save_summary(stats_dict: Dict, dist_dict: Dict, output_path: Path):
    """保存统计摘要"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PSNR/SSIM Statistical Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Data Overview:\n")
        f.write(f"  Total records: {stats_dict.get('count', 0)}\n\n")
        
        f.write("PSNR Statistics (dB):\n")
        psnr = stats_dict.get('psnr', {})
        for key, value in psnr.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nSSIM Statistics:\n")
        ssim_stats = stats_dict.get('ssim', {})
        for key, value in ssim_stats.items():
            f.write(f"  {key}: {value:.6f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Quality Distribution:\n")
        f.write("="*60 + "\n\n")
        if dist_dict:
            f.write("PSNR Quality Distribution:\n")
            for level, info in dist_dict['psnr_distribution'].items():
                f.write(f"  {level}: {info['count']} images ({info['percentage']:.1f}%)\n")
            
            f.write("\nSSIM Quality Distribution:\n")
            for level, info in dist_dict['ssim_distribution'].items():
                f.write(f"  {level}: {info['count']} images ({info['percentage']:.1f}%)\n")
    
    print(f"📄 统计摘要已保存至: {output_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*60)
    print("🎯 图像质量评估全流程工具")
    print("="*60)
    
    # 路径验证
    ref_path = Path(REF_PATH)
    pred_path = Path(PRED_PATH)
    output_dir = Path(OUTPUT_DIR)
    
    if not ref_path.exists():
        print(f"❌ 参考路径不存在: {ref_path}")
        return
    
    if not pred_path.exists():
        print(f"❌ 预测路径不存在: {pred_path}")
        return
    
    if not (ref_path.is_dir() and pred_path.is_dir()):
        print("❌ 路径必须为目录")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 参考图像: {ref_path}")
    print(f"📁 预测图像: {pred_path}")
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    
    # 步骤1: 计算PSNR/SSIM
    print("\n【步骤 1/3】计算 PSNR/SSIM 指标")
    print("-"*60)
    
    exts = {f".{e.strip().lower()}" for e in EXTS.replace('.', '').split(',') if e.strip()}
    pairs = collect_pairs(ref_path, pred_path, exts)
    
    if not pairs:
        print("❌ 未找到任何可匹配的图像对")
        return
    
    print(f"✅ 找到 {len(pairs)} 对可匹配的图像")
    
    task_processor = functools.partial(
        process_one_pair,
        grayscale=GRAYSCALE,
        resize=RESIZE,
        data_range=DATA_RANGE
    )
    
    results = []
    print(f"🚀 使用 {THREADS} 个线程开始处理...")
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        results_iterator = executor.map(task_processor, pairs)
        results = list(tqdm(results_iterator, total=len(pairs), desc="📊 计算进度", unit="张"))
    
    # 步骤2: 统计分析
    print("\n【步骤 2/3】统计分析")
    print("-"*60)
    
    analyzer = MetricsAnalyzer(results)
    stats_dict = analyzer.basic_statistics()
    dist_dict = analyzer.quality_distribution()
    
    analyzer.print_statistics(stats_dict)
    analyzer.print_quality_distribution(dist_dict)
    
    # 步骤3: 生成输出
    print("\n【步骤 3/3】生成输出文件")
    print("-"*60)
    
    if SAVE_CSV:
        save_csv_results(results, output_dir / "results.csv")
    
    if SAVE_SUMMARY:
        save_summary(stats_dict, dist_dict, output_dir / "summary.txt")
    
    if ENABLE_PLOT and HAS_PLOTTING and len(analyzer.psnr_array) > 0:
        create_visualizations(analyzer.psnr_array, analyzer.ssim_array, output_dir)
    elif ENABLE_PLOT and not HAS_PLOTTING:
        print("⚠️  已跳过可视化（matplotlib不可用）")
    
    print("\n" + "="*60)
    print("✨ 全流程评估完成！")
    print("="*60)
    print(f"\n📂 所有结果已保存至: {output_dir}")
    print(f"   • results.csv - 详细数据")
    print(f"   • summary.txt - 统计摘要")
    if ENABLE_PLOT and HAS_PLOTTING:
        print(f"   • *.png - 可视化图表 (5张)")
    print()


if __name__ == "__main__":
    main()

