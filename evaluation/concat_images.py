"""
图像横向拼接工具
将不同文件夹中相同文件名的图像按指定顺序横向拼接
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# ============== 配置参数 ==============
BASE_DIR = "/home/lzh/myCode/pytorch-CycleGAN-and-pix2pix/datasets/new53_white2masson/test_latest/images_organized"
OUTPUT_DIR = "/home/lzh/myCode/pytorch-CycleGAN-and-pix2pix/datasets/new53_white2masson/test_latest/images_concatenated"

FOLDERS = ["real_A", "fake_B", "real_B"]  # 拼接顺序：从左到右

ADD_BORDER = True  # 是否在图像之间添加边框
BORDER_WIDTH = 5  # 边框宽度（像素）
BORDER_COLOR = (255, 255, 255)  # 边框颜色 (B, G, R)
ADD_LABELS = True  # 是否在图像上方添加标签
LABEL_HEIGHT = 40  # 标签区域高度
LABEL_BG_COLOR = (240, 240, 240)  # 标签背景颜色
LABEL_TEXT_COLOR = (0, 0, 0)  # 标签文字颜色
# ======================================


def extract_base_name(filename: str, suffix: str) -> str:
    """
    从文件名中提取基础名称（去掉后缀部分）
    例如: L20-0444-1_patch_0_2048_real_A.png + real_A -> L20-0444-1_patch_0_2048
    """
    # 移除文件扩展名
    stem = Path(filename).stem
    
    # 移除后缀部分
    suffix_pattern = f"_{suffix}"
    if stem.endswith(suffix_pattern):
        return stem[:-len(suffix_pattern)]
    
    return stem


def add_label_to_image(img: np.ndarray, label: str, height: int, 
                       bg_color: tuple, text_color: tuple) -> np.ndarray:
    """
    在图像顶部添加标签
    """
    h, w = img.shape[:2]
    
    # 创建标签区域
    label_img = np.full((height, w, 3), bg_color, dtype=np.uint8)
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # 计算文字大小以居中
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (height + text_h) // 2
    
    cv2.putText(label_img, label, (text_x, text_y), font, font_scale, 
                text_color, thickness, cv2.LINE_AA)
    
    # 将标签和图像拼接
    return np.vstack([label_img, img])


def concat_images_horizontal(images: List[np.ndarray], labels: Optional[List[str]] = None,
                             add_border: bool = False, border_width: int = 5,
                             border_color: tuple = (255, 255, 255),
                             add_labels: bool = False, label_height: int = 40,
                             label_bg_color: tuple = (240, 240, 240),
                             label_text_color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    横向拼接多张图像
    
    Args:
        images: 图像列表
        labels: 标签列表（可选）
        add_border: 是否添加边框
        border_width: 边框宽度
        border_color: 边框颜色
        add_labels: 是否添加标签
        label_height: 标签高度
        label_bg_color: 标签背景颜色
        label_text_color: 标签文字颜色
    
    Returns:
        拼接后的图像
    """
    if not images:
        raise ValueError("图像列表为空")
    
    # 确保所有图像都是3通道
    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        processed_images.append(img)
    
    # 统一图像高度（使用最大高度）
    max_height = max(img.shape[0] for img in processed_images)
    resized_images = []
    
    for img in processed_images:
        h, w = img.shape[:2]
        if h != max_height:
            # 保持宽高比缩放
            new_w = int(w * max_height / h)
            img = cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_LINEAR)
        resized_images.append(img)
    
    # 添加标签
    if add_labels and labels:
        labeled_images = []
        for img, label in zip(resized_images, labels):
            img_with_label = add_label_to_image(
                img, label, label_height, label_bg_color, label_text_color
            )
            labeled_images.append(img_with_label)
        resized_images = labeled_images
    
    # 添加边框并拼接
    if add_border and len(resized_images) > 1:
        result_parts = []
        for i, img in enumerate(resized_images):
            result_parts.append(img)
            # 在图像之间添加边框（最后一张图像后不添加）
            if i < len(resized_images) - 1:
                h = img.shape[0]
                border = np.full((h, border_width, 3), border_color, dtype=np.uint8)
                result_parts.append(border)
        result = np.hstack(result_parts)
    else:
        result = np.hstack(resized_images)
    
    return result


def process_concatenation(base_dir: Path, folders: List[str], output_dir: Path,
                         add_border: bool, border_width: int, border_color: tuple,
                         add_labels: bool, label_height: int, 
                         label_bg_color: tuple, label_text_color: tuple):
    """
    批量处理图像拼接
    """
    # 验证文件夹是否存在
    folder_paths = []
    for folder in folders:
        folder_path = base_dir / folder
        if not folder_path.exists():
            print(f"❌ 文件夹不存在: {folder_path}")
            return
        folder_paths.append(folder_path)
    
    print(f"📁 基础目录: {base_dir}")
    print(f"📂 拼接文件夹: {' -> '.join(folders)}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)
    
    # 收集每个文件夹中的文件
    file_maps = []
    for folder, folder_path in zip(folders, folder_paths):
        files = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))
        file_map = {}
        for file in files:
            base_name = extract_base_name(file.name, folder)
            file_map[base_name] = file
        file_maps.append(file_map)
        print(f"  📊 {folder}: {len(file_map)} 个文件")
    
    # 找出所有文件夹中共同的文件名
    common_names = set(file_maps[0].keys())
    for file_map in file_maps[1:]:
        common_names &= set(file_map.keys())
    
    common_names = sorted(common_names)
    
    print(f"\n✅ 找到 {len(common_names)} 个可匹配的文件名")
    
    if not common_names:
        print("❌ 未找到任何可匹配的文件")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量处理
    print(f"\n🚀 开始拼接图像...")
    success_count = 0
    failed_count = 0
    
    for base_name in tqdm(common_names, desc="📊 拼接进度", unit="张"):
        try:
            # 读取所有图像
            images = []
            for file_map in file_maps:
                img_path = file_map[base_name]
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"无法读取图像: {img_path}")
                images.append(img)
            
            # 拼接图像
            concat_img = concat_images_horizontal(
                images=images,
                labels=folders if add_labels else None,
                add_border=add_border,
                border_width=border_width,
                border_color=border_color,
                add_labels=add_labels,
                label_height=label_height,
                label_bg_color=label_bg_color,
                label_text_color=label_text_color
            )
            
            # 保存拼接后的图像
            output_path = output_dir / f"{base_name}_concat.png"
            cv2.imwrite(str(output_path), concat_img)
            success_count += 1
            
        except Exception as e:
            print(f"\n  ❌ 处理失败: {base_name} - {e}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print("✨ 拼接完成！")
    print(f"📊 成功: {success_count} 张")
    if failed_count > 0:
        print(f"⚠️  失败: {failed_count} 张")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)


def main():
    """主函数"""
    base_dir = Path(BASE_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    if not base_dir.exists():
        print(f"❌ 基础目录不存在: {base_dir}")
        print(f"💡 请将 BASE_DIR 变量设置为正确的路径")
        return
    
    print("=" * 60)
    print("🖼️  图像横向拼接工具")
    print("=" * 60)
    print()
    
    process_concatenation(
        base_dir=base_dir,
        folders=FOLDERS,
        output_dir=output_dir,
        add_border=ADD_BORDER,
        border_width=BORDER_WIDTH,
        border_color=BORDER_COLOR,
        add_labels=ADD_LABELS,
        label_height=LABEL_HEIGHT,
        label_bg_color=LABEL_BG_COLOR,
        label_text_color=LABEL_TEXT_COLOR
    )


if __name__ == "__main__":
    main()

