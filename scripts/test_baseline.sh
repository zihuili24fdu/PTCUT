#!/bin/bash

# ==============================================================================
# 基线模型测试与评估脚本 (Baseline Test & Evaluate)
# 支持模型: pix2pix | cyclegan | cut
# ==============================================================================

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定基线模型。"
    echo "💡 用法: bash test_baseline.sh [pix2pix | cyclegan | cut] [gpu_id(可选,默认0)]"
    exit 1
fi

BASELINE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
GPU_IDS=${2:-0}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 1. 核心评估参数配置区 (在这里修改测试行为)
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_448"

# 测试目标数据集 (可选: test, val, train)
PHASE="test"

# 预处理方式
# - "none": 保持 1024x1024 原图分辨率进行推理和评估
# - "crop": 裁切到训练时的尺寸 (与训练一致，推荐用于最终测试)
PREPROCESS="crop"
LOAD_SIZE=1024
CROP_SIZE=448

# 测试数量 (-1 表示测试文件夹下的所有图像)
NUM_TEST=-1

# ✨ 核心功能开关 (留空表示关闭，填入对应参数表示开启)
#SAVE_IMAGES="--save_images"   # 是否保存推理生成的图片
CALC_METRICS="--calc_metrics" # 是否计算 PSNR, SSIM, Pearson
CALC_FID="--calc_fid"         # 是否计算 FID (比较耗时)

# 若不想执行某项功能，将其注释或设为空即可，例如：
# SAVE_IMAGES=""  # 关闭存图功能，实现极速指标评估

# ==============================================================================
# 2. 动态加载各模型的参数
# ==============================================================================
case $BASELINE in
    pix2pix)
        NAME="gnb_baseline_pix2pix_448"
        MODEL="pix2pix"
        ;;
    cyclegan)
        NAME="gnb_baseline_cyclegan_448"
        MODEL="cycle_gan"
        ;;
    cut)
        NAME="gnb_baseline_cut_448"
        MODEL="cut"
        ;;
    *)
        echo "❌ 错误: 不支持的基线模型 '$BASELINE'"
        exit 1
        ;;
esac

# ==============================================================================
# 3. 启动测试脚本
# ==============================================================================
echo "================================================================"
echo "🚀 开始基线模型测试: $NAME"
echo "使用的 GPU: $GPU_IDS"
echo "测试数据集: $PHASE ($DATAROOT)"
echo "预处理模式: $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE) | 测试数量: $NUM_TEST"
echo "================================================================"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model "$MODEL" \
    --phase "$PHASE" \
    --gpu_ids "$GPU_IDS" \
    --preprocess "$PREPROCESS" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --num_test "$NUM_TEST" \
    --eval \
    $SAVE_IMAGES \
    $CALC_METRICS \
    $CALC_FID

echo "================================================================"
echo "✅ 测试流程执行完毕！"
echo "================================================================"