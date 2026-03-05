#!/bin/bash

# ==============================================================================
# PTCUT 模型测试与评估脚本 (PTCUT Test & Evaluate)
# ==============================================================================

# 接收 GPU ID 参数，默认使用 0
GPU_IDS=${1:-0}
EPOCH=${2:-latest}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 1. 核心评估参数配置区
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"

# 实验名称 (必须与你 train.py 中设置的 NAME 保持完全一致)
NAME="gnb_ptcut_cls0.1_distill10_size224_all_dataset2"

# 测试目标数据集 (可选: test, val, train)
PHASE="test"

# 预处理方式 ("none" 保持 1024 原图推理; "crop" 裁切为 448)
PREPROCESS="none"
LOAD_SIZE=224
CROP_SIZE=224

# 测试数量 (-1 表示全部测试)
NUM_TEST=2000

# ✨ 核心功能开关 (留空表示关闭，填入对应参数表示开启)
#SAVE_IMAGES="--save_images"
#CALC_METRICS="--calc_metrics"  # PSNR / SSIM / Pearson
CALC_LPIPS="--calc_lpips"      # LPIPS（感知相似性，越低越好）
LPIPS_NET="alex"                # LPIPS 骨干网络: vgg（语义感知更强） 或 alex（更快）
#CALC_FID="--calc_fid"          # FID  （全图 Inception 距离）
CALC_KID="--calc_kid"           # KID  （内核 Inception 距离，对小样本更鲁棒）
CALC_CROP_FID="--calc_crop_fid" # Crop-FID（随机裁切 FID，评估局部质量）

# Crop-FID 裁切参数（仅当 CALC_CROP_FID 开启时生效）
CROP_FID_SIZE=128  # 裁切尺寸 (px)
CROP_FID_NUM=8    # 每张图随机裁切数量

# ==============================================================================
# 2. PTCUT 专属模型加载配置
# ==============================================================================
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# 检查 PTCUT 必要的特征文件
if [ ! -f "$PROMPT_FEATURES_PATH" ]; then 
    echo "❌ 错误: 找不到 Prompt 特征文件: $PROMPT_FEATURES_PATH"
    exit 1
fi

# ==============================================================================
# 3. 启动测试脚本
# ==============================================================================
echo "================================================================"
echo "🚀 开始 PTCUT 模型测试: $NAME"
echo "使用的 GPU: $GPU_IDS"
echo "测试数据集: $PHASE ($DATAROOT)"
echo "预处理模式: $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE) | 测试数量: $NUM_TEST"
echo "功能启用: [存图:$SAVE_IMAGES] [指标:$CALC_METRICS] [FID:$CALC_FID] [KID:$CALC_KID] [Crop-FID:$CALC_CROP_FID (sz=$CROP_FID_SIZE x$CROP_FID_NUM)]"
echo "================================================================"

python test.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model ptcut \
    --phase "$PHASE" \
    --gpu_ids "$GPU_IDS" \
    --preprocess "$PREPROCESS" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --num_test "$NUM_TEST" \
    --epoch "$EPOCH" \
    --eval \
    --conch_checkpoint "$CONCH_CHECKPOINT" \
    --prompt_features_path "$PROMPT_FEATURES_PATH" \
    --num_classes 2 \
    --use_labels True \
    $SAVE_IMAGES \
    $CALC_METRICS \
    $CALC_LPIPS \
    ${CALC_LPIPS:+--lpips_net ${LPIPS_NET}} \
    $CALC_FID \
    $CALC_KID \
    $CALC_CROP_FID \
    ${CALC_CROP_FID:+--crop_fid_size $CROP_FID_SIZE} \
    ${CALC_CROP_FID:+--crop_fid_num $CROP_FID_NUM}

echo "================================================================"
echo "✅ PTCUT 测试流程执行完毕！"
echo "================================================================"