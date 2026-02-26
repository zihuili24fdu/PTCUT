#!/bin/bash

# ==============================================================================
# PTCUT 模型测试与评估脚本 (PTCUT Test & Evaluate)
# ==============================================================================

# 接收 GPU ID 参数，默认使用 0
GPU_IDS=${1:-0}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 1. 核心评估参数配置区
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024_4"

# 实验名称 (必须与你 train.py 中设置的 NAME 保持完全一致)
NAME="gnb_ptcut_cls0.1_clsD0_distill10_size448"

# 测试目标数据集 (可选: test, val, train)
PHASE="test"

# 预处理方式 ("none" 保持 1024 原图推理; "crop" 裁切为 448)
PREPROCESS="none"
LOAD_SIZE=1024
CROP_SIZE=448

# 测试数量 (-1 表示全部测试)
NUM_TEST=-1

# ✨ 核心功能开关 (留空表示关闭，填入对应参数表示开启)
#SAVE_IMAGES="--save_images"
CALC_METRICS="--calc_metrics"
CALC_FID="--calc_fid"

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
echo "功能启用: [存图:$SAVE_IMAGES] [指标:$CALC_METRICS] [FID:$CALC_FID]"
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
    --eval \
    --conch_checkpoint "$CONCH_CHECKPOINT" \
    --prompt_features_path "$PROMPT_FEATURES_PATH" \
    --num_classes 2 \
    --use_labels True \
    $SAVE_IMAGES \
    $CALC_METRICS \
    $CALC_FID

echo "================================================================"
echo "✅ PTCUT 测试流程执行完毕！"
echo "================================================================"