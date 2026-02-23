#!/bin/bash

# PTCUT训练脚本 - GNB 2-Class 节细胞神经母细胞瘤分类
# 
# 数据集: patches_1024 (32,744张 1024×1024图像)
# 模型: 使用CONCH visual encoder + 预训练的prompt text features
#
# 类别:
# - i (intermixed/composite): 复合型
# - n (nodular): 结节型

cd /home/lzh/myCode/PTCUT

# 训练配置
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"
NAME="gnb_ptcut_cls0.1_clsD1_distill5_size256"
# NAME="gnb_ptcut_cls1_clsD1_distill1_size256"
GPU_IDS="0"

# 超参数
LOAD_SIZE=1024
CROP_SIZE=256
BATCH_SIZE=4
N_EPOCHS=30
N_EPOCHS_DECAY=10

# 损失权重
LAMBDA_CLS=0.1      # 生成器分类损失
LAMBDA_CLS_D=1.0    # 判别器分类损失 (AC-GAN风格，辅助分类器)
LAMBDA_DISTILL=5.0  # 知识蒸馏损失
LAMBDA_GAN=1.0      # GAN损失
LAMBDA_NCE=1.0      # NCE对比损失

echo "================================================================"
echo "PTCUT训练 - GNB 2-Class 节细胞神经母细胞瘤"
echo "================================================================"
echo "数据集目录: $DATAROOT"
echo "CONCH checkpoint: $CONCH_CHECKPOINT"
echo "Prompt features: $PROMPT_FEATURES_PATH"
echo "实验名称: $NAME"
echo "图像尺寸: ${LOAD_SIZE}×${CROP_SIZE}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
echo "================================================================"
echo ""

# 检查数据集
echo "检查数据集..."
if [ ! -d "$DATAROOT" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATAROOT"
    exit 1
fi

# 检查子目录
for subdir in trainA trainB testA testB; do
    if [ ! -d "$DATAROOT/$subdir" ]; then
        echo "❌ 错误: 子目录不存在: $DATAROOT/$subdir"
        exit 1
    fi
    num_files=$(ls -1 "$DATAROOT/$subdir"/*.jpg 2>/dev/null | wc -l)
    echo "  ✓ $subdir: $num_files 张图像"
done

# 检查CONCH和Prompt features
echo ""
echo "检查CONCH和Prompt features..."
if [ ! -f "$CONCH_CHECKPOINT" ]; then
    echo "❌ 错误: CONCH checkpoint不存在: $CONCH_CHECKPOINT"
    exit 1
fi
echo "  ✓ CONCH checkpoint存在"

if [ ! -f "$PROMPT_FEATURES_PATH" ]; then
    echo "❌ 错误: Prompt features不存在: $PROMPT_FEATURES_PATH"
    exit 1
fi
echo "  ✓ Prompt features存在"

# 开始训练
echo ""
echo "开始训练..."
echo "================================================================"
echo ""

python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model ptcut \
    --conch_checkpoint "$CONCH_CHECKPOINT" \
    --prompt_features_path "$PROMPT_FEATURES_PATH" \
    --num_classes 2 \
    --use_labels True \
    --input_nc 3 \
    --output_nc 3 \
    --ngf 64 \
    --ndf 64 \
    --netG resnet_9blocks \
    --netD basic \
    --n_layers_D 3 \
    --gpu_ids $GPU_IDS \
    --load_size $LOAD_SIZE \
    --crop_size $CROP_SIZE \
    --preprocess resize_and_crop \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --lr 0.0002 \
    --beta1 0.5 \
    --lambda_cls $LAMBDA_CLS \
    --lambda_cls_d $LAMBDA_CLS_D \
    --lambda_distill $LAMBDA_DISTILL \
    --lambda_GAN $LAMBDA_GAN \
    --lambda_NCE $LAMBDA_NCE \
    --nce_idt \
    --no_flip \
    --flip_equivariance False \
    --epoch_count 1 \
    --save_epoch_freq 5 \
    --display_freq 100 \
    --update_html_freq 500 \
    --print_freq 50

# 预处理方式说明:
# --preprocess resize_and_crop (默认): 先resize到load_size，再随机裁剪crop_size
# --preprocess crop: 直接随机裁剪，不resize
# --preprocess none: 不做任何预处理
# 
# ⚠️ 重要: PTCUT使用unaligned_dataset，A和B域会各自独立随机裁剪，
#          裁剪区域不同！这适用于无配对训练。
#          如果需要A和B裁剪相同区域，需改用aligned数据集(Pix2Pix风格)

echo ""
echo "================================================================"
echo "训练完成！"
echo "================================================================"
echo "检查点保存位置: ./checkpoints/$NAME/"
echo "TensorBoard日志: tensorboard --logdir=./checkpoints/$NAME/"
echo ""
echo "测试模型:"
echo "  bash test_ptcut_gnb.sh"
echo "================================================================"
