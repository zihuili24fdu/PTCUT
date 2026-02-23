#!/bin/bash

# CUT训练脚本 - GNB 2-Class 节细胞神经母细胞瘤 (Baseline)
# 
# 数据集: patches_1024 (32,744张 1024×1024图像)
# CUT: Contrastive Learning for Unpaired Image-to-Image Translation
#
# 类别:
# - i (intermixed/composite): 复合型
# - n (nodular): 结节型

cd /home/lzh/myCode/PTCUT

# 训练配置
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024"
NAME="gnb_cut_256"
GPU_IDS="0"

# 超参数 (与PTCUT保持一致)
LOAD_SIZE=1024
CROP_SIZE=256
BATCH_SIZE=4
N_EPOCHS=30
N_EPOCHS_DECAY=10

# CUT损失权重 (与PTCUT保持一致)
LAMBDA_GAN=1.0      # GAN损失
LAMBDA_NCE=1.0      # NCE对比损失

echo "================================================================"
echo "CUT训练 (Baseline) - GNB 2-Class 节细胞神经母细胞瘤"
echo "================================================================"
echo "数据集目录: $DATAROOT"
echo "实验名称: $NAME"
echo "图像尺寸: ${LOAD_SIZE}×${CROP_SIZE}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
echo "Lambda GAN: $LAMBDA_GAN"
echo "Lambda NCE: $LAMBDA_NCE"
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

# 开始训练
echo ""
echo "开始训练..."
echo "================================================================"
echo ""

python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model cut \
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
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --lr 0.0002 \
    --beta1 0.5 \
    --lambda_GAN $LAMBDA_GAN \
    --lambda_NCE $LAMBDA_NCE \
    --preprocess resize_and_crop \
    --nce_idt \
    --no_flip \
    --epoch_count 1 \
    --save_epoch_freq 5 \
    --display_freq 100 \
    --update_html_freq 500 \
    --print_freq 50

echo ""
echo "================================================================"
echo "训练完成！"
echo "================================================================"
echo "检查点保存位置: ./checkpoints/$NAME/"
echo "TensorBoard日志: tensorboard --logdir=./checkpoints/$NAME/"
echo ""
echo "测试模型:"
echo "  bash test_cut_gnb.sh"
echo "================================================================"