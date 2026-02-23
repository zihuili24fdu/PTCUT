#!/bin/bash

# Pix2Pix训练脚本 - GNB 2-Class 节细胞神经母细胞瘤 (Baseline)
# 
# 数据集: patches_1024 (32,744张 1024×1024图像)
# 注意: Pix2Pix需要配对数据，使用aligned数据集模式
#
# 类别:
# - i (intermixed/composite): 复合型
# - n (nodular): 结节型

cd /home/lzh/myCode/PTCUT

# 训练配置
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024"
NAME="gnb_pix2pix"
GPU_IDS="3"

# 超参数 (与PTCUT保持一致)
LOAD_SIZE=512
CROP_SIZE=512
BATCH_SIZE=2
N_EPOCHS=80
N_EPOCHS_DECAY=20

# Pix2Pix损失权重
LAMBDA_L1=100.0     # L1重建损失权重

echo "================================================================"
echo "Pix2Pix训练 (Baseline) - GNB 2-Class 节细胞神经母细胞瘤"
echo "================================================================"
echo "数据集目录: $DATAROOT"
echo "实验名称: $NAME"
echo "图像尺寸: ${LOAD_SIZE}×${CROP_SIZE}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
echo "Lambda L1: $LAMBDA_L1"
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
    --model pix2pix \
    --direction AtoB \
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
    --lambda_L1 $LAMBDA_L1 \
    --gan_mode vanilla \
    --preprocess none \
    --no_flip \
    --epoch_count 1 \
    --save_epoch_freq 10 \
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
echo "  bash test_pix2pix_gnb.sh"
echo "================================================================"