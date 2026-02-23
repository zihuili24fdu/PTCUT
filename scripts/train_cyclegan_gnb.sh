#!/bin/bash

# CycleGAN训练脚本 - GNB 2-Class 节细胞神经母细胞瘤 (Baseline)
# 
# 数据集: patches_1024 (32,744张 1024×1024图像)
# 注意: CycleGAN适用于非配对数据，包含循环一致性损失
#
# 类别:
# - i (intermixed/composite): 复合型
# - n (nodular): 结节型

cd /home/lzh/myCode/PTCUT

# 训练配置
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024"
NAME="gnb_cyclegan"
GPU_IDS="1"

# 超参数 (与PTCUT保持一致)
LOAD_SIZE=512
CROP_SIZE=512
BATCH_SIZE=1  # CycleGAN通常使用batch_size=1，因为需要更多显存
N_EPOCHS=30
N_EPOCHS_DECAY=10

# CycleGAN损失权重
LAMBDA_A=10.0       # A域循环一致性损失权重
LAMBDA_B=10.0       # B域循环一致性损失权重
LAMBDA_IDT=0.5      # Identity损失权重 (相对于lambda_A/B)

echo "================================================================"
echo "CycleGAN训练 (Baseline) - GNB 2-Class 节细胞神经母细胞瘤"
echo "================================================================"
echo "数据集目录: $DATAROOT"
echo "实验名称: $NAME"
echo "图像尺寸: ${LOAD_SIZE}×${CROP_SIZE}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
echo "Lambda A/B: $LAMBDA_A / $LAMBDA_B"
echo "Lambda Identity: $LAMBDA_IDT"
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
    --model cycle_gan \
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
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_identity $LAMBDA_IDT \
    --gan_mode lsgan \
    --pool_size 50 \
    --preprocess none \
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
echo "  bash test_cyclegan_gnb"
echo "================================================================"