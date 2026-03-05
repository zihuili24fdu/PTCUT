#!/bin/bash

# ==============================================================================
# 虚拟染色多基线训练脚本 (Baseline Training Script)
# 支持模型: pix2pix | cyclegan | cut
# ==============================================================================

if [ -z "$1" ]; then
    echo "❌ 错误: 未指定基线模型。"
    echo "💡 用法: bash run_baselines.sh [pix2pix | cyclegan | cut] [gpu_id(可选, 默认0)]"
    exit 1
fi

BASELINE=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# ---------------------------------------------------------
# ✨ 新增：动态接收第二个参数作为 GPU ID，如果没有传，则默认使用 0
GPU_IDS=${2:-0}
# ---------------------------------------------------------

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 全局公共配置 
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"

# 物理分辨率保持：加载1024，随机裁剪448，保持 0.45um/pixel
LOAD_SIZE=224
CROP_SIZE=224
PREPROCESS="crop"

# 网络基础结构
NET_G="resnet_9blocks"
NET_D="basic"
LR=0.0002

# ==============================================================================
# 动态加载各模型的专属超参数
# ==============================================================================
case $BASELINE in
    pix2pix)
        NAME="gnb_baseline_pix2pix_224"
        MODEL="pix2pix"
        BATCH_SIZE=2
        N_EPOCHS=80
        N_EPOCHS_DECAY=20
        # Pix2Pix: 使用L1损失和标准GAN损失
        EXTRA_ARGS="--dataset_mode aligned --direction AtoB --lambda_L1 100.0 --gan_mode vanilla --pool_size 0 --display_id -1 --no_html"
        echo "🟢 选择基线: Pix2Pix (有监督翻译，L1+GAN)"
        ;;
        
    cyclegan)
        NAME="gnb_baseline_cyclegan_224"
        MODEL="cycle_gan"
        BATCH_SIZE=2  # CycleGAN有两个生成器和判别器，显存消耗大
        N_EPOCHS=30
        N_EPOCHS_DECAY=10
        # CycleGAN: 循环一致性损失
        EXTRA_ARGS="--dataset_mode unaligned --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5 --display_id -1 --no_html"
        echo "🟢 选择基线: CycleGAN (无监督循环一致性翻译)"
        ;;
        
    cut)
        NAME="gnb_baseline_cut_224"
        MODEL="cut"
        BATCH_SIZE=2  # CUT单向结构，省显存，可以开到4
        N_EPOCHS=30
        N_EPOCHS_DECAY=10
        # CUT: 对比学习损失
        EXTRA_ARGS="--dataset_mode unaligned --nce_idt --lambda_GAN 1.0 --lambda_NCE 1.0 --display_id -1 --no_html"
        echo "🟢 选择基线: CUT (无监督对比学习翻译)"
        ;;
        
    *)
        echo "❌ 错误: 不支持的基线模型 '$BASELINE'"
        echo "支持列表: pix2pix, cyclegan, cut"
        exit 1
        ;;
esac

# ==============================================================================
# 打印信息 & 检查数据集结构
# ==============================================================================
echo "================================================================"
echo "准备启动训练..."
echo "实验名称: $NAME"
echo "模型架构: $MODEL (Dataset Mode 自动适配)"
echo "使用的GPU: $GPU_IDS"
echo "图像尺寸: Load $LOAD_SIZE -> Crop $CROP_SIZE ($PREPROCESS)"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
echo "特有参数: $EXTRA_ARGS"
echo "================================================================"

echo "检查数据集..."
if [ ! -d "$DATAROOT" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATAROOT"
    exit 1
fi

# 统一检查 A、B 目录结构
for subdir in trainA trainB testA testB; do
    if [ ! -d "$DATAROOT/$subdir" ]; then
        echo "⚠️ 警告: 子目录不存在: $DATAROOT/$subdir"
    else
        num_files=$(ls -1 "$DATAROOT/$subdir"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ $subdir: $num_files 张图像"
    fi
done

echo ""
echo "🚀 开始训练 $NAME ..."
echo "================================================================"

# ==============================================================================
# 启动训练
# ==============================================================================
python train.py \
    --dataroot "$DATAROOT" \
    --name "$NAME" \
    --model "$MODEL" \
    --input_nc 3 \
    --output_nc 3 \
    --ngf 64 \
    --ndf 64 \
    --netG "$NET_G" \
    --netD "$NET_D" \
    --n_layers_D 3 \
    --gpu_ids "$GPU_IDS" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --preprocess "$PREPROCESS" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$N_EPOCHS" \
    --n_epochs_decay "$N_EPOCHS_DECAY" \
    --lr "$LR" \
    --beta1 0.5 \
    --no_flip \
    --epoch_count 1 \
    --save_epoch_freq 5 \
    --display_freq 100 \
    --update_html_freq 500 \
    --print_freq 50 \
    $EXTRA_ARGS

echo ""
echo "================================================================"
echo "🎉 训练完成 ($NAME)！"
echo "================================================================"
echo "检查点保存位置: ./checkpoints/$NAME/"
echo "TensorBoard日志: tensorboard --logdir=./checkpoints/$NAME/"
echo "================================================================"