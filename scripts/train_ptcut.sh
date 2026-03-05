#!/bin/bash

# ==============================================================================
# PTCUT 训练脚本 - GNB 2-Class 节细胞神经母细胞瘤虚拟染色
# 核心机制: 使用 CONCH (Visual Encoder) + KgCoOp (Prompt Text Features) 进行语义监督
# ==============================================================================

# 支持动态指定 GPU，默认使用 0 号卡
GPU_IDS=${1:-0}
# 继续训练的起始 epoch，不指定则从头开始（例如: bash train_ptcut.sh 0 25 表示从第25轮权重继续训练）
CONTINUE_EPOCH=${2:-""}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 路径与名称配置
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# 实验名称 (根据参数自动更新了后缀尺寸)
NAME="gnb_ptcut_cls0.1_distill10_size224_all_dataset2"

# ==============================================================================
# 基础超参数
# ==============================================================================
# 保持 0.45um/pixel 物理分辨率对齐 CONCH
LOAD_SIZE=224
CROP_SIZE=224
PREPROCESS="crop"

BATCH_SIZE=2
N_EPOCHS=30
N_EPOCHS_DECAY=10
LR=0.0002

# ==============================================================================
# PTCUT 专有损失权重配置
# ==============================================================================
LAMBDA_CLS=0.1      # 1. 生成器分类损失 (基于 CONCH 特征与 Prompt 特征)
LAMBDA_DISTILL=10   # 2. 知识衇馏损失 (同位对齐特征对比)
LAMBDA_GAN=1.0      # 4. 基础 GAN 损失
LAMBDA_NCE=1.0      # 5. 基础 NCE 结构对比损失

# cls 渐进式调度（改进2）
# 前 CLS_WARMUP_EPOCHS 个 epoch：cls 权重=0，让 GAN+NCE 先收敛
# 之后 CLS_RAMPUP_EPOCHS 个 epoch：cls 权重从 0 线性爬坡到 LAMBDA_CLS
CLS_WARMUP_EPOCHS=10   # warmup 期（epoch 数），建议 = N_EPOCHS * 0.33
CLS_RAMPUP_EPOCHS=5    # 爬坡期（epoch 数），建议 = N_EPOCHS * 0.17

# ==============================================================================
# 打印实验信息
# ==============================================================================
echo "================================================================"
echo "🚀 准备启动 PTCUT 训练 (GNB 2-Class)"
echo "================================================================"
echo "使用的 GPU: $GPU_IDS"
echo "实验名称: $NAME"
echo "图像尺寸: Load $LOAD_SIZE -> Crop $CROP_SIZE ($PREPROCESS)"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS (训练) + $N_EPOCHS_DECAY (衰减)"
if [ -n "$CONTINUE_EPOCH" ]; then
    echo "继续训练: 从 Epoch $CONTINUE_EPOCH 的权重文件恢复"
else
    echo "训练模式: 从头开始"
fi
echo "----------------------------------------------------------------"
echo "Loss 权重设定:"
echo "  - GAN: $LAMBDA_GAN | NCE: $LAMBDA_NCE"
echo "  - CLS(生成器): $LAMBDA_CLS"
echo "  - DISTILL(蒸馏): $LAMBDA_DISTILL"
echo "  - CLS 调度: warmup ${CLS_WARMUP_EPOCHS} epochs -> 爬坡 ${CLS_RAMPUP_EPOCHS} epochs -> 满权重"
echo "================================================================"

# ==============================================================================
# 检查依赖文件与数据集
# ==============================================================================
echo "检查文件完整性..."

if [ ! -d "$DATAROOT" ]; then echo "❌ 目录不存在: $DATAROOT"; exit 1; fi
for subdir in trainA trainB testA testB; do
    if [ ! -d "$DATAROOT/$subdir" ]; then 
        echo "⚠️ 警告: 子目录不存在: $DATAROOT/$subdir"
    else
        num_files=$(ls -1 "$DATAROOT/$subdir"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ $subdir: $num_files 张图像"
    fi
done

if [ ! -f "$CONCH_CHECKPOINT" ]; then echo "❌ CONCH 权重丢失: $CONCH_CHECKPOINT"; exit 1; fi
echo "  ✓ CONCH 预训练权重存在"

if [ ! -f "$PROMPT_FEATURES_PATH" ]; then echo "❌ Prompt 特征丢失: $PROMPT_FEATURES_PATH"; exit 1; fi
echo "  ✓ KgCoOp Prompt 文本特征存在"

# 检查继续训练的权重文件是否存在
if [ -n "$CONTINUE_EPOCH" ]; then
    CKPT_DIR="./checkpoints/$NAME"
    # 检查该 epoch 对应的任意一个权重文件（如 G_A）
    SAMPLE_CKPT="$CKPT_DIR/${CONTINUE_EPOCH}_net_G.pth"
    if [ ! -f "$SAMPLE_CKPT" ]; then
        echo "❌ 找不到继续训练所需的权重文件: $SAMPLE_CKPT"
        echo "   请确认 $CKPT_DIR 下存在 ${CONTINUE_EPOCH}_net_*.pth 文件"
        exit 1
    fi
    echo "  ✓ 找到继续训练权重: ${CONTINUE_EPOCH}_net_G.pth"
fi

# ==============================================================================
# 启动训练
# ==============================================================================
echo ""
if [ -n "$CONTINUE_EPOCH" ]; then
    echo "↩️  从 Epoch $CONTINUE_EPOCH 继续训练..."
else
    echo "🚀 开始训练网络..."
fi
echo "================================================================"

# 构建继续训练参数（当指定 CONTINUE_EPOCH 时追加）
CONTINUE_ARGS=""
if [ -n "$CONTINUE_EPOCH" ]; then
    CONTINUE_ARGS="--continue_train --epoch $CONTINUE_EPOCH --epoch_count $((CONTINUE_EPOCH + 1))"
fi

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
    --gpu_ids "$GPU_IDS" \
    --load_size "$LOAD_SIZE" \
    --crop_size "$CROP_SIZE" \
    --preprocess "$PREPROCESS" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$N_EPOCHS" \
    --n_epochs_decay "$N_EPOCHS_DECAY" \
    --lr "$LR" \
    --beta1 0.5 \
    --lambda_cls "$LAMBDA_CLS" \
    --lambda_distill "$LAMBDA_DISTILL" \
    --cls_warmup_epochs "$CLS_WARMUP_EPOCHS" \
    --cls_rampup_epochs "$CLS_RAMPUP_EPOCHS" \
    --lambda_GAN "$LAMBDA_GAN" \
    --lambda_NCE "$LAMBDA_NCE" \
    --nce_idt \
    --no_flip \
    --flip_equivariance False \
    --save_epoch_freq 5 \
    --display_freq 100 \
    --update_html_freq 500 \
    --print_freq 50 \
    $CONTINUE_ARGS

echo ""
echo "================================================================"
echo "🎉 PTCUT 训练完成 ($NAME)！"
echo "================================================================"
echo "检查点保存: ./checkpoints/$NAME/"
echo "TensorBoard日志: tensorboard --logdir=./checkpoints/$NAME/"
echo "================================================================"