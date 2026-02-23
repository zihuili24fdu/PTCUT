#!/bin/bash

# CUT测试脚本 - GNB 2-Class 节细胞神经母细胞瘤分类 (Baseline)
# 
# 数据集: patches_1024 (32,744张 1024×1024图像)
# CUT: Contrastive Learning for Unpaired Image-to-Image Translation
#
# 类别:
# - i (intermixed/composite): 复合型
# - n (nodular): 结节型
#
# 使用方法:
#   bash test_cut_gnb.sh [mode]
# 模式:
#   test (默认): 仅运行测试生成图像
#   eval: 仅对已有的测试结果进行评估
#   all: 先测试生成图像，然后评估结果

cd /home/lzh/myCode/PTCUT

# 运行模式
MODE="${1:-test}"  # 默认为 test 模式

# 测试配置
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_1024_4"
NAME="gnb_ptcut_v2_1_0"
GPU_IDS="0"

# 测试参数
LOAD_SIZE=512
CROP_SIZE=512
BATCH_SIZE=2
EPOCH="latest"  # 可以指定具体的epoch，如 "100", "200" 等

# 结果保存目录
RESULTS_DIR="./results/$NAME"

echo "================================================================"
echo "CUT测试 (Baseline) - GNB 2-Class 节细胞神经母细胞瘤"
echo "================================================================"
echo "运行模式: $MODE"
echo "数据集目录: $DATAROOT"
echo "实验名称: $NAME"
echo "测试epoch: $EPOCH"
echo "图像尺寸: ${LOAD_SIZE}×${CROP_SIZE}"
echo "Batch size: $BATCH_SIZE"
echo "结果保存: $RESULTS_DIR"
echo "================================================================"
echo ""

# 验证模式参数
if [[ "$MODE" != "test" && "$MODE" != "eval" && "$MODE" != "all" ]]; then
    echo "❌ 错误: 无效的模式 '$MODE'"
    echo "有效模式: test, eval, all"
    exit 1
fi

# 是否需要运行测试
RUN_TEST=false
# 是否需要运行评估
RUN_EVAL=false

if [[ "$MODE" == "test" ]]; then
    RUN_TEST=true
elif [[ "$MODE" == "eval" ]]; then
    RUN_EVAL=true
elif [[ "$MODE" == "all" ]]; then
    RUN_TEST=true
    RUN_EVAL=true
fi

# 测试阶段
if [ "$RUN_TEST" = true ]; then
    echo "========================================"
    echo "阶段 1: 测试生成图像"
    echo "========================================"
    echo ""
    
    # 检查数据集
    echo "检查数据集..."
    if [ ! -d "$DATAROOT" ]; then
        echo "❌ 错误: 数据集目录不存在: $DATAROOT"
        exit 1
    fi

    # 检查测试集
    for subdir in testA testB; do
        if [ ! -d "$DATAROOT/$subdir" ]; then
            echo "❌ 错误: 测试集目录不存在: $DATAROOT/$subdir"
            exit 1
        fi
        num_files=$(ls -1 "$DATAROOT/$subdir"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ $subdir: $num_files 张图像"
    done

    # 检查模型checkpoint
    echo ""
    echo "检查模型checkpoint..."
    CHECKPOINT_DIR="./checkpoints/$NAME"
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "❌ 错误: Checkpoint目录不存在: $CHECKPOINT_DIR"
        echo "请先运行训练脚本: bash train_cut.sh"
        exit 1
    fi

    if [ "$EPOCH" = "latest" ]; then
        CHECKPOINT_FILE="$CHECKPOINT_DIR/latest_net_G.pth"
    else
        CHECKPOINT_FILE="$CHECKPOINT_DIR/${EPOCH}_net_G.pth"
    fi

    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "❌ 错误: Checkpoint文件不存在: $CHECKPOINT_FILE"
        echo "可用的checkpoints:"
        ls -1 "$CHECKPOINT_DIR"/*_net_G.pth 2>/dev/null || echo "  (无)"
        exit 1
    fi
    echo "  ✓ 使用checkpoint: $CHECKPOINT_FILE"

    # 开始测试
    echo ""
    echo "开始测试..."
    echo "================================================================"
    echo ""

    python test.py \
        --dataroot "$DATAROOT" \
        --name "$NAME" \
        --model cut \
        --input_nc 3 \
        --output_nc 3 \
        --ngf 64 \
        --ndf 64 \
        --netG resnet_9blocks \
        --gpu_ids $GPU_IDS \
        --load_size $LOAD_SIZE \
        --crop_size $CROP_SIZE \
        --batch_size $BATCH_SIZE \
        --epoch "$EPOCH" \
        --no_flip \
        --flip_equivariance False \
        --eval \
        --num_test -1

    echo ""
    echo "================================================================"
    echo "测试完成！"
    echo "================================================================"
    echo ""
fi

# 评估阶段
if [ "$RUN_EVAL" = true ]; then
    echo ""
    echo "========================================"
    echo "阶段 2: 评估结果"
    echo "========================================"
    echo ""
    
    # 检查结果目录是否存在
    RESULT_PATH="$RESULTS_DIR/test_$EPOCH"
    if [ ! -d "$RESULT_PATH/images" ]; then
        echo "❌ 错误: 结果目录不存在: $RESULT_PATH/images"
        echo "请先运行测试生成图像"
        exit 1
    fi
    
    num_images=$(ls -1 "$RESULT_PATH/images"/*.png 2>/dev/null | wc -l)
    echo "✓ 找到测试结果: $num_images 张图像"
    echo ""
    
    echo "开始评估..."
    echo "================================================================"
    echo ""
    
    python evaluate.py \
        --results_dir "$RESULT_PATH" \
        --num_workers 16 \
        --calculate_fid \
        --fid_batch_size 64 \
        --gpu_ids $GPU_IDS
    
    echo ""
    echo "================================================================"
    echo "评估完成！"
    echo "================================================================"
    echo "结果保存位置: $RESULTS_DIR/test_$EPOCH"
    echo ""
    echo "查看生成的图像:"
    echo "  ls $RESULTS_DIR/test_$EPOCH/images/"
fi

echo ""
echo "================================================================"
echo "全部完成！"
echo "================================================================"
