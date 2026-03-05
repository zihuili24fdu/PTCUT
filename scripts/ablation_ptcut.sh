#!/bin/bash

# ==============================================================================
# PTCUT 消融实验脚本 - 两项损失的全排列消融 (2^2 = 4 组)【并行版】
# 消融变量: LAMBDA_CLS / LAMBDA_DISTILL
# ==============================================================================
#
# 用法:
#   bash ablation_ptcut.sh [GPU_IDS] [EXP_IDS] [CONTINUE_EPOCH]
#
# 参数说明:
#   GPU_IDS        - GPU 编号，逗号分隔，支持多卡轮询分配（默认 "0"）
#                    例: "0,1,2,3" 表示 4 张卡，实验按顺序轮询使用
#   EXP_IDS        - 要运行的实验编号，逗号分隔或 "all"（默认 "all"）
#                    1: cls=0,    distill=0    (纯 CUT 基线)
#                    2: cls=0.1,  distill=0    (+生成器分类)
#                    3: cls=0,    distill=10   (+知识蒸馏)
#                    4: cls=0.1,  distill=10   (完整模型)
#   CONTINUE_EPOCH - 从指定 epoch 继续训练，不指定则从头开始
#
# 示例:
#   bash ablation_ptcut.sh 0 all          # 单卡跑全部 4 组（依次占用 GPU 0）
#   bash ablation_ptcut.sh 0,1,2,3 all   # 4 卡并行，4 组实验轮询 4 张卡
#   bash ablation_ptcut.sh 0,1 1,3       # 双卡并行跑2 组
#   bash ablation_ptcut.sh 0,1 all 25    # 双卡从第 25 轮继续训练全部 4 组
#
# 日志文件: ./logs/ablation/<实验名>.log
# ==============================================================================

GPU_IDS=${1:-0}
EXP_IDS=${2:-"all"}
CONTINUE_EPOCH=${3:-""}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 公共路径配置
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# ==============================================================================
# 公共超参数
# ==============================================================================
LOAD_SIZE=224
CROP_SIZE=224
PREPROCESS="crop"
BATCH_SIZE=2
N_EPOCHS=30
N_EPOCHS_DECAY=10
LR=0.0002
LAMBDA_GAN=1.0
LAMBDA_NCE=1.0

# 各损失的激活值（开启时取此值，关闭时取 0）
CLS_ON=0.1
DISTILL_ON=10

# cls 渐进式调度（与 train_ptcut.sh 保持一致）
# 前 CLS_WARMUP_EPOCHS 个 epoch：cls 权重=0
# 之后 CLS_RAMPUP_EPOCHS 个 epoch：cls 权重从 0 线性爬坡到 CLS_ON
CLS_WARMUP_EPOCHS=10   # 前 N 个 epoch cls 权重=0；当前设置：epoch 1-10 静默
CLS_RAMPUP_EPOCHS=5   # 之后 N 个 epoch 线性爬坡；当前设置：epoch 11-20 爬坡，epoch 21+ 满权重(0.1)

# 实验名称后缀（用于区分不同改进版本的消融组，留空则不追加后缀）
EXP_SUFFIX="all_dataset2"

# ==============================================================================
# 定义4 组消融实验: "CLS_ENABLE DISTILL_ENABLE"
# 1 表示开启，0 表示关闭
# ==============================================================================
declare -A EXP_FLAG
EXP_FLAG[1]="0 0"
EXP_FLAG[2]="1 0"
EXP_FLAG[3]="0 1"
EXP_FLAG[4]="1 1"

# ==============================================================================
# 解析 GPU 列表和实验 ID 列表
# ==============================================================================
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$EXP_IDS" = "all" ]; then
    RUN_LIST="1 2 3 4"
else
    RUN_LIST=$(echo "$EXP_IDS" | tr ',' ' ')
fi

# 校验 ID 范围
for id in $RUN_LIST; do
    if ! [[ "$id" =~ ^[1-4]$ ]]; then
        echo "❌ 无效的实验编号: $id（合法范围 1~4）"
        exit 1
    fi
done

TOTAL=$(echo $RUN_LIST | wc -w)

# ==============================================================================
# 创建日志目录
# ==============================================================================
LOG_DIR="./logs/ablation"
mkdir -p "$LOG_DIR"

# ==============================================================================
# 检查公共依赖文件
# ==============================================================================
echo "================================================================"
echo "🔍 检查公共依赖文件..."
echo "================================================================"

if [ ! -d "$DATAROOT" ]; then echo "❌ 数据集目录不存在: $DATAROOT"; exit 1; fi
for subdir in trainA trainB testA testB; do
    if [ ! -d "$DATAROOT/$subdir" ]; then
        echo "⚠️  警告: 子目录不存在: $DATAROOT/$subdir"
    else
        num_files=$(ls -1 "$DATAROOT/$subdir"/*.jpg 2>/dev/null | wc -l)
        echo "  ✓ $subdir: $num_files 张图像"
    fi
done
if [ ! -f "$CONCH_CHECKPOINT" ]; then echo "❌ CONCH 权重丢失: $CONCH_CHECKPOINT"; exit 1; fi
echo "  ✓ CONCH 预训练权重存在"
if [ ! -f "$PROMPT_FEATURES_PATH" ]; then echo "❌ Prompt 特征丢失: $PROMPT_FEATURES_PATH"; exit 1; fi
echo "  ✓ KgCoOp Prompt 文本特征存在"

# ==============================================================================
# 若指定 CONTINUE_EPOCH，预先检查所有权重文件
# ==============================================================================
if [ -n "$CONTINUE_EPOCH" ]; then
    echo ""
    echo "🔍 检查继续训练权重文件..."
    for id in $RUN_LIST; do
        flags=(${EXP_FLAG[$id]})
        f_cls=${flags[0]}; f_distill=${flags[1]}
        lam_cls=$([ "$f_cls" -eq 1 ]      && echo "$CLS_ON"     || echo "0")
        lam_dist=$([ "$f_distill" -eq 1 ] && echo "$DISTILL_ON" || echo "0")
        exp_name="gnb_ptcut_cls${lam_cls}_distill${lam_dist}_size${CROP_SIZE}${EXP_SUFFIX:+_${EXP_SUFFIX}}"
        ckpt="./checkpoints/${exp_name}/${CONTINUE_EPOCH}_net_G.pth"
        if [ ! -f "$ckpt" ]; then
            echo "❌ [实验 $id] 找不到权重文件: $ckpt"
            echo "   请确认 ./checkpoints/${exp_name} 下存在 ${CONTINUE_EPOCH}_net_*.pth"
            exit 1
        fi
        echo "  ✓ [实验 $id] $exp_name — 找到 ${CONTINUE_EPOCH}_net_G.pth"
    done
fi

# ==============================================================================
# 打印实验计划
# ==============================================================================
echo ""
echo "================================================================"
echo "📋 PTCUT 消融实验计划 (并行模式)"
echo "================================================================"
echo "可用 GPU:       ${GPU_ARRAY[*]}  (共 $NUM_GPUS 张)"
echo "运行实验编号:   $RUN_LIST  (共 $TOTAL 组)"
if [ -n "$CONTINUE_EPOCH" ]; then
    echo "继续训练:       从 Epoch $CONTINUE_EPOCH 恢复"
else
    echo "训练模式:       从头开始"
fi
echo "CLS 调度:       warmup ${CLS_WARMUP_EPOCHS} epochs → 爬坡 ${CLS_RAMPUP_EPOCHS} epochs → 满权重"
echo "日志目录:       $LOG_DIR"
echo "----------------------------------------------------------------"
echo " ID  | GPU | CLS    | DISTILL | 实验名称"
echo "-----+-----+--------+---------+---------------------------------------"
IDX=0
for id in $RUN_LIST; do
    flags=(${EXP_FLAG[$id]})
    f_cls=${flags[0]}; f_distill=${flags[1]}
    lam_cls=$([ "$f_cls" -eq 1 ]      && echo "$CLS_ON"     || echo "0")
    lam_dist=$([ "$f_distill" -eq 1 ] && echo "$DISTILL_ON" || echo "0")
    exp_name="gnb_ptcut_cls${lam_cls}_distill${lam_dist}_size${CROP_SIZE}${EXP_SUFFIX:+_${EXP_SUFFIX}}"
    assigned_gpu=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    printf " %-3s | %-3s | %-6s | %-7s | %s\n" \
        "$id" "$assigned_gpu" "$lam_cls" "$lam_dist" "$exp_name"
    IDX=$((IDX + 1))
done
echo "================================================================"

# ==============================================================================
# 清理函数：收到终止信号时杀死所有后台训练进程
# ==============================================================================
declare -a PIDS        # 各实验的后台 PID（提前声明，供 cleanup 引用）

cleanup() {
    echo ""
    echo "⚠️  收到终止信号，正在停止所有后台训练进程..."
    local killed=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  已终止 PID $pid"
            killed=$((killed + 1))
        fi
    done
    # 等待所有子进程真正退出
    wait 2>/dev/null
    echo "✅ 共终止 $killed 个训练进程，退出"
    exit 1
}

# 捕获：终端关闭(SIGHUP)、Ctrl+C(SIGINT)、kill(SIGTERM)
trap cleanup SIGHUP SIGINT SIGTERM

# ==============================================================================
# 并行启动所有实验
# ==============================================================================
echo ""
echo "🚀 并行启动全部 $TOTAL 个实验..."
echo "================================================================"

declare -a EXP_NAMES   # 各实验名称（与 PIDS 下标对应）
declare -a EXP_GPUS    # 各实验使用的 GPU
declare -a EXP_LOGS    # 各实验日志路径

IDX=0
for id in $RUN_LIST; do
    flags=(${EXP_FLAG[$id]})
    f_cls=${flags[0]}; f_distill=${flags[1]}

    LAMBDA_CLS=$([ "$f_cls" -eq 1 ]        && echo "$CLS_ON"     || echo "0")
    LAMBDA_DISTILL=$([ "$f_distill" -eq 1 ] && echo "$DISTILL_ON" || echo "0")

    NAME="gnb_ptcut_cls${LAMBDA_CLS}_distill${LAMBDA_DISTILL}_size${CROP_SIZE}${EXP_SUFFIX:+_${EXP_SUFFIX}}"
    ASSIGNED_GPU=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    LOG_FILE="${LOG_DIR}/${NAME}.log"

    # 继续训练参数
    CONTINUE_ARGS=""
    if [ -n "$CONTINUE_EPOCH" ]; then
        CONTINUE_ARGS="--continue_train --epoch $CONTINUE_EPOCH --epoch_count $((CONTINUE_EPOCH + 1))"
    fi

    echo "  ▶ 实验 #$id → GPU $ASSIGNED_GPU | $NAME"
    echo "    日志: $LOG_FILE"

    # 写入日志头
    {
        echo "============================================================"
        echo "实验 #$id: $NAME"
        echo "GPU: $ASSIGNED_GPU"
        echo "CLS=$LAMBDA_CLS  DISTILL=$LAMBDA_DISTILL"
        echo "CLS调度: warmup=${CLS_WARMUP_EPOCHS} rampup=${CLS_RAMPUP_EPOCHS}"
        [ -n "$CONTINUE_EPOCH" ] && echo "继续训练: 从 Epoch $CONTINUE_EPOCH"
        echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } > "$LOG_FILE"

    # 后台启动训练（stdout + stderr 均重定向到日志文件）
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
        --gpu_ids "$ASSIGNED_GPU" \
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
        --save_epoch_freq 10 \
        --display_freq 100 \
        --update_html_freq 500 \
        --print_freq 50 \
        $CONTINUE_ARGS \
        >> "$LOG_FILE" 2>&1 &

    PIDS[$IDX]=$!
    EXP_NAMES[$IDX]="$NAME"
    EXP_GPUS[$IDX]="$ASSIGNED_GPU"
    EXP_LOGS[$IDX]="$LOG_FILE"

    IDX=$((IDX + 1))
done

echo ""
echo "✅ 全部 $TOTAL 个实验已在后台启动，等待完成..."
echo "   实时查看日志示例: tail -f ${LOG_DIR}/<实验名>.log"
echo "================================================================"

# ==============================================================================
# 等待所有后台进程完成，收集退出码
# ==============================================================================
FAIL_COUNT=0
declare -a FAILED_NAMES

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${EXP_NAMES[$i]}
    gpu=${EXP_GPUS[$i]}
    log=${EXP_LOGS[$i]}

    wait "$pid"
    EXIT_CODE=$?

    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$log"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✅ 完成: $name  (GPU $gpu)"
    else
        echo "  ❌ 失败: $name  (GPU $gpu, exit code: $EXIT_CODE)"
        FAILED_NAMES+=("$name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# ==============================================================================
# 汇总
# ==============================================================================
echo ""
echo "================================================================"
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 消融实验全部完成！共 $TOTAL 组，全部成功"
else
    echo "⚠️  消融实验结束：$TOTAL 组中 $FAIL_COUNT 组失败"
    echo "失败实验:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  ✗ $name"
        echo "    日志: ${LOG_DIR}/${name}.log"
    done
fi
echo "================================================================"
echo "各实验检查点:"
IDX=0
for id in $RUN_LIST; do
    echo "  #$id  ./checkpoints/${EXP_NAMES[$IDX]}/"
    IDX=$((IDX + 1))
done
echo ""
echo "TensorBoard 对比可视化:"
echo "  tensorboard --logdir=./checkpoints/ --port=6006"
echo "================================================================"

[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
