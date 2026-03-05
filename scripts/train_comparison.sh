#!/bin/bash

# ==============================================================================
# 虚拟染色对比实验训练脚本
# 包含 5 种模型的完整对比组，全部并行启动
# ==============================================================================
#
# 模型列表:
#   1: pix2pix         —— 经典有监督 GAN（ResNet 生成器 + PatchGAN 判别器）
#   2: cyclegan        —— 无监督循环一致性 GAN（双向翻译）
#   3: cut             —— 对比无配对翻译 (Contrastive Unpaired Translation)
#   4: pyramid_pix2pix —— 带多尺度金字塔 L1 + Attention U-Net（BCI 方法）
#   5: ptcut           —— 本文方法：CONCH 语义监督 + 知识蒸馏 CUT（全损失配置）
#   6: pt_cyclegan     —— CycleGAN + CONCH 语义监督（分类损失 + 蒸馏损失）
#   7: pt_pix2pix      —— Pix2Pix + CONCH 语义监督（分类损失 + 蒸馏损失）
#
# 用法:
#   bash train_comparison.sh [GPU_IDS] [MODEL_IDS] [CONTINUE_EPOCH]
#
# 参数说明:
#   GPU_IDS        - GPU 编号，逗号分隔，支持多卡轮询（默认 "0"）
#                    例: "0,1,2,3" 表示 4 张卡，5 个实验按顺序轮询
#   MODEL_IDS      - 要运行的模型编号，逗号分隔或 "all"（默认 "all"）
#                    例: "1,4,5" 表示只运行 pix2pix、pyramid_pix2pix、ptcut
#   CONTINUE_EPOCH - 从指定 epoch 继续训练（留空则从头开始）
#
# 示例:
#   bash train_comparison.sh 0 all             # 单卡串行训练全部 7 个模型
#   bash train_comparison.sh 0,1,2,3 all       # 4 卡并行训练 7 个模型
#   bash train_comparison.sh 0,1 4,5           # 双卡并行训练 pyramid + ptcut
#   bash train_comparison.sh 0,1 6,7           # 双卡并行训练 pt_cyclegan + pt_pix2pix
#   bash train_comparison.sh 0 all 30          # 单卡从第 30 epoch 继续训练所有模型
#
# 日志目录: ./logs/comparison/<实验名>.log
# ==============================================================================

GPU_IDS=${1:-0}
MODEL_IDS=${2:-"all"}
CONTINUE_EPOCH=${3:-""}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 公共路径配置
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# ==============================================================================
# 公共超参数（所有模型共用同一数据和基础设置）
# ==============================================================================
LOAD_SIZE=224
CROP_SIZE=224
PREPROCESS="crop"
BATCH_SIZE=2
LR=0.0002

# ==============================================================================
# 各模型独立 Epoch 配置（在此处单独调整每个模型的训练轮数）
# 格式: M_EPOCHS[<模型ID>]=<初始学习率阶段轮数>  M_EPOCHS_DECAY[<模型ID>]=<学习率衰减阶段轮数>
# 总训练轮数 = M_EPOCHS[id] + M_EPOCHS_DECAY[id]
# ==============================================================================
declare -A M_EPOCHS
declare -A M_EPOCHS_DECAY

M_EPOCHS[1]=80  ;  M_EPOCHS_DECAY[1]=20   # pix2pix
M_EPOCHS[2]=30  ;  M_EPOCHS_DECAY[2]=10   # cyclegan
M_EPOCHS[3]=30  ;  M_EPOCHS_DECAY[3]=10   # cut
M_EPOCHS[4]=80  ;  M_EPOCHS_DECAY[4]=20   # pyramid_pix2pix（有监督，需更多轮次）
M_EPOCHS[5]=30  ;  M_EPOCHS_DECAY[5]=10   # ptcut
M_EPOCHS[6]=30  ;  M_EPOCHS_DECAY[6]=10   # pt_cyclegan
M_EPOCHS[7]=80  ;  M_EPOCHS_DECAY[7]=20   # pt_pix2pix（有监督，需更多轮次）
NET_D="basic"
N_LAYERS_D=3
SUFFIX="all_dataset2"   # 实验名后缀，用于区分数据集版本；留空则不追加

# ==============================================================================
# PTCUT 专属损失权重（模型 #5，全损失配置）
# ==============================================================================
LAMBDA_CLS=0.1
LAMBDA_DISTILL=10
LAMBDA_GAN_PTCUT=1.0
LAMBDA_NCE=1.0
CLS_WARMUP_EPOCHS=10
CLS_RAMPUP_EPOCHS=5

# ==============================================================================
# PT-CycleGAN / PT-Pix2Pix 专属损失权重（模型 #6 #7，与 PTCUT 共享 CONCH 配置）
# ==============================================================================
LAMBDA_CLS_PT=0.1
LAMBDA_DISTILL_PT=10
CLS_WARMUP_EPOCHS_PT=10
CLS_RAMPUP_EPOCHS_PT=5

# ==============================================================================
# 解析 GPU 列表 & 模型 ID 列表
# ==============================================================================
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$MODEL_IDS" = "all" ]; then
    RUN_LIST="1 2 3 4 5 6 7"
else
    RUN_LIST=$(echo "$MODEL_IDS" | tr ',' ' ')
fi

for id in $RUN_LIST; do
    if ! [[ "$id" =~ ^[1-7]$ ]]; then
        echo "❌ 无效的模型编号: $id（合法范围 1~7）"
        exit 1
    fi
done

TOTAL=$(echo $RUN_LIST | wc -w)

# ==============================================================================
# 创建日志目录
# ==============================================================================
LOG_DIR="./logs/comparison"
mkdir -p "$LOG_DIR"

# ==============================================================================
# 检查公共依赖文件
# ==============================================================================
echo "================================================================"
echo "🔍 检查公共依赖文件..."
echo "================================================================"
if [ ! -d "$DATAROOT" ]; then
    echo "❌ 数据集目录不存在: $DATAROOT"
    exit 1
fi
if [ ! -f "$CONCH_CHECKPOINT" ]; then
    echo "❌ CONCH 权重文件丢失: $CONCH_CHECKPOINT"
    exit 1
fi
if [ ! -f "$PROMPT_FEATURES_PATH" ]; then
    echo "❌ Prompt 特征文件丢失: $PROMPT_FEATURES_PATH"
    exit 1
fi
echo "  ✓ 数据集目录存在"
echo "  ✓ CONCH 预训练权重存在"
echo "  ✓ KgCoOp Prompt 文本特征存在"

# ==============================================================================
# 构造各模型的名称与 GPU 分配（表格预览）
# ==============================================================================
declare -A MODEL_NAME
declare -A MODEL_DESC

SUFF="${SUFFIX:+_${SUFFIX}}"

MODEL_NAME[1]="gnb_comp_pix2pix_${CROP_SIZE}${SUFF}"
MODEL_NAME[2]="gnb_comp_cyclegan_${CROP_SIZE}${SUFF}"
MODEL_NAME[3]="gnb_comp_cut_${CROP_SIZE}${SUFF}"
MODEL_NAME[4]="gnb_comp_pyramidp2p_${CROP_SIZE}${SUFF}"
MODEL_NAME[5]="gnb_comp_ptcut_cls${LAMBDA_CLS}_distill${LAMBDA_DISTILL}_${CROP_SIZE}${SUFF}"
MODEL_NAME[6]="gnb_comp_pt_cyclegan_cls${LAMBDA_CLS_PT}_distill${LAMBDA_DISTILL_PT}_${CROP_SIZE}${SUFF}"
MODEL_NAME[7]="gnb_comp_pt_pix2pix_cls${LAMBDA_CLS_PT}_distill${LAMBDA_DISTILL_PT}_${CROP_SIZE}${SUFF}"

MODEL_DESC[1]="Pix2Pix      (ResNet-9 + L1 + vanillaGAN)"
MODEL_DESC[2]="CycleGAN     (ResNet-9 + 循环一致性)"
MODEL_DESC[3]="CUT          (ResNet-9 + 对比 NCE)"
MODEL_DESC[4]="PyramidP2P   (AttentionUNet + 4级金字塔L1)"
MODEL_DESC[5]="PTCUT(ours)  (ResNet-9 + CLS + DISTILL, 全损失)"
MODEL_DESC[6]="PT-CycleGAN  (ResNet-9 + 循环一致性 + CLS + DISTILL)"
MODEL_DESC[7]="PT-Pix2Pix   (ResNet-9 + L1 + vanillaGAN + CLS + DISTILL)"

echo ""
echo "================================================================"
echo "📋 对比实验训练计划"
echo "================================================================"
echo "可用 GPU:       ${GPU_ARRAY[*]}  (共 $NUM_GPUS 张)"
echo "运行模型编号:   $RUN_LIST  (共 $TOTAL 个)"
echo "实验后缀:       ${SUFFIX:-(无)}"
echo "图像尺寸:       Load $LOAD_SIZE -> Crop $CROP_SIZE ($PREPROCESS)"
echo "日志目录:       $LOG_DIR"
echo "----------------------------------------------------------------"
printf " %-3s | %-3s | %-8s | %-8s | %s\n" "ID" "GPU" "Epochs" "Decay" "描述"
echo "-----+-----+----------+----------+-----------------------------------"
IDX=0
for id in $RUN_LIST; do
    assigned_gpu=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    printf " %-3s | %-3s | %-8s | %-8s | %s\n      |     |          |          | → %s\n" \
        "$id" "$assigned_gpu" "${M_EPOCHS[$id]}" "${M_EPOCHS_DECAY[$id]}" \
        "${MODEL_DESC[$id]}" "${MODEL_NAME[$id]}"
    IDX=$((IDX + 1))
done
echo "================================================================"
echo ""

# ==============================================================================
# 清理函数
# ==============================================================================
declare -a PIDS

cleanup() {
    echo ""
    echo "⚠️  收到终止信号，正在停止所有训练进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  已终止 PID $pid"
        fi
    done
    wait 2>/dev/null
    echo "✅ 已终止所有进程，退出"
    exit 1
}
trap cleanup SIGHUP SIGINT SIGTERM

# ==============================================================================
# 并行启动所有训练进程
# ==============================================================================
echo "🚀 并行启动全部 $TOTAL 个训练进程..."
echo "================================================================"

declare -a EXP_NAMES
declare -a EXP_GPUS
declare -a EXP_LOGS

IDX=0
for id in $RUN_LIST; do
    NAME="${MODEL_NAME[$id]}"
    ASSIGNED_GPU=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    LOG_FILE="${LOG_DIR}/${NAME}.log"

    _EPOCHS=${M_EPOCHS[$id]}
    _DECAY=${M_EPOCHS_DECAY[$id]}

    echo "  ▶ 模型 #$id → GPU $ASSIGNED_GPU | ${MODEL_DESC[$id]}"
    echo "    实验名: $NAME"
    echo "    Epochs: ${_EPOCHS} + ${_DECAY} (decay) = $((${_EPOCHS} + ${_DECAY})) 总轮数"
    echo "    日志:   $LOG_FILE"

    {
        echo "============================================================"
        echo "对比实验模型 #$id: ${MODEL_DESC[$id]}"
        echo "实验名: $NAME"
        echo "GPU: $ASSIGNED_GPU"
        echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } > "$LOG_FILE"

    # -------------------------------------------------------
    # 构建可选的 continue_train 参数
    # -------------------------------------------------------
    if [ -n "$CONTINUE_EPOCH" ]; then
        CONTINUE_ARGS="--continue_train --epoch_count $((CONTINUE_EPOCH + 1)) --epoch $CONTINUE_EPOCH"
    else
        CONTINUE_ARGS=""
    fi

    # -------------------------------------------------------
    # 根据模型 ID 构建特定参数并启动
    # -------------------------------------------------------
    case $id in
    1)  # ── Pix2Pix ─────────────────────────────────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model pix2pix \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG batch \
            --normD batch \
            --dataset_mode aligned \
            --direction AtoB \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_L1 100.0 \
            --gan_mode vanilla \
            --pool_size 0 \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    2)  # ── CycleGAN ─────────────────────────────────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model cycle_gan \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG instance \
            --normD instance \
            --dataset_mode unaligned \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_A 10.0 \
            --lambda_B 10.0 \
            --lambda_identity 0.5 \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    3)  # ── CUT ─────────────────────────────────────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model cut \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG instance \
            --normD instance \
            --dataset_mode unaligned \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_GAN 1.0 \
            --lambda_NCE 1.0 \
            --nce_idt \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    4)  # ── PyramidPix2pix ───────────────────────────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model pyramid_pix2pix \
            --netG attention_unet_32 \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG batch \
            --normD batch \
            --dataset_mode aligned \
            --direction AtoB \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_L1 25.0 \
            --weight_L2 25.0 \
            --weight_L3 25.0 \
            --weight_L4 25.0 \
            --gan_mode lsgan \
            --pool_size 0 \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    5)  # ── PTCUT (ours, 全损失) ─────────────────────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model ptcut \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG instance \
            --normD instance \
            --dataset_mode unaligned \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_GAN "$LAMBDA_GAN_PTCUT" \
            --lambda_NCE "$LAMBDA_NCE" \
            --lambda_cls "$LAMBDA_CLS" \
            --lambda_distill "$LAMBDA_DISTILL" \
            --cls_warmup_epochs "$CLS_WARMUP_EPOCHS" \
            --cls_rampup_epochs "$CLS_RAMPUP_EPOCHS" \
            --nce_idt \
            --conch_checkpoint "$CONCH_CHECKPOINT" \
            --prompt_features_path "$PROMPT_FEATURES_PATH" \
            --num_classes 2 \
            --use_labels True \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    6)  # ── PT-CycleGAN (CycleGAN + CLS + DISTILL) ────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model pt_cyclegan \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG instance \
            --normD instance \ \            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_A 10.0 \
            --lambda_B 10.0 \
            --lambda_identity 0.5 \
            --lambda_cls "$LAMBDA_CLS_PT" \
            --lambda_distill "$LAMBDA_DISTILL_PT" \
            --cls_warmup_epochs "$CLS_WARMUP_EPOCHS_PT" \
            --cls_rampup_epochs "$CLS_RAMPUP_EPOCHS_PT" \
            --conch_checkpoint "$CONCH_CHECKPOINT" \
            --prompt_features_path "$PROMPT_FEATURES_PATH" \
            --num_classes 2 \
            --use_labels True \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    7)  # ── PT-Pix2Pix (Pix2Pix + CLS + DISTILL) ──────────────────
        python train.py \
            --dataroot "$DATAROOT" \
            --name "$NAME" \
            --model pt_pix2pix \
            --netG resnet_9blocks \
            --netD "$NET_D" \
            --n_layers_D $N_LAYERS_D \
            --normG batch \
            --normD batch \
            --dataset_mode ptcut \
            --direction AtoB \
            --gpu_ids "$ASSIGNED_GPU" \
            --preprocess "$PREPROCESS" \
            --load_size "$LOAD_SIZE" \
            --crop_size "$CROP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --n_epochs "$_EPOCHS" \
            --n_epochs_decay "$_DECAY" \
            --lr "$LR" \
            --lambda_L1 100.0 \
            --gan_mode vanilla \
            --pool_size 0 \
            --lambda_cls "$LAMBDA_CLS_PT" \
            --lambda_distill "$LAMBDA_DISTILL_PT" \
            --cls_warmup_epochs "$CLS_WARMUP_EPOCHS_PT" \
            --cls_rampup_epochs "$CLS_RAMPUP_EPOCHS_PT" \
            --conch_checkpoint "$CONCH_CHECKPOINT" \
            --prompt_features_path "$PROMPT_FEATURES_PATH" \
            --num_classes 2 \
            --use_labels True \
            --display_id -1 \
            --no_html \
            $CONTINUE_ARGS \
            >> "$LOG_FILE" 2>&1 &
        ;;
    esac

    PIDS[$IDX]=$!
    EXP_NAMES[$IDX]="$NAME"
    EXP_GPUS[$IDX]="$ASSIGNED_GPU"
    EXP_LOGS[$IDX]="$LOG_FILE"

    echo "    PID: ${PIDS[$IDX]}"
    echo ""

    IDX=$((IDX + 1))
done

echo "✅ 全部 $TOTAL 个训练进程已在后台启动，等待完成..."
echo "   实时查看日志: tail -f ${LOG_DIR}/<实验名>.log"
echo "================================================================"

# ==============================================================================
# 等待所有进程结束
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
    echo "🎉 对比实验训练全部完成！共 $TOTAL 个模型，全部成功"
else
    echo "⚠️  训练结束：$TOTAL 个模型中 $FAIL_COUNT 个失败"
    echo "失败模型:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  ✗ $name"
        echo "    日志: ${LOG_DIR}/${name}.log"
    done
fi
echo "================================================================"
[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
