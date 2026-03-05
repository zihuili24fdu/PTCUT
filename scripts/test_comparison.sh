#!/bin/bash

# ==============================================================================
# 虚拟染色对比实验测试脚本
# 对 train_comparison.sh 训练的 5 个模型逐一测试评估
# ==============================================================================
#
# 模型列表:
#   1: pix2pix         —— 经典有监督 GAN
#   2: cyclegan        —— 无监督循环一致性 GAN
#   3: cut             —— 对比无配对翻译
#   4: pyramid_pix2pix —— 带多尺度金字塔 L1 + Attention U-Net
#   5: ptcut           —— 本文方法（全损失配置）
#   6: pt_cyclegan     —— CycleGAN + CONCH 语义监督
#   7: pt_pix2pix      —— Pix2Pix + CONCH 语义监督
#
# 用法:
#   bash test_comparison.sh [GPU_IDS] [MODEL_IDS] [EPOCH]
#
# 参数说明:
#   GPU_IDS    - GPU 编号，逗号分隔（默认 "0"）
#   MODEL_IDS  - 模型编号，逗号分隔或 "all"（默认 "all"）
#   EPOCH      - 指定测试的 epoch（默认 "latest"）
#
# 示例:
#   bash test_comparison.sh 0 all             # 单卡测试全部 7 个模型
#   bash test_comparison.sh 0,1,2,3 all       # 4 卡并行测试
#   bash test_comparison.sh 0 4,5             # 单卡测试 pyramid + ptcut
#   bash test_comparison.sh 0 6,7             # 单卡测试 pt_cyclegan + pt_pix2pix
#   bash test_comparison.sh 0 all 50          # 测试第 50 个 epoch
#
# 日志目录: ./logs/comparison_test/<实验名>.log
# 结果汇总: ./logs/comparison_test/summary.txt
# ==============================================================================

GPU_IDS=${1:-0}
MODEL_IDS=${2:-"all"}
EPOCH=${3:-latest}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 公共路径配置
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_224_all"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# ==============================================================================
# 测试参数配置（与 train_comparison.sh 保持一致）
# ==============================================================================
PHASE="test"
PREPROCESS="none"
LOAD_SIZE=224
CROP_SIZE=224
NUM_TEST=2000
SUFFIX="all_dataset2"

# ✨ 功能开关（留空表示关闭）
# SAVE_IMAGES="--save_images"
CALC_METRICS="--calc_metrics"   # PSNR / SSIM / Pearson
CALC_LPIPS="--calc_lpips"       # LPIPS（感知相似性，越低越好）
LPIPS_NET="vgg"                 # LPIPS 骨干网络: vgg（语义感知更强）或 alex（更快）
#CALC_FID="--calc_fid"           # FID
#CALC_KID="--calc_kid"          # KID
#CALC_CROP_FID="--calc_crop_fid" # Crop-FID

CROP_FID_SIZE=128
CROP_FID_NUM=8

# PTCUT 专属参数（模型 #5）
LAMBDA_CLS=0.1
LAMBDA_DISTILL=10

# PT-CycleGAN / PT-Pix2Pix 专属参数（模型 #6 #7）
LAMBDA_CLS_PT=0.1
LAMBDA_DISTILL_PT=10

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
# 构造实验名称
# ==============================================================================
SUFF="${SUFFIX:+_${SUFFIX}}"

declare -A MODEL_NAME
declare -A MODEL_MODEL
declare -A MODEL_DESC

MODEL_NAME[1]="gnb_comp_pix2pix_${CROP_SIZE}${SUFF}"
MODEL_NAME[2]="gnb_comp_cyclegan_${CROP_SIZE}${SUFF}"
MODEL_NAME[3]="gnb_comp_cut_${CROP_SIZE}${SUFF}"
MODEL_NAME[4]="gnb_comp_pyramidp2p_${CROP_SIZE}${SUFF}"
MODEL_NAME[5]="gnb_comp_ptcut_cls${LAMBDA_CLS}_distill${LAMBDA_DISTILL}_${CROP_SIZE}${SUFF}"
MODEL_NAME[6]="gnb_comp_pt_cyclegan_cls${LAMBDA_CLS_PT}_distill${LAMBDA_DISTILL_PT}_${CROP_SIZE}${SUFF}"
MODEL_NAME[7]="gnb_comp_pt_pix2pix_cls${LAMBDA_CLS_PT}_distill${LAMBDA_DISTILL_PT}_${CROP_SIZE}${SUFF}"

MODEL_MODEL[1]="pix2pix"
MODEL_MODEL[2]="cycle_gan"
MODEL_MODEL[3]="cut"
MODEL_MODEL[4]="pyramid_pix2pix"
MODEL_MODEL[5]="ptcut"
MODEL_MODEL[6]="pt_cyclegan"
MODEL_MODEL[7]="pt_pix2pix"

MODEL_DESC[1]="Pix2Pix"
MODEL_DESC[2]="CycleGAN"
MODEL_DESC[3]="CUT"
MODEL_DESC[4]="PyramidPix2pix"
MODEL_DESC[5]="PTCUT(ours)"
MODEL_DESC[6]="PT-CycleGAN"
MODEL_DESC[7]="PT-Pix2Pix"

# ==============================================================================
# 创建日志目录
# ==============================================================================
LOG_DIR="./logs/comparison_test"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="${LOG_DIR}/summary.txt"

# ==============================================================================
# 检查公共依赖
# ==============================================================================
echo "================================================================"
echo "🔍 检查公共依赖文件..."
echo "================================================================"
if [ ! -f "$CONCH_CHECKPOINT" ];    then echo "❌ CONCH 权重丢失: $CONCH_CHECKPOINT";    exit 1; fi
if [ ! -f "$PROMPT_FEATURES_PATH" ]; then echo "❌ Prompt 特征丢失: $PROMPT_FEATURES_PATH"; exit 1; fi
echo "  ✓ CONCH 预训练权重存在"
echo "  ✓ KgCoOp Prompt 文本特征存在"

# 预检各模型 checkpoint
echo ""
echo "🔍 检查各模型 checkpoint..."
MISSING=0
for id in $RUN_LIST; do
    # CycleGAN / PT-CycleGAN 使用 G_A/G_B 双生成器，权重文件名不同
    if [ "$id" -eq 2 ] || [ "$id" -eq 6 ]; then
        ckpt="./checkpoints/${MODEL_NAME[$id]}/latest_net_G_A.pth"
    else
        ckpt="./checkpoints/${MODEL_NAME[$id]}/latest_net_G.pth"
    fi
    if [ ! -f "$ckpt" ]; then
        echo "  ❌ [模型 $id] 找不到权重: $ckpt"
        MISSING=$((MISSING + 1))
    else
        echo "  ✓ [模型 $id] ${MODEL_NAME[$id]}"
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "❌ 有 $MISSING 个模型缺少 checkpoint，请先完成训练后再测试。"
    exit 1
fi

# ==============================================================================
# 测试计划预览
# ==============================================================================
echo ""
echo "================================================================"
echo "📋 对比实验测试计划"
echo "================================================================"
echo "可用 GPU:      ${GPU_ARRAY[*]}  (共 $NUM_GPUS 张)"
echo "运行模型:      $RUN_LIST  (共 $TOTAL 个)"
echo "测试数据集:    $PHASE | 预处理: $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE)"
echo "日志目录:      $LOG_DIR"
echo "----------------------------------------------------------------"
IDX=0
for id in $RUN_LIST; do
    assigned_gpu=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    printf " #%-2s | GPU %-2s | %-16s | %s\n" \
        "$id" "$assigned_gpu" "${MODEL_DESC[$id]}" "${MODEL_NAME[$id]}"
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
    echo "⚠️  收到终止信号，正在停止所有测试进程..."
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
# 初始化汇总文件
# ==============================================================================
{
    echo "================================================================"
    echo "对比实验测试汇总"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "测试集: $PHASE | $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE)"
    echo "================================================================"
} > "$SUMMARY_FILE"

# ==============================================================================
# 并行启动所有测试
# ==============================================================================
echo "🚀 并行启动全部 $TOTAL 个测试..."
echo "================================================================"

declare -a EXP_NAMES
declare -a EXP_GPUS
declare -a EXP_LOGS

IDX=0
for id in $RUN_LIST; do
    NAME="${MODEL_NAME[$id]}"
    MODEL="${MODEL_MODEL[$id]}"
    ASSIGNED_GPU=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    LOG_FILE="${LOG_DIR}/${NAME}.log"

    echo "  ▶ 模型 #$id → GPU $ASSIGNED_GPU | ${MODEL_DESC[$id]}"
    echo "    日志: $LOG_FILE"

    {
        echo "============================================================"
        echo "测试模型 #$id: ${MODEL_DESC[$id]}"
        echo "实验名: $NAME"
        echo "GPU: $ASSIGNED_GPU"
        echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } > "$LOG_FILE"

    # ---- 构建 CONCH 相关参数（模型 #5 #6 #7 需要）----
    PTCUT_ARGS=""
    if [ "$id" -eq 5 ] || [ "$id" -eq 6 ] || [ "$id" -eq 7 ]; then
        PTCUT_ARGS="--conch_checkpoint $CONCH_CHECKPOINT --prompt_features_path $PROMPT_FEATURES_PATH --num_classes 2 --use_labels True"
    fi

    python test.py \
        --dataroot "$DATAROOT" \
        --name "$NAME" \
        --model "$MODEL" \
        --phase "$PHASE" \
        --gpu_ids "$ASSIGNED_GPU" \
        --preprocess "$PREPROCESS" \
        --load_size "$LOAD_SIZE" \
        --crop_size "$CROP_SIZE" \
        --num_test "$NUM_TEST" \
        --epoch "$EPOCH" \
        --eval \
        $PTCUT_ARGS \
        $SAVE_IMAGES \
        $CALC_METRICS \
        $CALC_LPIPS \
        ${CALC_LPIPS:+--lpips_net ${LPIPS_NET}} \
        $CALC_FID \
        $CALC_KID \
        $CALC_CROP_FID \
        ${CALC_CROP_FID:+--crop_fid_size $CROP_FID_SIZE} \
        ${CALC_CROP_FID:+--crop_fid_num $CROP_FID_NUM} \
        >> "$LOG_FILE" 2>&1 &

    PIDS[$IDX]=$!
    EXP_NAMES[$IDX]="$NAME"
    EXP_GPUS[$IDX]="$ASSIGNED_GPU"
    EXP_LOGS[$IDX]="$LOG_FILE"

    IDX=$((IDX + 1))
done

echo ""
echo "✅ 全部 $TOTAL 个测试已在后台启动..."
echo "   实时查看: tail -f ${LOG_DIR}/<实验名>.log"
echo "================================================================"

# ==============================================================================
# 等待所有进程结束，收集结果
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

        RESULT_JSON="./results/${name}/eval_stats_${PHASE}_latest.json"
        {
            printf "\n模型: %s\n" "$name"
            if [ -f "$RESULT_JSON" ]; then
                python -c "
import json
with open('$RESULT_JSON') as f:
    d = json.load(f)
def fmt(v, decimals=4):
    return f'{v:.{decimals}f}' if isinstance(v, (int, float)) else str(v)
if 'PSNR' in d:
    print(f\"  PSNR    : {fmt(d['PSNR'].get('mean','N/A'))}  (std={fmt(d['PSNR'].get('std','N/A'))})\")
if 'SSIM' in d:
    print(f\"  SSIM    : {fmt(d['SSIM'].get('mean','N/A'))}  (std={fmt(d['SSIM'].get('std','N/A'))})\")
if 'PEARSON' in d:
    print(f\"  PEARSON : {fmt(d['PEARSON'].get('mean','N/A'))}  (std={fmt(d['PEARSON'].get('std','N/A'))})\")
if 'LPIPS' in d:
    print(f\"  LPIPS   : {fmt(d['LPIPS'].get('mean','N/A'))}  (std={fmt(d['LPIPS'].get('std','N/A'))}, net={d['LPIPS'].get('net','?')})\")
if 'FID' in d:
    print(f\"  FID     : {fmt(d['FID'])}\")
if 'KID' in d:
    print(f\"  KID     : {fmt(d['KID'].get('mean','N/A'), 6)}  (std={fmt(d['KID'].get('std','N/A'), 6)})\")
"
            else
                echo "  ⚠️  结果文件不存在: $RESULT_JSON"
            fi
        } >> "$SUMMARY_FILE"
    else
        echo "  ❌ 失败: $name  (GPU $gpu, exit code: $EXIT_CODE)"
        FAILED_NAMES+=("$name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        printf "\n模型: %s\n  ❌ 测试失败 (exit code: %d)\n" "$name" "$EXIT_CODE" >> "$SUMMARY_FILE"
    fi
done

# ==============================================================================
# 汇总输出
# ==============================================================================
{
    echo ""
    echo "================================================================"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "共 $TOTAL 个模型，成功 $((TOTAL - FAIL_COUNT)) 个，失败 $FAIL_COUNT 个"
    echo "================================================================"
} >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 对比实验测试全部完成！共 $TOTAL 个，全部成功"
else
    echo "⚠️  对比实验测试结束：$TOTAL 个中 $FAIL_COUNT 个失败"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  ✗ $name  →  日志: ${LOG_DIR}/${name}.log"
    done
fi
echo ""
echo "📊 指标汇总: $SUMMARY_FILE"
echo "================================================================"
cat "$SUMMARY_FILE"
echo "================================================================"

[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
