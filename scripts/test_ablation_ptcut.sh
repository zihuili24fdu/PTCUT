#!/bin/bash

# ==============================================================================
# PTCUT 消融实验测试脚本 - 对 ablation_ptcut.sh 训练的 8 组实验逐一测试评估
# ==============================================================================
#
# 用法:
#   bash test_ablation_ptcut.sh [GPU_IDS] [EXP_IDS]
#
# 参数说明:
#   GPU_IDS  - GPU 编号，逗号分隔，支持多卡轮询分配（默认 "0"）
#              例: "0,1,2,3" 表示 4 张卡，实验按顺序轮询使用
#   EXP_IDS  - 要测试的实验编号，逗号分隔或 "all"（默认 "all"）
#              1: cls=0,    cls_d=0, distill=0   (纯 CUT 基线)
#              2: cls=0.1,  cls_d=0, distill=0   (+生成器分类)
#              3: cls=0,    cls_d=1, distill=0   (+判别器分类)
#              4: cls=0,    cls_d=0, distill=10  (+知识蒸馏)
#              5: cls=0.1,  cls_d=1, distill=0   (+生成器+判别器分类)
#              6: cls=0.1,  cls_d=0, distill=10  (+生成器分类+知识蒸馏)
#              7: cls=0,    cls_d=1, distill=10  (+判别器分类+知识蒸馏)
#              8: cls=0.1,  cls_d=1, distill=10  (全损失完整模型)
#
# 示例:
#   bash test_ablation_ptcut.sh 0 all         # 单卡依次测试全部 8 组
#   bash test_ablation_ptcut.sh 0,1,2,3 all   # 4 卡并行测试 8 组
#   bash test_ablation_ptcut.sh 0 1,4,8       # 单卡测试指定 3 组
#
# 日志文件: ./logs/ablation_test/<实验名>.log
# 结果汇总: ./logs/ablation_test/summary.txt
# ==============================================================================

GPU_IDS=${1:-0}
EXP_IDS=${2:-"all"}

cd /home/lzh/myCode/PTCUT

# ==============================================================================
# 公共路径配置
# ==============================================================================
DATAROOT="/home/lzh/myCode/virtual_stain_dataset/GNB_registered/patches_448"
CONCH_CHECKPOINT="/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
PROMPT_FEATURES_PATH="/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth"

# ==============================================================================
# 测试参数配置
# ==============================================================================
PHASE="test"
PREPROCESS="none"
LOAD_SIZE=1024
CROP_SIZE=448
NUM_TEST=-1

# ✨ 功能开关（留空表示关闭）
# SAVE_IMAGES="--save_images"
CALC_METRICS="--calc_metrics"
CALC_FID="--calc_fid"

# ==============================================================================
# 与 ablation_ptcut.sh 对应的损失激活值
# ==============================================================================
CLS_ON=0.1
CLS_D_ON=1
DISTILL_ON=10
CROP_SIZE_STR=448

# ==============================================================================
# 8 组消融实验配置: "CLS_ENABLE CLS_D_ENABLE DISTILL_ENABLE"
# ==============================================================================
declare -A EXP_FLAG
EXP_FLAG[1]="0 0 0"
EXP_FLAG[2]="1 0 0"
EXP_FLAG[3]="0 1 0"
EXP_FLAG[4]="0 0 1"
EXP_FLAG[5]="1 1 0"
EXP_FLAG[6]="1 0 1"
EXP_FLAG[7]="0 1 1"
EXP_FLAG[8]="1 1 1"

# ==============================================================================
# 解析 GPU 列表和实验 ID 列表
# ==============================================================================
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

if [ "$EXP_IDS" = "all" ]; then
    RUN_LIST="1 2 3 4 5 6 7 8"
else
    RUN_LIST=$(echo "$EXP_IDS" | tr ',' ' ')
fi

for id in $RUN_LIST; do
    if ! [[ "$id" =~ ^[1-8]$ ]]; then
        echo "❌ 无效的实验编号: $id（合法范围 1~8）"
        exit 1
    fi
done

TOTAL=$(echo $RUN_LIST | wc -w)

# ==============================================================================
# 创建日志目录
# ==============================================================================
LOG_DIR="./logs/ablation_test"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="${LOG_DIR}/summary.txt"

# ==============================================================================
# 检查公共依赖
# ==============================================================================
echo "================================================================"
echo "🔍 检查公共依赖文件..."
echo "================================================================"

if [ ! -f "$CONCH_CHECKPOINT" ];    then echo "❌ CONCH 权重丢失: $CONCH_CHECKPOINT";         exit 1; fi
if [ ! -f "$PROMPT_FEATURES_PATH" ]; then echo "❌ Prompt 特征丢失: $PROMPT_FEATURES_PATH"; exit 1; fi
echo "  ✓ CONCH 预训练权重存在"
echo "  ✓ KgCoOp Prompt 文本特征存在"

# 预先检查各实验的 checkpoint 是否存在
echo ""
echo "🔍 检查各实验 checkpoint..."
MISSING=0
for id in $RUN_LIST; do
    flags=(${EXP_FLAG[$id]})
    f_cls=${flags[0]}; f_clsd=${flags[1]}; f_distill=${flags[2]}
    lam_cls=$([ "$f_cls" -eq 1 ]       && echo "$CLS_ON"    || echo "0")
    lam_clsd=$([ "$f_clsd" -eq 1 ]     && echo "$CLS_D_ON"  || echo "0")
    lam_dist=$([ "$f_distill" -eq 1 ]  && echo "$DISTILL_ON" || echo "0")
    exp_name="gnb_ptcut_cls${lam_cls}_clsD${lam_clsd}_distill${lam_dist}_size${CROP_SIZE_STR}"
    ckpt="./checkpoints/${exp_name}/latest_net_G.pth"
    if [ ! -f "$ckpt" ]; then
        echo "  ❌ [实验 $id] 找不到权重文件: $ckpt"
        MISSING=$((MISSING + 1))
    else
        echo "  ✓ [实验 $id] $exp_name"
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "❌ 有 $MISSING 个实验缺少 checkpoint，请先完成训练后再测试。"
    exit 1
fi

# ==============================================================================
# 打印测试计划
# ==============================================================================
echo ""
echo "================================================================"
echo "📋 PTCUT 消融实验测试计划"
echo "================================================================"
echo "可用 GPU:       ${GPU_ARRAY[*]}  (共 $NUM_GPUS 张)"
echo "运行实验编号:   $RUN_LIST  (共 $TOTAL 组)"
echo "测试数据集:     $PHASE | 预处理: $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE)"
echo "日志目录:       $LOG_DIR"
echo "----------------------------------------------------------------"
echo " ID  | GPU | CLS    | CLS_D  | DISTILL | 实验名称"
echo "-----+-----+--------+--------+---------+----------------------------------"
IDX=0
for id in $RUN_LIST; do
    flags=(${EXP_FLAG[$id]})
    f_cls=${flags[0]}; f_clsd=${flags[1]}; f_distill=${flags[2]}
    lam_cls=$([ "$f_cls" -eq 1 ]       && echo "$CLS_ON"    || echo "0")
    lam_clsd=$([ "$f_clsd" -eq 1 ]     && echo "$CLS_D_ON"  || echo "0")
    lam_dist=$([ "$f_distill" -eq 1 ]  && echo "$DISTILL_ON" || echo "0")
    exp_name="gnb_ptcut_cls${lam_cls}_clsD${lam_clsd}_distill${lam_dist}_size${CROP_SIZE_STR}"
    assigned_gpu=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    printf " %-3s | %-3s | %-6s | %-6s | %-7s | %s\n" \
        "$id" "$assigned_gpu" "$lam_cls" "$lam_clsd" "$lam_dist" "$exp_name"
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
    echo "PTCUT 消融实验测试汇总"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "测试集: $PHASE | 预处理: $PREPROCESS (load=$LOAD_SIZE, crop=$CROP_SIZE)"
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
    flags=(${EXP_FLAG[$id]})
    f_cls=${flags[0]}; f_clsd=${flags[1]}; f_distill=${flags[2]}

    LAMBDA_CLS=$([ "$f_cls" -eq 1 ]        && echo "$CLS_ON"     || echo "0")
    LAMBDA_CLS_D=$([ "$f_clsd" -eq 1 ]     && echo "$CLS_D_ON"   || echo "0")
    LAMBDA_DISTILL=$([ "$f_distill" -eq 1 ] && echo "$DISTILL_ON" || echo "0")

    NAME="gnb_ptcut_cls${LAMBDA_CLS}_clsD${LAMBDA_CLS_D}_distill${LAMBDA_DISTILL}_size${CROP_SIZE_STR}"
    ASSIGNED_GPU=${GPU_ARRAY[$((IDX % NUM_GPUS))]}
    LOG_FILE="${LOG_DIR}/${NAME}.log"

    echo "  ▶ 实验 #$id → GPU $ASSIGNED_GPU | $NAME"
    echo "    日志: $LOG_FILE"

    {
        echo "============================================================"
        echo "测试实验 #$id: $NAME"
        echo "GPU: $ASSIGNED_GPU"
        echo "CLS=$LAMBDA_CLS  CLS_D=$LAMBDA_CLS_D  DISTILL=$LAMBDA_DISTILL"
        echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"
    } > "$LOG_FILE"

    python test.py \
        --dataroot "$DATAROOT" \
        --name "$NAME" \
        --model ptcut \
        --phase "$PHASE" \
        --gpu_ids "$ASSIGNED_GPU" \
        --preprocess "$PREPROCESS" \
        --load_size "$LOAD_SIZE" \
        --crop_size "$CROP_SIZE" \
        --num_test "$NUM_TEST" \
        --eval \
        --conch_checkpoint "$CONCH_CHECKPOINT" \
        --prompt_features_path "$PROMPT_FEATURES_PATH" \
        --num_classes 2 \
        --use_labels True \
        $SAVE_IMAGES \
        $CALC_METRICS \
        $CALC_FID \
        >> "$LOG_FILE" 2>&1 &

    PIDS[$IDX]=$!
    EXP_NAMES[$IDX]="$NAME"
    EXP_GPUS[$IDX]="$ASSIGNED_GPU"
    EXP_LOGS[$IDX]="$LOG_FILE"

    IDX=$((IDX + 1))
done

echo ""
echo "✅ 全部 $TOTAL 个测试已在后台启动，等待完成..."
echo "   实时查看日志: tail -f ${LOG_DIR}/<实验名>.log"
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

        # 从 eval_stats JSON 提取指标写入汇总
        RESULT_JSON="./results/${name}/eval_stats_${PHASE}_latest.json"
        {
            printf "\n实验: %s\n" "$name"
            if [ -f "$RESULT_JSON" ]; then
                python -c "
import json, sys
with open('$RESULT_JSON') as f:
    d = json.load(f)
print(f\"  PSNR   : {d.get('PSNR',{}).get('mean', 'N/A'):.4f}  (std={d.get('PSNR',{}).get('std', 'N/A'):.4f})\")
print(f\"  SSIM   : {d.get('SSIM',{}).get('mean', 'N/A'):.4f}  (std={d.get('SSIM',{}).get('std', 'N/A'):.4f})\")
print(f\"  PEARSON: {d.get('PEARSON',{}).get('mean', 'N/A'):.4f}  (std={d.get('PEARSON',{}).get('std', 'N/A'):.4f})\")
print(f\"  FID    : {d.get('FID', 'N/A'):.4f}\")
"
            else
                echo "  ⚠️  结果文件不存在: $RESULT_JSON"
            fi
        } >> "$SUMMARY_FILE"
    else
        echo "  ❌ 失败: $name  (GPU $gpu, exit code: $EXIT_CODE)"
        FAILED_NAMES+=("$name")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        printf "\n实验: %s\n  ❌ 测试失败 (exit code: %d)\n" "$name" "$EXIT_CODE" >> "$SUMMARY_FILE"
    fi
done

# ==============================================================================
# 汇总输出
# ==============================================================================
{
    echo ""
    echo "================================================================"
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "共 $TOTAL 组，成功 $((TOTAL - FAIL_COUNT)) 组，失败 $FAIL_COUNT 组"
    echo "================================================================"
} >> "$SUMMARY_FILE"

echo ""
echo "================================================================"
if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 消融实验测试全部完成！共 $TOTAL 组，全部成功"
else
    echo "⚠️  消融实验测试结束：$TOTAL 组中 $FAIL_COUNT 组失败"
    echo "失败实验:"
    for name in "${FAILED_NAMES[@]}"; do
        echo "  ✗ $name"
        echo "    日志: ${LOG_DIR}/${name}.log"
    done
fi
echo ""
echo "📊 指标汇总: $SUMMARY_FILE"
echo "================================================================"
cat "$SUMMARY_FILE"
echo "================================================================"

[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
