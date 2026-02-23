# PTCUT训练脚本说明

## 脚本版本对比

| 特性 | `train_ptcut_gnb.sh` (旧版) | `train_ptcut_gnb_v2.sh` (新版) |
|------|---------------------------|----------------------------|
| **文本特征来源** | 预保存的`.pth`文件 | KgCoOp模型权重动态生成 |
| **参数** | `--prompt_text_features` | `--kgcoop_model_dir`<br>`--kgcoop_epoch`<br>`--kgcoop_seed` |
| **可靠性** | ⚠️ 可能有类别顺序错误 | ✅ 与训练时完全一致 |
| **灵活性** | ❌ 固定特征文件 | ✅ 可切换不同epoch |
| **实验名称** | `gnb_ptcut` | `gnb_ptcut_v2` |

---

## 新版本使用方法

### 1. 快速开始

```bash
cd /home/lzh/myCode/PTCUT
bash train_ptcut_gnb_v2.sh
```

### 2. 检查数据集

脚本会自动检查：
- ✅ 数据集目录是否存在
- ✅ trainA, trainB, testA, testB 子目录
- ✅ KgCoOp模型checkpoint是否存在

### 3. 监控训练

**查看损失**：
```bash
tail -f checkpoints/gnb_ptcut_v2/loss_log.txt
```

**TensorBoard**：
```bash
tensorboard --logdir=checkpoints/gnb_ptcut_v2
```

### 4. 重要参数说明

#### KgCoOp模型参数
```bash
KGCOOP_MODEL_DIR="...output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite"
KGCOOP_EPOCH=100        # 使用epoch 100的模型
KGCOOP_SEED=1           # 保持与训练时一致
```

#### 损失权重
```bash
LAMBDA_CLS=1.0         # 分类损失 - 引导生成图像包含正确类别特征
LAMBDA_DISTILL=1.0     # 蒸馏损失 - 确保生成图像保留语义信息
LAMBDA_GAN=1.0         # GAN损失 - 生成真实感图像
LAMBDA_NCE=1.0         # NCE损失 - 保持内容结构
```

#### 图像尺寸
```bash
LOAD_SIZE=512          # 加载时resize到512
CROP_SIZE=512          # 随机裁剪512×512
BATCH_SIZE=2           # 批大小（1024×1024图像内存占用大）
```

---

## 调整建议

### 如果显存不足
```bash
BATCH_SIZE=1           # 减小batch size
LOAD_SIZE=256          # 减小图像尺寸
CROP_SIZE=256
```

### 如果想快速验证
```bash
N_EPOCHS=5             # 减少epoch数
SAVE_EPOCH_FREQ=1      # 每个epoch保存
```

### 如果想提高质量
```bash
N_EPOCHS=50            # 增加训练时间
LAMBDA_CLS=2.0         # 加强分类监督
LAMBDA_DISTILL=2.0     # 加强语义保持
```

---

## 训练输出

### 保存位置
```
checkpoints/gnb_ptcut_v2/
├── latest_net_G.pth          # 最新的生成器
├── latest_net_D.pth          # 最新的判别器
├── 5_net_G.pth               # epoch 5的生成器
├── 5_net_D.pth               # epoch 5的判别器
├── loss_log.txt              # 损失日志
└── web/                      # 可视化结果
    └── images/
```

### 查看生成结果
```bash
# 方法1: 浏览器打开
firefox checkpoints/gnb_ptcut_v2/web/index.html

# 方法2: 查看最新图像
ls -lt checkpoints/gnb_ptcut_v2/web/images/ | head -20
```

---

## 故障排除

### 问题1: CUDA out of memory
**解决**: 减小 `BATCH_SIZE` 或 `CROP_SIZE`

### 问题2: KgCoOp模型加载失败
**解决**: 检查路径和epoch是否正确
```bash
ls -l /home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_learner/model.pth.tar-100
```

### 问题3: 数据集找不到
**解决**: 确认数据集路径和文件名格式（应该是 `*_i.jpg` 或 `*_n.jpg`）

---

## 下一步

训练完成后运行测试：
```bash
bash test_ptcut_gnb_v2.sh
```

**推荐工作流程**：
1. 先用小参数快速验证（5 epochs，batch=1）
2. 检查生成质量和损失趋势
3. 调整参数后完整训练
4. 测试并评估结果
