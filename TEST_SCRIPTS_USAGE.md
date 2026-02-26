# 测试脚本使用说明

## 概述

所有测试脚本已更新，现在支持两种运行模式：

1. **完整测试模式 (test)** - 默认模式：运行测试生成图像，然后评估
2. **仅评估模式 (eval-only)**：使用已有的测试结果图像进行评估，不重新生成图像

## 脚本列表

- `test_cut_gnb.sh` - CUT模型测试
- `test_cyclegan_gnb.sh` - CycleGAN模型测试
- `test_pix2pix_gnb.sh` - Pix2Pix模型测试
- `test_ptcut_gnb.sh` - PTCUT模型测试
- `test_ptcut_no_lcls_gnb.sh` - PTCUT (无分类损失) 模型测试

## 使用方法

### 1. 完整测试模式（默认）

运行测试生成图像，然后自动评估：

```bash
# 不带参数，默认为test模式
bash test_cut_gnb.sh

# 或者显式指定test模式
bash test_cut_gnb.sh test
```

这会：
- ✓ 检查数据集和模型checkpoint
- ✓ 运行 `test.py` 生成测试图像
- ✓ 运行 `fast_evaluate.py` 评估结果

### 2. 仅评估模式

仅对已有的测试结果进行评估，不重新生成图像：

```bash
bash test_cut_gnb.sh eval-only
```

这会：
- ✓ 检查结果目录是否存在
- ✓ 验证是否有已生成的图像
- ✓ 运行 `fast_evaluate.py` 评估结果
- ✗ 不运行 `test.py`，节省时间

### 3. 应用场景

#### 使用完整测试模式 (test)：
- 首次运行测试
- 模型checkpoint更新后
- 想要重新生成测试图像
- 测试参数（如epoch）发生变化

#### 使用仅评估模式 (eval-only)：
- 已经运行过测试，有现成的测试图像
- 只想重新计算评估指标
- 调试评估脚本时
- 节省GPU计算时间

## 示例

### 示例 1: 首次测试CUT模型
```bash
cd /home/lzh/myCode/PTCUT
bash test_cut_gnb.sh
```

### 示例 2: 重新评估PTCUT模型的已有结果
```bash
cd /home/lzh/myCode/PTCUT
bash test_ptcut_gnb.sh eval-only
```

### 示例 3: 批量重新评估所有模型
```bash
cd /home/lzh/myCode/PTCUT
bash test_cut_gnb.sh eval-only
bash test_cyclegan_gnb.sh eval-only
bash test_pix2pix_gnb.sh eval-only
bash test_ptcut_gnb.sh eval-only
bash test_ptcut_no_lcls_gnb.sh eval-only
```

## 结果目录结构

测试和评估的结果保存在：
```
./results/$NAME/test_latest/
├── images/              # 生成的测试图像
│   ├── *_real_A.png    # 输入图像
│   ├── *_fake_B.png    # 生成图像
│   └── *_real_B.png    # 真实目标图像
└── evaluation/          # 评估结果
    └── statistics.json  # 评估指标
```

## 错误处理

### 错误 1: 无效的模式参数
```
❌ 错误: 无效的模式 'xxx'
有效模式: test, eval-only
```
**解决方案**: 使用 `test` 或 `eval-only` 作为参数

### 错误 2: eval-only模式下结果目录不存在
```
❌ 错误: 结果目录不存在: ./results/xxx/test_latest/images
请先运行测试生成图像，或使用 'test' 模式
```
**解决方案**: 先使用test模式运行一次完整测试

### 错误 3: test模式下checkpoint不存在
```
❌ 错误: Checkpoint文件不存在: ./checkpoints/xxx/latest_net_G.pth
```
**解决方案**: 先运行对应的训练脚本

## 注意事项

1. **eval-only模式要求**：必须已经运行过至少一次完整测试，且结果目录存在
2. **EPOCH参数**：脚本中的EPOCH变量可以修改为特定epoch（如"100", "200"）
3. **GPU设置**：不同脚本使用的GPU可能不同，请根据需要调整GPU_IDS参数
4. **结果覆盖**：test模式会覆盖已有的测试图像

## 修改历史

- 2026-01-25: 添加 eval-only 模式支持，允许跳过测试直接评估已有结果
