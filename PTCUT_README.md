# PTCUT模型使用说明

## 概述

PTCUT (Prompt-Tuned CUT) 是一个基于CUT模型的虚拟染色模型，结合了训练好的提示调优特征进行增强。

## 模型架构

PTCUT模型在CUT的基础上添加了两个新的损失：

1. **分类损失 (Classification Loss)**
   - 将生成图像(fakeB)通过CONCH视觉编码器提取特征
   - 与预训练的提示文本特征计算相似度
   - 根据相似度计算分类损失，引导生成符合特定类别特征的图像

2. **蒸馏损失 (Distillation Loss)**
   - 将真实图像(realB)和生成图像(fakeB)都通过CONCH视觉编码器
   - 计算两者特征的余弦相似度
   - 确保生成图像保留语义信息

## 前置要求

### 1. 训练KgCoOp模型

首先需要训练KgCo模型以获得类别提示特征：

```bash
cd /home/lzh/myCode/myKgCoOp/myKgCoOp

# 训练KgCoOp模型（4类GNB数据集）
python train.py \
    --root data \
    --trainer KgCoOpConchCSC \
    --dataset-config-file configs/datasets/gnb4class.yaml \
    --config-file configs/trainers/kgcoop_conch_csc.yaml \
    --output-dir output/gnb_kgcoop_4class \
    --gpu-ids 0
```

训练完成后，需要提取text_features并保存为.pt文件：
```python
import torch

# 加载KgCoOp checkpoint
checkpoint = torch.load('output/gnb_kgcoop_4class/prompt_learner/model-best.pth.tar', 
                       map_location='cpu', weights_only=False)

# 提取text_features
text_features = checkpoint['state_dict']['text_features']
print(f'提取的特征形状: {text_features.shape}')  # 应该是 (4, 512)

# 保存为.pt文件
torch.save(text_features, 'output/gnb_kgcoop_4class/prompt_text_features.pt')
print('✓ 已保存到 prompt_text_features.pt')
```

### 2. 准备虚拟染色数据集

数据集应采用CUT模型的标准格式：
```
dataroot/
├── trainA/  # 源域图像（如H&E染色）
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
└── trainB/  # 目标域图像（如IHC染色）
    ├── image_1.png
    ├── image_2.png
    └── ...
```

**重要**: 如果使用分类损失（`--use_labels True`），图像文件名的最后一位应该是类别号（1-4）：
```
trainB/
├── sample_001_1.png  # 类别1
├── sample_002_2.png  # 类别2
├── sample_003_3.png  # 类别3
└── sample_004_4.png  # 类别4
```

## 训练PTCUT模型

### 基本训练命令

```bash
cd /home/lzh/myCode/awesome-virtual-staining

python train_ptcut.py \
    --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512 \
    --name gnb_ptcut_experiment1 \
    --model ptcut \
    --prompt_text_features /home/lzh/myCode/myKgCoOp/myKgCoOp/output/gnb_kgcoop_4class/prompt_text_features.pt \
    --num_classes 4 \
    --load_size 512 \
    --crop_size 512 \
    --gpu_ids 0 \
    --batch_size 2 \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --lambda_GAN 1.0 \
    --lambda_NCE 1.0 \
    --lambda_cls 0.5 \
    --lambda_distill 0.5 \
    --use_labels True
```

### 参数说明

#### PTCUT特有参数

- `--prompt_text_features`: 预编码的文本特征文件路径（.pt格式，必需）
- `--num_classes`: 类别数量（默认：4）
- `--lambda_cls`: 分类损失权重（默认：0.5）
- `--lambda_distill`: 蒸馏损失权重（默认：0.5）
- `--use_labels`: 是否使用图像标签进行监督（默认：True）

#### CUT模型参数

- `--lambda_GAN`: GAN损失权重（默认：1.0）
- `--lambda_NCE`: NCE对比损失权重（默认：1.0）
- `--nce_idt`: 是否使用identity NCE（CUT模式默认：True）
- `--CUT_mode`: CUT或FastCUT（默认：CUT）

#### 通用参数

- `--dataroot`: 数据集根目录
- `--name`: 实验名称
- `--gpu_ids`: 使用的GPU ID（多GPU用逗号分隔）
- `--batch_size`: 批次大小
- `--load_size`: 加载图像大小
- `--crop_size`: 裁剪大小
- `--n_epochs`: 固定学习率的epoch数
- `--n_epochs_decay`: 学习率衰减的epoch数

## 训练监控

### 查看训练损失

训练过程中会打印以下损失：
- `G_GAN`: 生成器GAN损失
- `D_real`: 判别器真实样本损失
- `D_fake`: 判别器生成样本损失
- `NCE`: 对比学习损失
- `CLS`: 分类损失（PTCUT新增）
- `DISTILL`: 蒸馏损失（PTCUT新增）

### 查看生成结果

结果保存在 `checkpoints/{name}/web/` 目录：
```bash
# 启动本地服务器查看结果
cd checkpoints/gnb_ptcut_experiment1/web
python -m http.server 8000
# 浏览器访问 http://localhost:8000
```

## 测试模型

```bash
python test.py \
    --dataroot /path/to/test/data \
    --name gnb_ptcut_experiment1 \
    --model ptcut \
    --prompt_text_features /path/to/prompt_text_features.pt \
    --num_classes 4 \
    --phase test \
    --no_dropout \
    --load_size 512 \
    --crop_size 512
```

## 超参数调优建议

### 损失权重平衡

1. **初始配置**（推荐）：
   ```
   --lambda_GAN 1.0 --lambda_NCE 1.0 --lambda_cls 0.5 --lambda_distill 0.5
   ```

2. **强调语义保持**：
   ```
   --lambda_GAN 1.0 --lambda_NCE 1.0 --lambda_cls 0.3 --lambda_distill 1.0
   ```

3. **强调类别区分**：
   ```
   --lambda_GAN 1.0 --lambda_NCE 1.0 --lambda_cls 1.0 --lambda_distill 0.3
   ```

### 训练策略

1. **从零开始**：使用上述基本训练命令
2. **从CUT预训练**：先训练标准CUT模型，然后用PTCUT继续训练
3. **消融实验**：分别设置 `--lambda_cls 0` 或 `--lambda_distill 0` 来评估各损失的贡献

## 常见问题

### Q: 如何从KgCoOp checkpoint提取text_features？

A: 使用以下Python代码：
```python
import torch
checkpoint = torch.load('path/to/checkpoint.pth.tar', weights_only=False)
text_features = checkpoint['state_dict']['text_features']
torch.save(text_features, 'prompt_text_features.pt')
```

### Q: text_features文件不存在怎么办？

A: 确保使用更新后的KgCoOp trainer重新训练，或运行 `after_train()` 方法提取特征。

### Q: 分类损失总是很高？

A: 检查：
1. 图像文件名是否正确标注类别
2. `--num_classes` 设置是否正确
3. 降低 `--lambda_cls` 权重
4. 确认prompt checkpoint与数据集匹配

### Q: 生成图像质量不理想？

A: 尝试：
1. 增加训练epochs
2. 调整loss权重平衡
3. 先用CUT预训练再用PTCUT微调
4. 检查数据集质量和配对

## 引用

如果使用PTCUT模型，请引用CUT和KgCoOp相关论文：

```bibtex
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Park, Taesung and Efros, Alexei A and Zhang, Richard and Zhu, Jun-Yan},
  booktitle={ECCV},
  year={2020}
}
```
