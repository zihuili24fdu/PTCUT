"""PTCUT Model (Prompt-Tuned CUT)

基于CUT模型的虚拟染色模型,结合KgCoOp训练好的提示特征和CONCH视觉编码器

============================================================================
核心思想：使用CONCH作为"语义监督器"
============================================================================

传统CUT模型的问题：
- 仅使用GAN损失和NCE对比损失
- 缺乏明确的语义（分类）监督
- 生成图像可能在视觉上相似,但语义信息丢失

PTCUT的解决方案：
添加两个基于CONCH的语义损失：

1. 分类损失 (Classification Loss):
   - 使用CONCH视觉编码器提取生成图像(fakeB)的特征
   - 与KgCoOp训练的文本特征计算相似度
   - 确保生成图像具有正确的语义类别
   
2. 蒸馏损失 (Distillation Loss):
   - 使用CONCH同时编码真实图像(realB)和生成图像(fakeB)
   - 约束两者在CONCH特征空间中相似
   - 确保语义信息从真实图像传递到生成图像

完整损失函数：
loss_total = loss_GAN + loss_NCE + λ_cls * loss_CLS + λ_distill * loss_DISTILL
             ^^^^^^^^   ^^^^^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
             对抗损失    对比损失    分类监督(新)         知识蒸馏(新)

CONCH在PTCUT中的角色：
- ✅ 提供预训练的视觉编码器（冻结,不训练）
- ✅ 作为"语义评估器"指导生成器学习
- ✅ 确保生成图像保持正确的病理学类别特征
- ✅ logit_scale提供正确的相似度温度参数
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os.path as osp
import sys

# ============================================================================
# 导入CONCH模型
# ============================================================================
# CONCH提供预训练的视觉-语言编码器,用于提取图像的语义特征
conch_path = "/home/lzh/myCode/CONCH"
if conch_path not in sys.path:
    sys.path.insert(0, conch_path)
from conch.open_clip_custom import create_model_from_pretrained


def build_conch_preprocess(image_size=448):
    """
    构建 CONCH/CLIP 的正确预处理流程
    
    关键：
    - 使用 Resize(smaller_edge) + CenterCrop，而不是直接 Resize
    - 使用 CLIP/CONCH 的 mean/std，而不是 ImageNet 的
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP/CONCH mean
            std=[0.26862954, 0.26130258, 0.27577711]   # CLIP/CONCH std
        ),
    ])


class PTCUTModel(BaseModel):
    """
    PTCUT模型类：基于CUT模型的提示调优虚拟染色模型
    
    在CUT模型的基础上添加：
    - CONCH视觉编码器用于特征提取
    - 预训练的文本提示特征用于分类
    - 分类损失和蒸馏损失
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """配置PTCUT模型特有的参数选项"""
        # 继承CUT模型的所有参数
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='GAN损失的权重')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='NCE损失的权重')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, 
                          help='是否对identity映射使用NCE损失')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='在哪些层上计算NCE损失')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                          type=util.str2bool, nargs='?', const=True, default=False,
                          help='计算对比损失时是否包含minibatch中其他样本的负样本')
        parser.add_argument('--netF', type=str, default='mlp_sample', 
                          choices=['sample', 'reshape', 'mlp_sample'], help='特征图的下采样方式')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='NCE损失的温度参数')
        parser.add_argument('--num_patches', type=int, default=256, help='每层采样的patch数量')
        parser.add_argument('--flip_equivariance',
                          type=util.str2bool, nargs='?', const=True, default=False,
                          help="强制翻转等变性作为额外正则项")

        # PTCUT特有参数
        parser.add_argument('--lambda_cls', type=float, default=0.5, 
                          help='生成器分类损失的权重')
        parser.add_argument('--lambda_cls_d', type=float, default=0.0, 
                          help='判别器分类损失的权重（辅助分类器GAN）')
        parser.add_argument('--lambda_distill', type=float, default=0.5, 
                          help='蒸馏损失的权重')
        parser.add_argument('--cls_temperature', type=float, default=5.0,
                          help='分类损失的温度参数，用于缩放CONCH的logit_scale (默认5.0，降低损失值)')
        
        # CONCH 和 prompt features 加载参数
        parser.add_argument('--conch_checkpoint', type=str,
                          default='/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin',
                          help='CONCH 预训练模型路径')
        parser.add_argument('--prompt_features_path', type=str,
                          default='/home/lzh/myCode/KgCoOp/KgCoOp/output/gnb_kgcoop_conch_csc_2class_nodular_vs_composite/prompt_text_features.pth',
                          help='KgCoOp 训练好的 prompt text features 路径')
        parser.add_argument('--num_classes', type=int, default=2, 
                          help='分类数量（2分类：i/n）')
        parser.add_argument('--use_labels', type=util.str2bool, nargs='?', const=True, default=True,
                          help='是否使用图像标签进行分类损失计算（从文件名提取）')

        parser.set_defaults(pool_size=0)
        
        # PTCUT默认使用ptcut数据集（共享裁剪参数）
        parser.set_defaults(dataset_mode='ptcut')

        opt, _ = parser.parse_known_args()

        # 为CUT和FastCUT设置默认参数
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        """初始化PTCUT模型
        
        核心组件：
        1. CUT组件：
           - netG: 生成器 (训练)
           - netD: 判别器 (训练)
           - netF: 特征提取器 (训练)
        
        2. CONCH组件 (新增):
           - conch_model.visual: 视觉编码器 (冻结,不训练)
           - conch_model.logit_scale: 相似度缩放 (冻结)
        
        3. KgCoOp组件 (新增):
           - prompt_text_features: 文本特征 (冻结,来自KgCoOp训练)
        """
        BaseModel.__init__(self, opt)

        # 指定需要打印的训练损失
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        
        # 添加PTCUT特有的损失
        if opt.lambda_cls > 0:
            self.loss_names.append('CLS')  # 生成器分类损失
            if getattr(opt, 'lambda_cls_d', 0) > 0:
                self.loss_names.append('CLS_D')  # 判别器分类损失
        if opt.lambda_distill > 0:
            self.loss_names.append('DISTILL')  # 蒸馏损失
            
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        # ====================================================================
        # 定义CUT网络（生成器和判别器）
        # ====================================================================
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                     opt.normG, not opt.no_dropout, opt.init_type, 
                                     opt.init_gain, opt.no_antialias, opt.no_antialias_up, 
                                     self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
                                     opt.init_type, opt.init_gain, opt.no_antialias, 
                                     self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                         opt.normD, opt.init_type, opt.init_gain, 
                                         opt.no_antialias, self.gpu_ids, opt)
            
            # 如果启用判别器分类损失，添加分类头
            if getattr(opt, 'lambda_cls_d', 0) > 0:
                # 判别器分类头：从判别器特征到类别预测
                # 使用全局平均池化 + 全连接层
                self.netD_classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                    nn.Flatten(),
                    nn.Linear(opt.ndf * min(2 ** opt.n_layers_D, 8), opt.num_classes)
                ).to(self.device)
                self.model_names.append('D_classifier')

            # ================================================================
            # 定义CUT损失函数
            # ================================================================
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            
            # ================================================================
            # 定义PTCUT特有损失函数
            # ================================================================
            if opt.lambda_cls > 0:
                # 分类损失：使用标准交叉熵
                self.criterionCLS = nn.CrossEntropyLoss().to(self.device)
            if opt.lambda_distill > 0:
                # 蒸馏损失：使用余弦相似度损失
                # target=1表示希望realB和fakeB的特征相似
                self.criterionDistill = nn.CosineEmbeddingLoss().to(self.device)
            
            # ================================================================
            # 步骤2: 加载 CONCH 模型和 prompt text features
            # ================================================================
            # 直接加载预训练的 CONCH 和预先保存的 prompt text features
            # 无需加载整个 KgCoOp 模型，更简洁高效
            print("正在加载 CONCH 和 prompt text features...")
            self.load_conch_and_features(opt)
            
            # ================================================================
            # 定义优化器（只优化CUT的网络）
            # ================================================================
            # 注意：CONCH和KgCoOp特征不参与优化
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, 
                                              betas=(opt.beta1, opt.beta2))
            
            # 判别器优化器：如果有分类头，一起优化
            if getattr(opt, 'lambda_cls_d', 0) > 0:
                d_params = list(self.netD.parameters()) + list(self.netD_classifier.parameters())
                self.optimizer_D = torch.optim.Adam(d_params, lr=opt.lr, 
                                                  betas=(opt.beta1, opt.beta2))
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, 
                                                  betas=(opt.beta1, opt.beta2))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def load_conch_and_features(self, opt):
        """
        直接加载 CONCH 模型和预保存的 prompt text features
        
        策略（最简化版）：
        1. 直接加载 CONCH 模型（只需要 visual encoder）
        2. 直接加载预保存的 prompt_text_features.pth
        
        优势：
        - 无需加载 KgCoOp 的 text encoder 和 prompt learner
        - 加载速度更快，内存占用更少
        
        最终保留：
        - self.image_encoder: CONCH visual encoder (86M)
        - self.logit_scale: 相似度缩放参数
        - self.prompt_text_features: [num_classes, 512]
        - self.conch_preprocess: CONCH 预处理流程
        
        参数:
            opt: 命令行选项
        """
        print(f"\n正在加载 CONCH 模型和 prompt text features...")
        print(f"  CONCH checkpoint: {opt.conch_checkpoint}")
        print(f"  Prompt features: {opt.prompt_features_path}")
        print(f"  类别数: {opt.num_classes}")
        
        # ====================================================================
        # 步骤1: 加载 CONCH 模型
        # ====================================================================
        print("  加载 CONCH 预训练模型...")
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", 
            checkpoint_path=opt.conch_checkpoint
        )
        conch_model = conch_model.to(self.device)
        conch_model.eval()
        
        # 提取 image encoder 和 logit_scale
        self.image_encoder = conch_model.visual
        self.logit_scale = conch_model.logit_scale
        
        # 冻结 image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        print(f"✓ CONCH image encoder 已加载并冻结")
        print(f"✓ Logit scale: {self.logit_scale.exp().item():.4f}")
        
        # ====================================================================
        # 步骤2: 加载预保存的 prompt text features
        # ====================================================================
        if not osp.exists(opt.prompt_features_path):
            raise FileNotFoundError(f"Prompt features 文件不存在: {opt.prompt_features_path}")
        
        print(f"  加载 prompt text features...")
        text_features = torch.load(opt.prompt_features_path, map_location=self.device, weights_only=True)
        
        # 确保已归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 验证形状
        if text_features.size(0) != opt.num_classes:
            raise ValueError(
                f"文本特征数量与类别数量不匹配\n"
                f"  期望: {opt.num_classes} 个类别\n"
                f"  实际: {text_features.size(0)} 个特征\n"
                f"  特征形状: {text_features.shape}"
            )
        
        self.prompt_text_features = text_features
        self.prompt_text_features.requires_grad = False
        
        print(f"✓ Prompt text features 已加载，形状: {text_features.shape}")
        
        # ====================================================================
        # 步骤3: 构建 CONCH 预处理流程
        # ====================================================================
        self.conch_preprocess = build_conch_preprocess(image_size=448)
        print(f"✓ CONCH 预处理流程已构建")

        # 缓存 mean/std 为固定张量，避免每次前向传播重复创建（替代 register_buffer）
        self.conch_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                       device=self.device).view(1, 3, 1, 1)
        self.conch_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                      device=self.device).view(1, 3, 1, 1)

        # ====================================================================
        # 步骤4: 清理不需要的组件
        # ====================================================================
        # 删除 CONCH 的 text encoder（只保留 visual）
        del conch_model.text
        del conch_model
        
        # 强制垃圾回收
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ 加载完成！")
        print(f"✓ 保留: image_encoder (86M) + text_features ({self.prompt_text_features.numel()} 元素)\n")



    def extract_class_labels(self, image_paths):
        """
        从图像路径中提取类别标签
        
        支持两种文件名格式:
        1. 旧格式: *_X.jpg，其中X是类别号(1,2,3,4)
        2. 新格式: *_label.jpg，其中label是字母标签（如 i, n）
        
        参数:
            image_paths: 图像路径列表
        返回:
            labels: 类别标签张量 (batch_size,)，类别从0开始索引
        """
        # 定义标签映射（字母 -> 数字索引）
        label_map = {
            'i': 0,  # intermixed/composite
            'n': 1,  # nodular
            # 兼容4类情况
            '1': 0, '2': 1, '3': 2, '4': 3
        }
        
        labels = []
        for path in image_paths:
            # 提取文件名（不含扩展名）
            filename = osp.splitext(osp.basename(path))[0]
            
            # 尝试提取标签（最后一个下划线后的部分）
            try:
                # 分割文件名，获取最后一部分
                parts = filename.split('_')
                label_str = parts[-1]  # 最后一部分应该是标签
                
                if label_str in label_map:
                    label = label_map[label_str]
                elif label_str.isdigit():
                    # 如果是数字，转换为0-indexed
                    label = int(label_str) - 1
                else:
                    raise ValueError(f"未知标签: {label_str}")
                    
            except Exception as e:
                print(f"警告: 无法从 {path} 提取类别标签 ({e})，使用默认值0")
                label = 0
                
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def data_dependent_initialize(self, data):
        """
        特征网络netF的定义依赖于netG编码器部分中间特征的形状
        """
        import gc

        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        
        print(f"  [DEBUG] real_A shape: {self.real_A.shape}, real_B shape: {self.real_B.shape}")
        
        self.forward()
        
        if self.opt.isTrain:
            self.compute_D_loss().backward()
            
            self.compute_G_loss().backward()
            
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                   betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        """优化参数：更新判别器D和生成器G"""
        # 前向传播
        self.forward()

        # 更新判别器D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # 更新生成器G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """从dataloader中解包输入数据并进行必要的预处理"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        # 提取类别标签（如果使用分类损失）
        # 注意：必须在训练时且启用标签时才提取
        if self.isTrain and self.opt.lambda_cls > 0 and self.opt.use_labels:
            self.labels = self.extract_class_labels(self.image_paths)
        elif self.isTrain and self.opt.lambda_cls > 0 and not self.opt.use_labels:
            # 如果启用了 CLS 损失但没有启用标签，发出警告
            if not hasattr(self, '_label_warning_printed'):
                print("\n⚠️  警告: lambda_cls > 0 但 use_labels=False")
                print("  CLS 损失将无法提供有效监督")
                print("  建议设置 --use_labels True 以启用标签监督\n")
                self._label_warning_printed = True

    def forward(self):
        """前向传播；被<optimize_parameters>和<test>调用"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # 调试：打印输入尺寸
        if not hasattr(self, '_forward_debug_printed'):
            print(f"  [DEBUG] forward: self.real shape = {self.real.shape}")
            self._forward_debug_printed = True
        
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """计算判别器的GAN损失 + 分类损失（可选）"""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # 基础GAN损失
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        # 添加判别器分类损失（辅助分类器GAN, AC-GAN）
        if getattr(self.opt, 'lambda_cls_d', 0) > 0 and self.opt.use_labels:
            # 从真实图像的判别器特征中提取分类logits
            # 需要获取判别器的中间特征
            d_features_real = self.get_D_features(self.real_B)
            cls_logits_real = self.netD_classifier(d_features_real)
            
            # 使用真实标签计算分类损失
            if hasattr(self, 'labels'):
                self.loss_CLS_D = self.criterionCLS(cls_logits_real, self.labels) * self.opt.lambda_cls_d
                self.loss_D = self.loss_D + self.loss_CLS_D
                
                # 调试信息（仅首次）
                if not hasattr(self, '_cls_d_debug_printed'):
                    with torch.no_grad():
                        pred_labels = cls_logits_real.argmax(dim=1)
                        accuracy = (pred_labels == self.labels).float().mean().item()
                    print(f"\n[Discriminator CLS Loss Debug]")
                    print(f"  D features shape: {d_features_real.shape}")
                    print(f"  D logits shape: {cls_logits_real.shape}")
                    print(f"  D Batch accuracy: {accuracy:.2%}")
                    print(f"  D CLS Loss value: {self.loss_CLS_D.item():.4f}\n")
                    self._cls_d_debug_printed = True
        
        return self.loss_D
    
    def get_D_features(self, x):
        """从判别器中提取中间特征用于分类"""
        # 遍历判别器的层，提取倒数第二层（最后的LeakyReLU）的输出
        # 倒数第二层输出维度应该是 [batch, ndf*8, H, W]，即 [batch, 512, H, W]
        
        # 处理DataParallel包装的情况
        netD = self.netD.module if hasattr(self.netD, 'module') else self.netD
        
        if hasattr(netD, 'model'):
            # NLayerDiscriminator使用Sequential
            # model[-1] 是最后的 Conv2d(512, 1, 4, 1, 1) 输出真假预测
            # model[:-1] 是前面所有层，最后一层是 LeakyReLU
            features = x
            # 排除最后一层（1x1卷积层），保留到LeakyReLU
            for i, layer in enumerate(netD.model[:-1]):
                features = layer(features)
            # 现在 features 应该是 [batch, 512, H, W]
            return features
        else:
            # 其他判别器类型，直接使用输入（可能需要根据具体类型调整）
            raise NotImplementedError("当前只支持NLayerDiscriminator的分类头")

    def compute_G_loss(self):
        """计算生成器的损失：GAN + NCE + CLS + DISTILL"""
        fake = self.fake_B

        # 1. GAN损失
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # 2. NCE损失
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
            if self.opt.nce_idt:
                self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
                loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
            else:
                loss_NCE_both = self.loss_NCE
        else:
            loss_NCE_both = 0.0

        # ====================================================================
        # 🚀 提速优化区: 统一提取 CONCH 特征，避免重复计算
        # ====================================================================
        if self.opt.lambda_cls > 0.0 or self.opt.lambda_distill > 0.0:
            _, _, h, w = self.fake_B.shape
            # 仅生成一次同位坐标，CLS 和 DISTILL 共享同一裁剪区域
            i, j = self.get_random_crop_coords(h, w, crop_size=448)

            fake_B_conch_in = self.differentiable_conch_preprocess(self.fake_B, i, j, crop_size=448)
            real_B_conch_in = self.differentiable_conch_preprocess(self.real_B, i, j, crop_size=448)

            # 使用 AMP 加速 ViT 的矩阵乘法（Tensor Core），显著提升速度并节省显存
            from torch.amp import autocast
            with autocast('cuda'):
                # fake_B 特征：必须允许梯度回传到生成器
                fake_B_features = self.image_encoder(fake_B_conch_in)
                if isinstance(fake_B_features, tuple):
                    fake_B_features = fake_B_features[0]
                fake_B_features = fake_B_features / fake_B_features.norm(dim=-1, keepdim=True)

                # real_B 特征：真实图像，不需要求导
                with torch.no_grad():
                    real_B_features = self.image_encoder(real_B_conch_in)
                    if isinstance(real_B_features, tuple):
                        real_B_features = real_B_features[0]
                    real_B_features = real_B_features / real_B_features.norm(dim=-1, keepdim=True)

            # 转回 FP32 保证损失计算的数值稳定性
            self.fake_B_conch_feat = fake_B_features.float()
            self.real_B_conch_feat = real_B_features.float()

        # 3. 分类损失 (直接传入预计算好的特征，无额外推理开销)
        if self.opt.lambda_cls > 0.0:
            self.loss_CLS = self.compute_classification_loss(self.fake_B_conch_feat)
        else:
            self.loss_CLS = 0.0

        # 4. 蒸馏损失 (直接传入预计算好的特征，无额外推理开销)
        if self.opt.lambda_distill > 0.0:
            self.loss_DISTILL = self.compute_distillation_loss(self.fake_B_conch_feat, self.real_B_conch_feat)
        else:
            self.loss_DISTILL = 0.0

        # 总生成器损失
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_CLS + self.loss_DISTILL
        return self.loss_G


    def get_random_crop_coords(self, h, w, crop_size=448):
        """生成随机裁剪的坐标，用于从高分辨率图像中提取 patch"""
        if h <= crop_size or w <= crop_size:
            return 0, 0
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        return i, j

    def differentiable_conch_preprocess(self, image_tensor, i, j, crop_size=448):
        """全过程可导的预处理 (极速版：使用缓存的 mean/std buffer)"""
        # 1. 还原到 [0, 1]
        img = (image_tensor + 1.0) / 2.0
        # 2. 空间裁剪 (切片操作完全保留梯度)
        img_crop = img[:, :, i:i+crop_size, j:j+crop_size]
        # 3. 标准化 (直接使用缓存的张量，避免重新开辟显存)
        img_norm = (img_crop - self.conch_mean) / self.conch_std
        return img_norm

    def compute_classification_loss(self, fake_B_features):
        """计算分类损失 (极速版：直接使用预先计算好的特征)"""
        if not self.opt.use_labels or not hasattr(self, 'labels'):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        logit_scale = self.logit_scale.exp()
        temperature = getattr(self.opt, 'cls_temperature', 1.0)
        logits = (logit_scale / temperature) * fake_B_features @ self.prompt_text_features.t()

        loss_cls = self.criterionCLS(logits, self.labels) * self.opt.lambda_cls
        return loss_cls

    def compute_distillation_loss(self, fake_B_features, real_B_features):
        """计算蒸馏损失 (极速版：直接使用预先计算好的特征)"""
        target = torch.ones(real_B_features.size(0), device=self.device)
        loss_distill = self.criterionDistill(
            fake_B_features,
            real_B_features,
            target
        ) * self.opt.lambda_distill
        return loss_distill

    def calculate_NCE_loss(self, src, tgt):
        """计算NCE对比损失"""
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
