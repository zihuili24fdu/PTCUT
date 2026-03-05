"""PT-Pix2Pix Model (Prompt-Tuned Pix2Pix)

基于Pix2Pix模型的虚拟染色模型，结合KgCoOp训练好的提示特征和CONCH视觉编码器

============================================================================
核心思想：在Pix2Pix的基础上添加CONCH语义监督
============================================================================

标准Pix2Pix损失：
loss_total = loss_GAN + λ_L1 * loss_L1

PT-Pix2Pix新增损失（针对 G(A)=fake_B）：

1. 分类损失 (Classification Loss):
   - 使用CONCH视觉编码器提取生成图像(fake_B)的特征
   - 与KgCoOp训练的文本特征计算相似度
   - 确保生成图像具有正确的语义类别

2. 蒸馏损失 (Distillation Loss):
   - 使用CONCH同时编码真实图像(real_B)和生成图像(fake_B)
   - 约束两者在CONCH特征空间中相似
   - 确保语义信息从真实图像传递到生成图像

完整损失函数：
loss_G = loss_GAN + λ_L1 * loss_L1 + λ_cls * loss_CLS + λ_distill * loss_DISTILL
"""

import random
import torch
import torch.nn as nn
import os.path as osp
import sys
from torchvision import transforms
from .base_model import BaseModel
from . import networks
import util.util as util

# ============================================================================
# 导入CONCH模型
# ============================================================================
conch_path = "/home/lzh/myCode/CONCH"
if conch_path not in sys.path:
    sys.path.insert(0, conch_path)
from conch.open_clip_custom import create_model_from_pretrained


def build_conch_preprocess(image_size=448):
    """构建 CONCH/CLIP 的正确预处理流程"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


class PTPix2PixModel(BaseModel):
    """
    PT-Pix2Pix 模型类：基于Pix2Pix的提示调优虚拟染色模型

    在Pix2Pix的基础上添加：
    - CONCH视觉编码器用于特征提取（冻结）
    - 预训练的文本提示特征用于分类（来自KgCoOp）
    - 针对 G(A)=fake_B 的分类损失和蒸馏损失
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加PT-Pix2Pix特有参数选项"""
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0,
                                help='L1损失的权重')

            # PT 新增参数
            parser.add_argument('--lambda_cls', type=float, default=0.5,
                                help='生成器分类损失的权重')
            parser.add_argument('--lambda_distill', type=float, default=0.5,
                                help='蒸馏损失的权重')
            parser.add_argument('--cls_temperature', type=float, default=5.0,
                                help='分类损失温度参数（用于缩放fake_B logits）')
            parser.add_argument('--cls_soft_temperature', type=float, default=10.0,
                                help='软标签生成温度（将real_B logits转化为软分布）')
            parser.add_argument('--cls_warmup_epochs', type=int, default=30,
                                help='cls损失预热期：前N个epoch权重为0')
            parser.add_argument('--cls_rampup_epochs', type=int, default=20,
                                help='cls损失线性爬坡期：warmup后经N个epoch线性增长到lambda_cls')

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

        return parser

    def __init__(self, opt):
        """初始化PT-Pix2Pix模型"""
        BaseModel.__init__(self, opt)

        self.current_epoch = getattr(opt, 'epoch_count', 0)

        # 损失名称
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.isTrain:
            if opt.lambda_cls > 0.0:
                self.loss_names.append('CLS')
            if opt.lambda_distill > 0.0:
                self.loss_names.append('DISTILL')

        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 定义生成器
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # 定义判别器（条件GAN需要同时输入A和B）
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # 标准Pix2Pix损失
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # PTCUT 新增损失
            if opt.lambda_cls > 0.0:
                self.criterionCLS = nn.CrossEntropyLoss().to(self.device)
            if opt.lambda_distill > 0.0:
                self.criterionDistill = nn.CosineEmbeddingLoss().to(self.device)

            # 加载 CONCH 和 prompt text features
            if opt.lambda_cls > 0.0 or opt.lambda_distill > 0.0:
                print("正在加载 CONCH 和 prompt text features...")
                self.load_conch_and_features(opt)

            # 优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def load_conch_and_features(self, opt):
        """加载 CONCH 视觉编码器和预保存的 prompt text features"""
        print(f"\n正在加载 CONCH 模型和 prompt text features...")
        print(f"  CONCH checkpoint: {opt.conch_checkpoint}")
        print(f"  Prompt features: {opt.prompt_features_path}")
        print(f"  类别数: {opt.num_classes}")

        # 加载 CONCH 模型
        print("  加载 CONCH 预训练模型...")
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=opt.conch_checkpoint
        )
        conch_model = conch_model.to(self.device)
        conch_model.eval()

        self.image_encoder = conch_model.visual
        self.logit_scale = conch_model.logit_scale

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        print(f"✓ CONCH image encoder 已加载并冻结")
        print(f"✓ Logit scale: {self.logit_scale.exp().item():.4f}")

        # 加载 prompt text features
        if not osp.exists(opt.prompt_features_path):
            raise FileNotFoundError(f"Prompt features 文件不存在: {opt.prompt_features_path}")

        print(f"  加载 prompt text features...")
        text_features = torch.load(opt.prompt_features_path, map_location=self.device, weights_only=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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

        # 构建预处理流程
        self.conch_preprocess = build_conch_preprocess(image_size=448)

        # 缓存 mean/std 张量
        self.conch_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                       device=self.device).view(1, 3, 1, 1)
        self.conch_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                      device=self.device).view(1, 3, 1, 1)

        # 清理不需要的组件
        del conch_model.text
        del conch_model

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ 加载完成！\n")

    def get_random_crop_coords(self, h, w, crop_size=448):
        """生成随机裁剪坐标"""
        if h <= crop_size or w <= crop_size:
            return 0, 0
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        return i, j

    def differentiable_conch_preprocess(self, image_tensor, i, j, crop_size=448):
        """全过程可导的预处理"""
        img = (image_tensor + 1.0) / 2.0
        img_crop = img[:, :, i:i + crop_size, j:j + crop_size]
        img_norm = (img_crop - self.conch_mean) / self.conch_std
        return img_norm

    def compute_classification_loss(self, fake_B_features, real_B_features=None):
        """计算分类损失（硬标签交叉熵，支持渐进式权重调度）"""
        if not self.opt.use_labels or not hasattr(self, 'labels'):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        warmup = getattr(self.opt, 'cls_warmup_epochs', 30)
        rampup = getattr(self.opt, 'cls_rampup_epochs', 20)
        epoch = getattr(self, 'current_epoch', 0)
        if epoch < warmup:
            return torch.tensor(0.0, device=self.device)
        ramp_factor = min(1.0, (epoch - warmup) / max(rampup, 1))
        effective_lambda = self.opt.lambda_cls * ramp_factor

        logit_scale = self.logit_scale.exp()
        temperature = getattr(self.opt, 'cls_temperature', 5.0)

        logits = (logit_scale / temperature) * fake_B_features @ self.prompt_text_features.t()
        loss_cls = self.criterionCLS(logits, self.labels) * effective_lambda
        return loss_cls

    def compute_distillation_loss(self, fake_B_features, real_B_features):
        """计算蒸馏损失（余弦相似度）"""
        target = torch.ones(real_B_features.size(0), device=self.device)
        loss_distill = self.criterionDistill(
            fake_B_features,
            real_B_features,
            target
        ) * self.opt.lambda_distill
        return loss_distill

    def set_input(self, input):
        """解包dataloader数据"""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # 读取标签（用于CLS损失）
        if self.isTrain and self.opt.lambda_cls > 0.0 and self.opt.use_labels:
            label_key = 'A_label' if AtoB else 'B_label'
            if label_key in input:
                self.labels = input[label_key].to(self.device)
            else:
                if not hasattr(self, '_label_warning_printed'):
                    print(f"\n⚠️  警告: Dataset 未返回 '{label_key}'，CLS 损失将被跳过。\n")
                    self._label_warning_printed = True

    def forward(self):
        """前向传播"""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """计算判别器GAN损失并反向传播"""
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # 合并损失
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """计算生成器损失：GAN + L1 + CLS + DISTILL"""
        # GAN loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # ====================================================================
        # CONCH 语义损失（针对 fake_B = G(A)）
        # ====================================================================
        self.loss_CLS = 0.0
        self.loss_DISTILL = 0.0

        if self.opt.lambda_cls > 0.0 or self.opt.lambda_distill > 0.0:
            _, _, h, w = self.fake_B.shape
            i, j = self.get_random_crop_coords(h, w, crop_size=448)

            fake_B_conch_in = self.differentiable_conch_preprocess(self.fake_B, i, j, crop_size=448)
            real_B_conch_in = self.differentiable_conch_preprocess(self.real_B, i, j, crop_size=448)

            from torch.amp import autocast
            with autocast('cuda'):
                fake_B_features = self.image_encoder(fake_B_conch_in)
                if isinstance(fake_B_features, tuple):
                    fake_B_features = fake_B_features[0]
                fake_B_features = fake_B_features / fake_B_features.norm(dim=-1, keepdim=True)

                with torch.no_grad():
                    real_B_features = self.image_encoder(real_B_conch_in)
                    if isinstance(real_B_features, tuple):
                        real_B_features = real_B_features[0]
                    real_B_features = real_B_features / real_B_features.norm(dim=-1, keepdim=True)

            fake_B_feat = fake_B_features.float()
            real_B_feat = real_B_features.float()

            if self.opt.lambda_cls > 0.0:
                self.loss_CLS = self.compute_classification_loss(fake_B_feat, real_B_feat)

            if self.opt.lambda_distill > 0.0:
                self.loss_DISTILL = self.compute_distillation_loss(fake_B_feat, real_B_feat)

        # 总生成器损失
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_CLS + self.loss_DISTILL
        self.loss_G.backward()

    def data_dependent_initialize(self, data):
        """数据依赖初始化（Pix2Pix不需要特殊初始化）"""
        return

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重"""
        self.forward()

        # 更新 D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # 更新 G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
