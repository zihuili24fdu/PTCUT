"""PT-CycleGAN Model (Prompt-Tuned CycleGAN)

基于CycleGAN模型的虚拟染色模型，结合KgCoOp训练好的提示特征和CONCH视觉编码器

============================================================================
核心思想：在CycleGAN的基础上添加CONCH语义监督
============================================================================

标准CycleGAN损失：
loss_total = loss_GAN + λ_A * loss_cycle_A + λ_B * loss_cycle_B + λ_idt * loss_idt

PT-CycleGAN新增损失（针对 G_A: A->B 的输出 fake_B）：

1. 分类损失 (Classification Loss):
   - 使用CONCH视觉编码器提取生成图像(fake_B)的特征
   - 与KgCoOp训练的文本特征计算相似度
   - 确保生成图像具有正确的语义类别

2. 蒸馏损失 (Distillation Loss):
   - 使用CONCH同时编码真实图像(real_B)和生成图像(fake_B)
   - 约束两者在CONCH特征空间中相似
   - 确保语义信息从真实图像传递到生成图像

完整损失函数：
loss_G = loss_GAN + loss_cycle + loss_idt + λ_cls * loss_CLS + λ_distill * loss_DISTILL
"""

import random
import torch
import torch.nn as nn
import itertools
import os.path as osp
import sys
from torchvision import transforms
from util.image_pool import ImagePool
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


class PTCycleGANModel(BaseModel):
    """
    PT-CycleGAN 模型类：基于CycleGAN的提示调优虚拟染色模型

    在CycleGAN的基础上添加：
    - CONCH视觉编码器用于特征提取（冻结）
    - 预训练的文本提示特征用于分类（来自KgCoOp）
    - 针对 G_A(A)=fake_B 的分类损失和蒸馏损失
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加PT-CycleGAN特有参数选项"""        # 使用与PTCUT相同的配对数据集：按文件名匹配、A/B共享裁剪参数、返回标签字段
        parser.set_defaults(dataset_mode='ptcut')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='cycle loss weight (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='cycle loss weight (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='identity mapping loss weight')

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
        """初始化PT-CycleGAN模型"""
        BaseModel.__init__(self, opt)

        self.current_epoch = getattr(opt, 'epoch_count', 0)

        # 损失名称
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        if self.isTrain:
            if opt.lambda_cls > 0.0:
                self.loss_names.append('CLS')
            if opt.lambda_distill > 0.0:
                self.loss_names.append('DISTILL')

        # 可视化图像
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        # 模型名称
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # 定义生成器和判别器
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain,
                                        opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain,
                                        opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            opt.no_antialias, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            opt.no_antialias, self.gpu_ids, opt=opt)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert opt.input_nc == opt.output_nc

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # 标准CycleGAN损失
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

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
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.fake_B = self.netG_A(self.real_A)   # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)    # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)   # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)    # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """计算判别器GAN损失并反向传播"""
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """计算 D_A 的GAN损失"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """计算 D_B 的GAN损失"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """计算生成器损失：GAN + Cycle + Idt + CLS + DISTILL"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # ====================================================================
        # CONCH 语义损失（针对 fake_B = G_A(A)，即目标域生成图像）
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
        self.loss_G = (self.loss_G_A + self.loss_G_B
                       + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_idt_A + self.loss_idt_B
                       + self.loss_CLS + self.loss_DISTILL)

        self.loss_G.backward()

    def data_dependent_initialize(self, data):
        """数据依赖初始化（CycleGAN不需要特殊初始化）"""
        return

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重"""
        self.forward()

        # 更新 G_A 和 G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 更新 D_A 和 D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
