"""
PTCUT Model (Prompt-Tuned CUT)
基于CUT模型的虚拟染色模型，结合训练好的提示调优特征

该模型在CUT的基础上添加两个新损失：
1. 分类损失(Classification Loss): 将生成图像(fakeB)通过CONCH视觉编码器提取特征，
   与预训练的提示文本特征计算相似度，作为分类任务的监督信号
2. 蒸馏损失(Distillation Loss): 将真实图像(realB)和生成图像(fakeB)都通过CONCH
   视觉编码器，计算两者特征的相似度，确保生成图像保留语义信息
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os.path as osp
import sys

# 导入CONCH模型
conch_path = "/home/lzh/myCode/CONCH"
if conch_path not in sys.path:
    sys.path.insert(0, conch_path)
from conch.open_clip_custom import create_model_from_pretrained


def load_conch_to_cpu():
    """加载CONCH模型到CPU"""
    checkpoint_path = "/home/lzh/myCode/CONCH/checkpoints/conch/pytorch_model.bin"
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    return model


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
                          help='分类损失的权重')
        parser.add_argument('--lambda_distill', type=float, default=0.5, 
                          help='蒸馏损失的权重')
        parser.add_argument('--prompt_text_features', type=str, 
                          default='/home/lzh/myCode/myKgCoOp/myKgCoOp/output/gnb_kgcoop_4class/prompt_text_features.pt',
                          help='预编码的文本特征文件路径（.pt格式）')
        parser.add_argument('--num_classes', type=int, default=4, 
                          help='分类数量（对应提示数量）')
        parser.add_argument('--use_labels', type=util.str2bool, nargs='?', const=True, default=True,
                          help='是否使用图像标签进行分类损失计算（从文件名提取）')

        parser.set_defaults(pool_size=0)

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
        BaseModel.__init__(self, opt)

        # 指定需要打印的训练损失
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        
        # 添加PTCUT特有的损失
        if opt.lambda_cls > 0:
            self.loss_names.append('CLS')
        if opt.lambda_distill > 0:
            self.loss_names.append('DISTILL')
            
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        # 定义网络（生成器和判别器）- 继承自CUT
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

            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            
            # PTCUT特有损失函数
            if opt.lambda_cls > 0:
                self.criterionCLS = nn.CrossEntropyLoss().to(self.device)
            if opt.lambda_distill > 0:
                self.criterionDistill = nn.CosineEmbeddingLoss().to(self.device)
            
            # 加载CONCH模型
            print("正在加载CONCH模型用于特征提取...")
            self.conch_model = load_conch_to_cpu()
            if len(self.gpu_ids) > 0:
                self.conch_model = self.conch_model.to(self.device)
            self.conch_model.eval()  # 设置为评估模式
            
            # 冻结CONCH模型参数
            for param in self.conch_model.parameters():
                param.requires_grad = False
            print("CONCH模型加载完成并已冻结参数")
            
            # 加载预编码的文本特征
            self.load_prompt_features(opt.prompt_text_features, opt.num_classes)
            
            # 定义优化器
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, 
                                              betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, 
                                              betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def load_prompt_features(self, features_path, num_classes):
        """
        从.pt文件加载预编码的文本特征
        
        参数:
            features_path: 文本特征文件路径（.pt格式）
            num_classes: 类别数量
        """
        print(f"正在从 {features_path} 加载预编码文本特征...")
        
        if not osp.exists(features_path):
            raise FileNotFoundError(
                f"找不到文本特征文件: {features_path}\n"
                f"请确保已从KgCoOp checkpoint提取并保存text_features到.pt文件\n"
                f"提取方法:\n"
                f"  checkpoint = torch.load('path/to/kgcoop/checkpoint.pth.tar')\n"
                f"  text_features = checkpoint['state_dict']['text_features']\n"
                f"  torch.save(text_features, '{features_path}')"
            )
        
        # 直接加载.pt文件
        text_features = torch.load(features_path, map_location='cpu', weights_only=False)
        
        # 验证形状
        if not isinstance(text_features, torch.Tensor):
            raise TypeError(f"加载的特征必须是torch.Tensor，实际类型: {type(text_features)}")
        
        if text_features.dim() != 2:
            raise ValueError(f"特征维度必须是2D (num_classes, feature_dim)，实际形状: {text_features.shape}")
        
        if text_features.size(0) != num_classes:
            raise ValueError(
                f"特征数量与类别数量不匹配\n"
                f"  期望: {num_classes} 个类别\n"
                f"  实际: {text_features.size(0)} 个特征\n"
                f"  特征形状: {text_features.shape}"
            )
        
        print(f"✓ 成功加载文本特征，形状: {text_features.shape}")
        
        # 注册为buffer（不参与梯度计算）
        self.register_buffer('prompt_text_features', text_features)
        print(f"✓ 文本特征已注册为模型buffer")

    def extract_class_labels(self, image_paths):
        """
        从图像路径中提取类别标签
        假设文件名格式为: *_X.png，其中X是类别号(1,2,3,4)
        
        参数:
            image_paths: 图像路径列表
        返回:
            labels: 类别标签张量 (batch_size,)，类别从0开始索引
        """
        labels = []
        for path in image_paths:
            # 提取文件名（不含扩展名）
            filename = osp.splitext(osp.basename(path))[0]
            # 假设最后一个字符是类别号
            try:
                class_id = int(filename[-1])  # 1, 2, 3, 4
                label = class_id - 1  # 转换为0-indexed: 0, 1, 2, 3
            except:
                print(f"警告: 无法从 {path} 提取类别标签，使用默认值0")
                label = 0
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def data_dependent_initialize(self, data):
        """
        特征网络netF的定义依赖于netG编码器部分中间特征的形状
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
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
        
        # 提取类别标签（如果使用）
        if self.isTrain and self.opt.use_labels and self.opt.lambda_cls > 0:
            self.labels = self.extract_class_labels(self.image_paths)

    def forward(self):
        """前向传播；被<optimize_parameters>和<test>调用"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """计算判别器的GAN损失"""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

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
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # 3. 分类损失 (Classification Loss)
        if self.opt.lambda_cls > 0.0:
            self.loss_CLS = self.compute_classification_loss()
        else:
            self.loss_CLS = 0.0

        # 4. 蒸馏损失 (Distillation Loss)
        if self.opt.lambda_distill > 0.0:
            self.loss_DISTILL = self.compute_distillation_loss()
        else:
            self.loss_DISTILL = 0.0

        # 总生成器损失
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_CLS + self.loss_DISTILL
        return self.loss_G

    def compute_classification_loss(self):
        """
        计算分类损失
        将fakeB通过CONCH视觉编码器，与预训练的prompt text features比较
        """
        with torch.no_grad():
            # 提取fakeB的视觉特征
            fake_B_features = self.conch_model.encode_image(self.fake_B, proj_contrast=False, normalize=False)
            if isinstance(fake_B_features, tuple):
                fake_B_features = fake_B_features[0]
            fake_B_features = fake_B_features / fake_B_features.norm(dim=-1, keepdim=True)
        
        # 计算与所有类别prompt features的相似度
        # fake_B_features: (batch_size, 512)
        # prompt_text_features: (num_classes, 512)
        logit_scale = self.conch_model.logit_scale.exp()
        logits = logit_scale * fake_B_features @ self.prompt_text_features.t()  # (batch_size, num_classes)
        
        if self.opt.use_labels and hasattr(self, 'labels'):
            # 使用真实标签计算交叉熵损失
            loss_cls = self.criterionCLS(logits, self.labels) * self.opt.lambda_cls
        else:
            # 无监督：使用最高置信度的伪标签
            pseudo_labels = logits.argmax(dim=1)
            loss_cls = self.criterionCLS(logits, pseudo_labels) * self.opt.lambda_cls
        
        return loss_cls

    def compute_distillation_loss(self):
        """
        计算蒸馏损失
        确保fakeB和realB在CONCH特征空间中相似
        """
        with torch.no_grad():
            # 提取realB和fakeB的视觉特征
            real_B_features = self.conch_model.encode_image(self.real_B, proj_contrast=False, normalize=False)
            fake_B_features = self.conch_model.encode_image(self.fake_B, proj_contrast=False, normalize=False)
            
            if isinstance(real_B_features, tuple):
                real_B_features = real_B_features[0]
            if isinstance(fake_B_features, tuple):
                fake_B_features = fake_B_features[0]
            
            # L2归一化
            real_B_features = real_B_features / real_B_features.norm(dim=-1, keepdim=True)
            fake_B_features = fake_B_features / fake_B_features.norm(dim=-1, keepdim=True)
        
        # 使用余弦相似度损失
        # target=1表示我们希望两个特征相似
        target = torch.ones(real_B_features.size(0), device=self.device)
        loss_distill = self.criterionDistill(fake_B_features, real_B_features, target) * self.opt.lambda_distill
        
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
