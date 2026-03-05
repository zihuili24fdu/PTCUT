"""
PyramidPix2pix 模型 —— 集成到 PTCUT 框架，用于对比实验

核心思想（来源：BCI/PyramidPix2pix）：
  在 pix2pix 的 L1 损失基础上，增加多尺度高斯金字塔 L1 损失（L2/L3/L4），
  使生成器在不同频率尺度上都能与目标图像对齐，改善细节保真度。

网络结构：
  - 生成器：Attention U-Net (attention_unet_32)，在每个跳连接上施加空间注意力门控
  - 判别器：PatchGAN (basic)
  - 损失：GAN + pyramid-L1（4个尺度）

参数说明：
  --lambda_L1     原始尺度 L1 权重（默认 25.0）
  --weight_L2     1/2 尺度 L1 权重（默认 25.0）
  --weight_L3     1/4 尺度 L1 权重（默认 25.0）
  --weight_L4     1/8 尺度 L1 权重（默认 25.0）

尺度下采样方式（与原论文 BCI 保持一致）：
  对 fake_B 和 real_B 各做 5 次 Gaussian blur（kernel=3, sigma=1），
  再用 blur_pool2d(stride=2) 抗锯齿下采样到下一尺度。

依赖：kornia（用于 gaussian_blur2d / blur_pool2d）

用法示例：
  python train.py \\
    --dataroot /path/to/data \\
    --name gnb_pyramidp2p_224 \\
    --model pyramid_pix2pix \\
    --netG attention_unet_32 \\
    --norm batch \\
    --lambda_L1 25 --weight_L2 25 --weight_L3 25 --weight_L4 25 \\
    --gan_mode lsgan --batch_size 2
"""

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks

try:
    import kornia
    _KORNIA_AVAILABLE = True
except ImportError:
    _KORNIA_AVAILABLE = False


def _gaussian_blur_5x(x):
    """对输入张量施加 5 次高斯模糊（kernel=3×3, sigma=1），模拟高斯金字塔平滑。"""
    if _KORNIA_AVAILABLE:
        for _ in range(5):
            x = kornia.filters.gaussian_blur2d(x, (3, 3), (1, 1))
        return x
    else:
        # 纯 PyTorch 替代：用 5 次 avg_pool2d + 还原尺寸近似模拟
        # 注意：这只是近似，精确结果请安装 kornia
        for _ in range(5):
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x


def _downsample(x):
    """抗锯齿 2× 下采样（blur_pool2d，strip_factor=1, stride=2）。"""
    if _KORNIA_AVAILABLE:
        return kornia.filters.blur_pool2d(x, 1, stride=2)
    else:
        return F.avg_pool2d(x, kernel_size=2, stride=2)


def _pyramid_downscale(x):
    """一次高斯金字塔降级：5 次 Gaussian blur + 2× 下采样。"""
    return _downsample(_gaussian_blur_5x(x))


class PyramidPix2pixModel(BaseModel):
    """PyramidPix2pix：带多尺度金字塔 L1 损失的 Pix2Pix 变体（配合 Attention U-Net）。

    相比标准 pix2pix：
      1. 生成器替换为 Attention U-Net，跳连接处提供空间注意力门控
      2. 添加多尺度金字塔 L1 损失（原图 + 1/2 + 1/4 + 1/8），提升多频率保真度
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            norm='batch',
            dataset_mode='aligned',
            netG='attention_unet_32',
            gan_mode='lsgan',
        )
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_L1',  type=float, default=25.0,
                                help='原始尺度 L1 损失权重，对应金字塔第 1 级（默认 25）')
            parser.add_argument('--weight_L2',  type=float, default=25.0,
                                help='1/2 尺度 L1 损失权重，对应金字塔第 2 级（默认 25）')
            parser.add_argument('--weight_L3',  type=float, default=25.0,
                                help='1/4 尺度 L1 损失权重，对应金字塔第 3 级（默认 25）')
            parser.add_argument('--weight_L4',  type=float, default=25.0,
                                help='1/8 尺度 L1 损失权重，对应金字塔第 4 级（默认 25）')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'G_L1', 'G_L2', 'G_L3', 'G_L4', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # 生成器
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids
        )

        if self.isTrain:
            # 判别器（条件 PatchGAN，输入通道 = input_nc + output_nc）
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1  = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if not _KORNIA_AVAILABLE:
            print('[PyramidPix2pix] ⚠️  kornia 未安装，采用 avg_pool2d 近似金字塔下采样。'
                  '建议 pip install kornia 以精确复现论文结果。')

    # ------------------------------------------------------------------
    # 数据接口
    # ------------------------------------------------------------------
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    # ------------------------------------------------------------------
    # 判别器反向传播
    # ------------------------------------------------------------------
    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    # ------------------------------------------------------------------
    # 生成器反向传播：GAN + 4 级金字塔 L1 损失
    # ------------------------------------------------------------------
    def backward_G(self):
        # GAN 损失
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1（第 1 级，原始尺度）
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # 构建第 2 级（1/2 尺度）
        fake_s2 = _pyramid_downscale(self.fake_B)
        real_s2 = _pyramid_downscale(self.real_B)
        self.loss_G_L2 = self.criterionL1(fake_s2, real_s2) * self.opt.weight_L2

        # 构建第 3 级（1/4 尺度）
        fake_s3 = _pyramid_downscale(fake_s2)
        real_s3 = _pyramid_downscale(real_s2)
        self.loss_G_L3 = self.criterionL1(fake_s3, real_s3) * self.opt.weight_L3

        # 构建第 4 级（1/8 尺度）
        fake_s4 = _pyramid_downscale(fake_s3)
        real_s4 = _pyramid_downscale(real_s3)
        self.loss_G_L4 = self.criterionL1(fake_s4, real_s4) * self.opt.weight_L4

        self.loss_G = (self.loss_G_GAN
                       + self.loss_G_L1
                       + self.loss_G_L2
                       + self.loss_G_L3
                       + self.loss_G_L4)
        self.loss_G.backward()

    # ------------------------------------------------------------------
    # 参数优化
    # ------------------------------------------------------------------
    def optimize_parameters(self):
        self.forward()

        # 更新判别器
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # 更新生成器
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
