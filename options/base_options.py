import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """该类定义了训练和测试时使用的选项。

    同时实现了如解析、打印和保存选项等辅助函数。
    还会收集数据集类和模型类中 <modify_commandline_options> 函数定义的额外选项。
    """

    def __init__(self, cmd_line=None):
        """重置类；表示该类尚未初始化"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """定义训练和测试时都用到的通用选项。"""
        # 基本参数
        parser.add_argument('--dataroot', default='placeholder', help='图片路径（应包含子文件夹 trainA, trainB, valA, valB 等）')
        parser.add_argument('--name', type=str, default='experiment_name', help='实验名称。决定样本和模型的存储位置')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='可解释名称')
        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU编号: 例如 0  0,1,2, 0,2。-1 表示使用CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存路径')
        # 模型参数
        parser.add_argument('--model', type=str, default='cut', help='选择使用的模型。')
        parser.add_argument('--input_nc', type=int, default=3, help='输入图片通道数: 3为RGB，1为灰度')
        parser.add_argument('--output_nc', type=int, default=3, help='输出图片通道数: 3为RGB，1为灰度')
        parser.add_argument('--ngf', type=int, default=64, help='生成器最后一层卷积的通道数')
        parser.add_argument('--ndf', type=int, default=64, help='判别器第一层卷积的通道数')
        parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='指定判别器结构。basic为70x70 PatchGAN。n_layers可指定判别器层数')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='指定生成器结构')
        parser.add_argument('--n_layers_D', type=int, default=3, help='仅当 netD==n_layers 时使用')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='生成器归一化方式: instance 或 batch')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='判别器归一化方式: instance 或 batch')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='网络初始化方式')
        parser.add_argument('--init_gain', type=float, default=0.02, help='normal、xavier和orthogonal的缩放因子')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='生成器是否不使用dropout')
        parser.add_argument('--no_antialias', action='store_true', help='若指定，则使用stride=2卷积代替抗锯齿下采样')
        parser.add_argument('--no_antialias_up', action='store_true', help='若指定，则使用[upconv(learned filter)]代替[upconv(硬编码[1,3,3,1]滤波器), conv]')
        # 数据集参数
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='选择数据集加载方式。 [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB 或 BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='若为True，按顺序取图片组成batch，否则随机取')
        parser.add_argument('--num_threads', default=4, type=int, help='加载数据的线程数')
        parser.add_argument('--batch_size', type=int, default=1, help='输入batch大小')
        parser.add_argument('--load_size', type=int, default=286, help='将图片缩放到该尺寸')
        parser.add_argument('--crop_size', type=int, default=256, help='再裁剪到该尺寸')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='每个数据集允许的最大样本数。若目录下样本数超过该值，仅加载子集。')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='加载时对图片的缩放和裁剪方式 [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='若指定，则不进行图片翻转数据增强')
        parser.add_argument('--display_winsize', type=int, default=256, help='visdom和HTML的显示窗口大小')
        parser.add_argument('--random_scale_max', type=float, default=3.0,
                            help='（用于单张图片翻译）以指定因子随机缩放图片作为数据增强。')
        # 其他参数
        parser.add_argument('--epoch', type=str, default='latest', help='加载哪个epoch？设为latest使用最新缓存模型')
        parser.add_argument('--verbose', action='store_true', help='若指定，打印更多调试信息')
        parser.add_argument('--suffix', default='', type=str, help='自定义后缀: opt.name = opt.name + suffix，例如 {model}_{netG}_size{load_size}')

        # 与StyleGAN2相关的参数
        parser.add_argument('--stylegan2_G_num_downsampling',
                            default=1, type=int,
                            help='StyleGAN2生成器使用的下采样层数')

        self.initialized = True
        return parser

    def gather_options(self):
        """初始化解析器（仅一次），添加模型和数据集的特定选项。
        这些选项在模型和数据集类的 <modify_commandline_options> 函数中定义。
        """
        if not self.initialized:  # 检查是否已初始化
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # 获取基本选项
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # 修改与模型相关的解析器选项
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # 用新默认值再次解析
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # 用新默认值再次解析

        # 修改与数据集相关的解析器选项
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # 保存并返回解析器
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """打印并保存选项

        会打印当前选项和默认值（如有不同）。
        并将选项保存到 / [checkpoints_dir] / opt.txt 文件中。
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # 保存到磁盘
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """解析选项，创建checkpoints目录后缀，并设置GPU设备。"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # 训练或测试

        # 处理 opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # 设置gpu编号
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
