import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """
    这是一个模型的抽象基类（ABC）。
    要创建子类，需要实现以下五个函数：
        -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)。
        -- <set_input>:                     从数据集中解包数据并进行预处理。
        -- <forward>:                       产生中间结果。
        -- <optimize_parameters>:           计算损失、梯度，并更新网络权重。
        -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项。
    """

    def __init__(self, opt):
        """
        初始化 BaseModel 类。

        参数:
            opt (Option class) -- 存储所有实验参数；需要是 BaseOptions 的子类

        当创建自定义类时，需要实现自己的初始化。
        在此函数中，首先应调用 <BaseModel.__init__(self, opt)>
        然后，需要定义四个列表：
            -- self.loss_names (str list):          指定要绘制和保存的训练损失名称。
            -- self.model_names (str list):         指定要显示和保存的图像名称。
            -- self.visual_names (str list):        定义训练中使用的网络。
            -- self.optimizers (optimizer list):    定义并初始化优化器。每个网络可以定义一个优化器。如果两个网络同时更新，可以用 itertools.chain 组合。参见 cycle_gan_model.py 示例。
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 获取设备名：CPU 或 GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 所有检查点保存到 save_dir
        if opt.preprocess != 'scale_width':  # 使用 [scale_width] 时，输入图片可能有不同尺寸，会影响 cudnn.benchmark 性能
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # 用于 'plateau' 学习率策略

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        添加新的模型特定选项，并重写已有选项的默认值。

        参数:
            parser          -- 原始参数解析器
            is_train (bool) -- 是否为训练阶段。可用此标志添加训练或测试特定选项。

        返回:
            修改后的 parser。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """
        从 dataloader 解包输入数据并进行必要的预处理。

        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """
        前向传播；由 <optimize_parameters> 和 <test> 调用。
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        计算损失、梯度，并更新网络权重；每次训练迭代时调用。
        """
        pass

    def setup(self, opt):
        """
        加载并打印网络结构；创建学习率调度器

        参数:
            opt (Option class) -- 存储所有实验参数；需要是 BaseOptions 的子类
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """
        测试时将模型切换为 eval 模式
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """
        测试时的前向函数。

        此函数在 no_grad() 下调用 <forward>，避免保存反向传播的中间步骤。
        同时调用 <compute_visuals> 生成额外的可视化结果。
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """
        计算用于 visdom 和 HTML 可视化的额外输出图像
        """
        pass

    def get_image_paths(self):
        """
        返回用于加载当前数据的图片路径
        """
        return self.image_paths

    def update_learning_rate(self):
        """
        更新所有网络的学习率；每个 epoch 结束时调用
        """
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """
        返回可视化图像。train.py 会用 visdom 显示这些图像，并保存到 HTML。
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """
        返回训练损失/误差。train.py 会在控制台打印这些误差，并保存到文件。
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) 兼容标量 tensor 和 float
        return errors_ret

    def save_networks(self, epoch):
        """
        保存所有网络到磁盘。

        参数:
            epoch (int) -- 当前 epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
        修复 InstanceNorm 检查点与 0.4 之前版本的不兼容问题
        """
        key = keys[i]
        if i + 1 == len(keys):  # 到达末尾，指向参数/缓冲区
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """
        从磁盘加载所有网络。

        参数:
            epoch (int) -- 当前 epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # 如果你使用的是 PyTorch 0.4 以上版本，可以去掉 str() 包裹
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复 0.4 之前版本的 InstanceNorm 检查点
                # for key in list(state_dict.keys()):  # 需要复制 keys，因为循环中会修改
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """
        打印网络的参数总数，以及（如果 verbose）网络结构

        参数:
            verbose (bool) -- 如果为 True，打印网络结构
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        设置所有网络的 requies_grad=False，以避免不必要的计算

        参数:
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 网络是否需要梯度
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}
