from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """该类包含训练选项。

    同时包含在 BaseOptions 中定义的共享选项。
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom 和 HTML 可视化参数
        parser.add_argument('--display_freq', type=int, default=400, help='在屏幕上显示训练结果的频率')
        parser.add_argument('--display_ncols', type=int, default=4, help='若为正数，则在单个 visdom 网页面板中每行显示指定数量的图片')
        parser.add_argument('--display_id', type=int, default=None, help='网页显示的窗口 id，默认随机窗口 id')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom 网页显示的服务器地址')
        parser.add_argument('--display_env', type=str, default='main', help='visdom 显示环境名称（默认 "main"）')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom 网页显示的端口')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='保存训练结果到 html 的频率')
        parser.add_argument('--print_freq', type=int, default=100, help='在控制台显示训练结果的频率')
        parser.add_argument('--no_html', action='store_true', help='不保存中间训练结果到 [opt.checkpoints_dir]/[opt.name]/web/')
        # 网络保存和加载参数
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='保存最新结果的频率')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='每隔多少个 epoch 保存一次模型')
        parser.add_argument('--evaluation_freq', type=int, default=5000, help='评估频率')
        parser.add_argument('--save_by_iter', action='store_true', help='是否按迭代次数保存模型')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：加载最新的模型')
        parser.add_argument('--epoch_count', type=int, default=1, help='起始 epoch 计数，模型将按 <epoch_count>、<epoch_count>+<save_latest_freq> 等保存')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test 等')
        parser.add_argument('--pretrained_name', type=str, default=None, help='从其他 checkpoint 恢复训练')

        # 训练参数
        parser.add_argument('--n_epochs', type=int, default=200, help='初始学习率下的训练 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=0, help='线性衰减学习率到 0 的 epoch 数')
        parser.add_argument('--beta1', type=float, default=0.5, help='adam 优化器的 beta1 参数')
        parser.add_argument('--beta2', type=float, default=0.999, help='adam 优化器的 beta2 参数')
        parser.add_argument('--lr', type=float, default=0.0002, help='adam 优化器的初始学习率')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN 目标类型。 [vanilla| lsgan | wgangp]。vanilla 为原始 GAN 论文中的交叉熵损失。')
        parser.add_argument('--pool_size', type=int, default=50, help='用于存储先前生成图片的 buffer 大小')
        parser.add_argument('--lr_policy', type=str, default='linear', help='学习率调整策略。 [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='每隔 lr_decay_iters 迭代乘以一次 gamma')

        self.isTrain = True
        return parser
