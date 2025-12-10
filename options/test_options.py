from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """该类包含测试选项。

    它还包含在 BaseOptions 中定义的共享选项。
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # 定义共享选项
        parser.add_argument('--results_dir', type=str, default='./results/', help='将结果保存到这里。')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test 等')
        # Dropout 和 Batchnorm 在训练和测试期间有不同的行为。
        parser.add_argument('--eval', action='store_true', help='在测试时使用 eval 模式。')
        parser.add_argument('--num_test', type=int, default=50, help='要运行的测试图片数量')
        parser.add_argument('--test_mode', type=str, default='evaluate', 
                          choices=['evaluate', 'generate', 'both'],
                          help='测试模式: evaluate(仅评估指标), generate(仅生成图像), both(评估+生成)')

        # 为了避免裁剪，load_size 应该与 crop_size 相同
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
