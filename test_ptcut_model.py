#!/usr/bin/env python3
"""
PTCUT模型测试脚本
测试模型初始化、前向传播和损失计算
"""

import sys
import torch
import os.path as osp
sys.path.insert(0, '/home/lzh/myCode/awesome-virtual-staining')

from options.train_options import TrainOptions
from models import create_model

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 80)
    print("测试1: PTCUT模型初始化")
    print("=" * 80)
    
    # 设置命令行参数
    sys.argv = [
        'test',
        '--dataroot', '/home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512',
        '--name', 'test_ptcut',
        '--model', 'ptcut',
        '--gpu_ids', '0',  # 使用GPU 0
        '--batch_size', '1',
        '--input_nc', '3',
        '--output_nc', '3',
        '--load_size', '512',
        '--crop_size', '512',
        '--prompt_checkpoint', '/home/lzh/myCode/myKgCoOp/myKgCoOp/output/gnb_coop_conch_csc_fulldata_4class/prompt_learner/model.pth.tar-50',
        '--num_classes', '4',
        '--lambda_cls', '0.5',
        '--lambda_distill', '0.5',
        '--preprocess', 'none',  # 不进行额外的预处理
    ]
    
    try:
        opt = TrainOptions().parse()
        model = create_model(opt)
        print("✓ PTCUT模型创建成功")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  损失名称: {model.loss_names}")
        print(f"  可视化名称: {model.visual_names}")
        print(f"  模型名称: {model.model_names}")
        return True, model, opt
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_forward_pass(model, opt):
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("测试2: 前向传播")
    print("=" * 80)
    
    # 创建虚拟数据
    batch_size = 2
    data = {
        'A': torch.randn(batch_size, 3, 512, 512).cuda() if len(opt.gpu_ids) > 0 else torch.randn(batch_size, 3, 512, 512),
        'B': torch.randn(batch_size, 3, 512, 512).cuda() if len(opt.gpu_ids) > 0 else torch.randn(batch_size, 3, 512, 512),
        'A_paths': ['/test/image_1.png', '/test/image_2.png'],
        'B_paths': ['/test/image_1.png', '/test/image_2.png']
    }
    
    try:
        model.set_input(data)
        model.forward()
        print("✓ 前向传播成功")
        print(f"  生成的fakeB shape: {model.fake_B.shape}")
        print(f"  真实的realA shape: {model.real_A.shape}")
        print(f"  真实的realB shape: {model.real_B.shape}")
        if hasattr(model, 'labels'):
            print(f"  提取的标签: {model.labels}")
        return True, data
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_loss_computation(model, data):
    """测试损失计算"""
    print("\n" + "=" * 80)
    print("测试3: 损失计算")
    print("=" * 80)
    
    # 首先进行data_dependent_initialize
    try:
        print("正在进行data_dependent_initialize...")
        model.data_dependent_initialize(data)
        print("✓ data_dependent_initialize完成")
    except Exception as e:
        print(f"✗ data_dependent_initialize失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 计算损失
    try:
        model.set_input(data)
        model.forward()
        loss_G = model.compute_G_loss()
        losses = model.get_current_losses()
        
        print("✓ 损失计算成功")
        print(f"  总生成器损失: {loss_G.item():.4f}")
        print("  各项损失:")
        for name, value in losses.items():
            print(f"    {name}: {value:.4f}")
        
        # 检查必需的损失
        required_losses = ['G_GAN', 'NCE', 'CLS', 'DISTILL']
        missing_losses = [l for l in required_losses if l not in losses]
        if missing_losses:
            print(f"⚠ 警告: 缺少以下损失: {missing_losses}")
        else:
            print("✓ 所有必需的损失都存在")
        
        return True
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_step(model, data):
    """测试优化步骤"""
    print("\n" + "=" * 80)
    print("测试4: 优化步骤")
    print("=" * 80)
    
    try:
        model.set_input(data)
        model.optimize_parameters()
        print("✓ 优化步骤成功")
        
        # 获取损失
        losses = model.get_current_losses()
        print("  优化后的损失:")
        for name, value in losses.items():
            print(f"    {name}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 优化步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("开始PTCUT模型测试...\n")
    
    # 测试1: 模型初始化
    success, model, opt = test_model_initialization()
    if not success:
        print("\n模型初始化失败，终止测试")
        sys.exit(1)
    
    # 测试2: 前向传播
    success, data = test_forward_pass(model, opt)
    if not success:
        print("\n前向传播失败，终止测试")
        sys.exit(1)
    
    # 测试3: 损失计算
    success = test_loss_computation(model, data)
    if not success:
        print("\n损失计算失败，终止测试")
        sys.exit(1)
    
    # 测试4: 优化步骤
    success = test_optimization_step(model, data)
    if not success:
        print("\n优化步骤失败，终止测试")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("所有测试通过！ ✓")
    print("=" * 80)
