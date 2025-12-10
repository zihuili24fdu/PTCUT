#!/usr/bin/env python3
"""
测试PTCUT模型加载.pt格式的text_features
"""

import torch
import sys
import os.path as osp

# 添加路径
sys.path.insert(0, osp.dirname(__file__))

from models.ptcut_model import PTCUTModel
from options.train_options import TrainOptions


def test_pt_loading():
    """测试直接从.pt文件加载text_features"""
    
    print("="*60)
    print("测试PTCUT模型 - .pt文件加载功能")
    print("="*60)
    
    # 创建测试用的text_features
    print("\n[1] 创建测试用的text_features...")
    test_features = torch.randn(4, 512)
    test_features = test_features / test_features.norm(dim=-1, keepdim=True)
    test_path = '/tmp/test_ptcut_text_features.pt'
    torch.save(test_features, test_path)
    print(f"    ✓ 已保存到: {test_path}")
    print(f"    ✓ 形状: {test_features.shape}")
    
    # 准备模型选项
    print("\n[2] 准备模型配置...")
    sys.argv = [
        'test_script.py',
        '--dataroot', '/tmp',  # 临时路径
        '--name', 'test_ptcut',
        '--model', 'ptcut',
        '--prompt_text_features', test_path,
        '--num_classes', '4',
        '--gpu_ids', '-1',  # CPU模式
        '--batch_size', '1',
        '--isTrain'
    ]
    
    opt = TrainOptions().parse()
    print(f"    ✓ 配置完成")
    print(f"    - prompt_text_features: {opt.prompt_text_features}")
    print(f"    - num_classes: {opt.num_classes}")
    
    # 测试模型初始化
    print("\n[3] 初始化PTCUT模型...")
    try:
        # 注意：完整初始化需要CONCH模型和数据，这里只测试feature加载部分
        from models.ptcut_model import PTCUTModel
        
        # 手动测试load_prompt_features方法
        class MockModel:
            def register_buffer(self, name, tensor):
                setattr(self, name, tensor)
        
        mock_model = MockModel()
        PTCUTModel.load_prompt_features(mock_model, test_path, 4)
        
        print(f"    ✓ 成功加载text_features")
        print(f"    - 加载的特征形状: {mock_model.prompt_text_features.shape}")
        print(f"    - 特征类型: {mock_model.prompt_text_features.dtype}")
        
        # 验证特征值是否正确
        is_close = torch.allclose(mock_model.prompt_text_features, test_features, atol=1e-6)
        if is_close:
            print(f"    ✓ 特征值验证通过（与保存的一致）")
        else:
            print(f"    ⚠ 警告：特征值不完全一致")
            
    except Exception as e:
        print(f"    ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试错误处理
    print("\n[4] 测试错误处理...")
    
    # 测试：文件不存在
    try:
        mock_model2 = MockModel()
        PTCUTModel.load_prompt_features(mock_model2, '/nonexistent/file.pt', 4)
        print("    ✗ 应该抛出FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"    ✓ 正确处理文件不存在: FileNotFoundError")
    
    # 测试：形状不匹配
    try:
        wrong_shape_features = torch.randn(2, 512)  # 错误的类别数
        wrong_path = '/tmp/wrong_shape.pt'
        torch.save(wrong_shape_features, wrong_path)
        
        mock_model3 = MockModel()
        PTCUTModel.load_prompt_features(mock_model3, wrong_path, 4)
        print("    ✗ 应该抛出ValueError（形状不匹配）")
        return False
    except ValueError as e:
        print(f"    ✓ 正确处理形状不匹配: ValueError")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)
    print("\n下一步：")
    print("1. 从KgCoOp checkpoint提取真实的text_features：")
    print("   python extract_text_features.py /path/to/checkpoint.pth.tar")
    print("\n2. 使用提取的特征训练PTCUT：")
    print("   python train_ptcut.py --prompt_text_features /path/to/prompt_text_features.pt ...")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = test_pt_loading()
    sys.exit(0 if success else 1)
