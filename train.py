#!/usr/bin/env python3
"""
文本调优CUT模型训练脚本

该脚本用于训练带文本调优功能的虚拟染色模型，结合：
1. CUT模型的无配对图像转换能力
2. CONCH模型的文本-图像对齐损失
3. 病理组织HE图像的文本描述引导

使用示例:
python train_text_tuned_cut.py \
    --dataroot /path/to/dataset \
    --text_descriptions_file pathology_descriptions.txt \
    --name experiment_name \
    --model text_tuned_cut \
    --lambda_text 0.5 \
    --lambda_feat 0.5 \
    --gpu_ids 0,1 \
    --batch_size 2 \
    --n_epochs 100
"""

import time
import torch
from datetime import datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"数据集创建完成，训练图像数量: {dataset_size}\n")
    
    model = create_model(opt)  # 自动加载text_tuned_cut模型
    visualizer = Visualizer(opt)  # 用于损失可视化和结果展示
    opt.visualizer = visualizer  # 将visualizer传递给opt，便于模型访问
    total_iters = 0
    optimize_time = 0.1  # 初始化优化时间（用于平滑移动平均）
    
    # 4. 开始训练循环
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} 开始")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        
        dataset.set_epoch(epoch)
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            # 动态获取batch_size（处理最后一个batch可能不足的情况）
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            
            # GPU同步（确保准确的时间测量）
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_start_time = time.time()
            
            # 首次迭代：初始化模型（如PatchNCE的网络层）
            if epoch == opt.epoch_count and i == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始模型初始化（首次迭代）")
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 模型初始化完成")
            
            # 核心训练步骤：前向传播 + 反向传播 + 参数更新
            model.set_input(data)
            model.optimize_parameters()  # 计算损失并更新G和D
            
            # GPU同步（确保优化操作完成）
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            # 计算优化时间（使用平滑移动平均，减少波动）
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
            
            # 定期显示可视化结果（生成的图像）
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            # 定期打印和绘制损失曲线
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()  # 包含G_GAN, NCE, TEXT, FEAT等
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            # 定期保存模型检查点
            if total_iters % opt.save_latest_freq == 0:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] 保存最新模型 (epoch {epoch}, total_iters {total_iters})')
                print(f'实验名称: {opt.name}')
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()
        
        # Epoch结束：保存模型
        if epoch % opt.save_epoch_freq == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存 epoch {epoch} 模型")
            model.save_networks('latest')
            model.save_networks(epoch)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch模型保存完成")
        
        # 打印epoch统计信息并更新学习率
        epoch_time = time.time() - epoch_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} 完成, 耗时: {epoch_time:.0f}秒")
        model.update_learning_rate()  # 根据lr_policy调整学习率 

#cut 实验
#python train.py --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512 --name gnb_cut --model cut --load_size 512 --crop_size 512 --gpu_ids 1 --batch_size 2 --n_epochs 30 --continue_train --epoch_count 20
#cyclegan 实验
#python train.py --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512 --name gnb_cyclegan --model cycle_gan --load_size 512 --crop_size 512 --gpu_ids 2 --batch_size 1 --n_epochs 30 --continue_train --epoch_count 10

#pix2pix 实验
#python train.py --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512 --name gnb_pix2pix --model pix2pix --load_size 512 --crop_size 512 --gpu_ids 3 --batch_size 2 --n_epochs 30 --continue_train --epoch_count 20

#继续训练
#python train.py --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 --name pix2pix --model pix2pix --load_size 1024 --crop_size 512 --gpu_ids 3 --batch_size 8 --n_epochs 50 --n_epochs_decay 50 --use_separate_folders --continue_train --epoch_count 30