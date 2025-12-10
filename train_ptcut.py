#!/usr/bin/env python3
"""
PTCUT模型训练脚本

使用示例:
python train_ptcut.py \
    --dataroot /home/lzh/myCode/myKgCoOp/myKgCoOp/data/GNB4Class_512 \
    --name gnb_ptcut \
    --model ptcut \
    --prompt_checkpoint /path/to/kgcoop/output/prompt_learner/model.pth.tar-50 \
    --load_size 512 \
    --crop_size 512 \
    --gpu_ids 0 \
    --batch_size 2 \
    --n_epochs 100 \
    --lambda_cls 0.5 \
    --lambda_distill 0.5 \
    --lambda_GAN 1.0 \
    --lambda_NCE 1.0
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
    
    model = create_model(opt)  # 自动加载ptcut模型
    visualizer = Visualizer(opt)  # 用于损失可视化和结果展示
    opt.visualizer = visualizer
    total_iters = 0
    optimize_time = 0.1
    
    # 开始训练循环
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
            
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_start_time = time.time()
            
            # 首次迭代：初始化模型
            if epoch == opt.epoch_count and i == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始模型初始化（首次迭代）")
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 模型初始化完成")
            
            # 核心训练步骤
            model.set_input(data)
            model.optimize_parameters()
            
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
            
            # 定期显示可视化结果
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            # 定期打印损失
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            # 定期保存模型
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
        
        # 打印epoch统计
        epoch_time = time.time() - epoch_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} 完成, 耗时: {epoch_time:.0f}秒")
        model.update_learning_rate()
