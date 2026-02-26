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

import copy
import os
import time
import torch
from datetime import datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from fast_evaluate import evaluate_model_in_memory

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"数据集创建完成，训练图像数量: {dataset_size}\n")

    opt_val = copy.deepcopy(opt)
    opt_val.isTrain = False
    opt_val.phase = 'test'
    opt_val.serial_batches = True
    opt_val.batch_size = max(1, len(opt.gpu_ids))  # 多GPU时确保每张卡至少分到1张图，避免 DataParallel 崩溃
    val_dataset = create_dataset(opt_val)
    print(f"验证/测试数据集创建完成，图像数量: {len(val_dataset)}\n")

    model = create_model(opt)  # 自动加载text_tuned_cut模型
    visualizer = Visualizer(opt)  # 用于损失可视化和结果展示
    opt.visualizer = visualizer  # 将visualizer传递给opt，便于模型访问
    total_iters = 0
    optimize_time = 0.1  # 初始化优化时间（用于平滑移动平均）

    # 在进入 epoch 循环之前执行模型初始化（避免在循环内初始化导致预训练权重被覆盖）
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始模型初始化...")
    dummy_data = next(iter(dataset))
    model.data_dependent_initialize(dummy_data)
    model.setup(opt)
    model.parallelize()
    print("模型初始化完成。\n")

    # 4. 开始训练循环
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch} 开始")
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(dataset, 'set_epoch'):  # 仅 DDP DistributedSampler 场景下存在此方法
            dataset.set_epoch(epoch)
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time  # 始终计算，避免 NameError

            # 动态获取batch_size（处理最后一个batch可能不足的情况）
            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            optimize_start_time = time.time()

            # 核心训练步骤：前向传播 + 反向传播 + 参数更新
            model.set_input(data)
            model.optimize_parameters()  # 计算损失并更新G和D

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

        if epoch % 5 == 0:
            print(f"\n{'=' * 80}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始 Epoch {epoch} 快速评估...")
            try:
                eval_results_str = evaluate_model_in_memory(
                    model=model,
                    val_dataset=val_dataset,
                    num_test=1000,
                    compute_fid=True
                )
                print(eval_results_str)

                log_file_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'\n{"=" * 80}\n')
                    log_file.write(f'Evaluation at Epoch {epoch} ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})\n')
                    log_file.write(f'{"=" * 80}\n')
                    log_file.write(eval_results_str)
                    log_file.write(f'\n{"=" * 80}\n\n')
            except Exception as e:
                print(f"\n❌ 评估过程出错: {e}")
            print(f"{'=' * 80}\n")

