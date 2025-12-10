"""Test script for image-to-image translation.
Performs inference and saves images using multi-threaded acceleration.
For evaluation metrics, use evaluate.py separately.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    print(f"Model: {opt.name} | Dataset: {opt.dataroot} | Images: {opt.num_test}")
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    
    # Initialize model
    first_data = next(iter(dataset))
    model.data_dependent_initialize(first_data)
    model.setup(opt)
    model.parallelize()
    if opt.eval:
        model.eval()
    
    # Create output directory
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # Save images task for threads
    def save_images_task(webpage, visuals_cpu, img_path, width):
        save_images(webpage, visuals_cpu, img_path, width=width)
        return time.time()
    
    # Multi-threaded inference and saving
    print(f"Saving to: {web_dir}")
    print("Processing images (32 threads)...\n")
    
    inference_times = []
    start_time = time.time()
    
    executor = ThreadPoolExecutor(max_workers=32)
    futures = []
    
    try:
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            
            # Inference (main thread)
            t_infer = time.time()
            model.set_input(data)
            model.test()
            inference_times.append(time.time() - t_infer)
            
            # Get results and move to CPU
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            
            visuals_cpu = OrderedDict()
            for label, tensor in visuals.items():
                if isinstance(tensor, torch.Tensor):
                    visuals_cpu[label] = tensor.cpu()
                else:
                    visuals_cpu[label] = tensor
            
            # Submit save task to thread pool
            future = executor.submit(save_images_task, webpage, visuals_cpu, img_path, opt.display_winsize)
            futures.append(future)
            
            # Progress display
            if (i + 1) % 50 == 0 or i == opt.num_test - 1:
                elapsed = time.time() - start_time
                actual_speed = elapsed / (i + 1)  # Real speed (total time / images)
                avg_infer = np.mean(inference_times)
                eta = actual_speed * (opt.num_test - i - 1)
                
                completed = sum(1 for f in futures if f.done())
                print(f"Progress: {i+1}/{opt.num_test} | Saved: {completed}/{i+1} | "
                      f"Speed: {actual_speed:.3f}s/img (infer: {avg_infer:.3f}s) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        # Wait for all save tasks
        print("\nWaiting for save tasks to complete...")
        for future in futures:
            future.result()
        
    finally:
        executor.shutdown(wait=True)
    
    webpage.save()
    total_time = time.time() - start_time
    
    # Summary
    print(f"\nCompleted: {len(inference_times)} images in {total_time:.2f}s "
          f"({len(inference_times)/total_time:.2f} img/s)")
    print(f"Images saved to: {web_dir}")
    print(f"For evaluation: python evaluate.py --results_dir {web_dir}\n")

# python test.py --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 --results_dir ./datasets/ --name pix2pix --model pix2pix --eval --preprocess none --gpu_ids 0 --num_test 500

# python test.py --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 --results_dir ./datasets/ --name cut --model cut --eval --preprocess none --gpu_ids 0 --num_test 500
# python test.py --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 --results_dir ./datasets/ --name cyclegan --model cycle_gan --eval --preprocess none --gpu_ids 0 --num_test 500

# python test.py --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 --results_dir ./datasets/ --name TTC1 --model text_tuned_cut --eval --preprocess none --gpu_ids 1 --num_test 500