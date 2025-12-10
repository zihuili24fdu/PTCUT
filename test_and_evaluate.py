"""Integrated test and evaluation script for virtual staining models.
Performs inference and immediately evaluates the results in a single run.
"""
import os
import sys
import argparse
import subprocess
import time
import json

def run_command(cmd, description):
    """Run a command and display its output"""
    print(f"\n{'='*80}")
    print(f"🔄 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with error code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed in {elapsed:.2f}s")
    return True

def print_summary(results_dir):
    """Print evaluation summary if available"""
    stats_file = os.path.join(results_dir, 'evaluation', 'statistics.json')
    if os.path.exists(stats_file):
        print(f"\n{'='*80}")
        print("📊 EVALUATION RESULTS")
        print(f"{'='*80}\n")
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"Total images evaluated: {stats['count']}")
        print()
        
        for metric in ['psnr', 'ssim', 'pearson']:
            m = stats['metrics'][metric]
            print(f"{metric.upper():8} - Mean: {m['mean']:7.4f}  Std: {m['std']:7.4f}  "
                  f"Min: {m['min']:7.4f}  Max: {m['max']:7.4f}")
        
        if 'fid' in stats:
            print(f"{'FID':8} - Score: {stats['fid']:.4f}  (lower is better)")
        
        print(f"\n{'='*80}")
        print(f"📁 Results saved to: {results_dir}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Combined test and evaluation for virtual staining models')
    
    # Test (inference) arguments
    test_group = parser.add_argument_group('Testing Arguments')
    test_group.add_argument('--dataroot', type=str, required=True,
                           help='Path to test dataset')
    test_group.add_argument('--name', type=str, required=True,
                           help='Model name (experiment name)')
    test_group.add_argument('--model', type=str, required=True,
                           choices=['pix2pix', 'cut', 'cycle_gan', 'text_tuned_cut'],
                           help='Model type')
    test_group.add_argument('--results_dir', type=str, default='./results',
                           help='Directory to save results')
    test_group.add_argument('--epoch', type=str, default='latest',
                           help='Which epoch to load')
    test_group.add_argument('--num_test', type=int, default=500,
                           help='Number of test images')
    test_group.add_argument('--gpu_ids', type=str, default='0',
                           help='GPU ids (e.g., 0 or 0,1,2,3)')
    test_group.add_argument('--preprocess', type=str, default='none',
                           choices=['none', 'resize_and_crop', 'crop', 'scale_width'],
                           help='Preprocessing method')
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Arguments')
    eval_group.add_argument('--calculate_fid', action='store_true',
                           help='Calculate FID score (requires GPU, slower)')
    eval_group.add_argument('--fid_batch_size', type=int, default=32,
                           help='Batch size for FID calculation')
    eval_group.add_argument('--num_workers', type=int, default=None,
                           help='Number of workers for evaluation')
    
    # Control arguments
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip testing phase (only evaluate existing results)')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation phase (only run inference)')
    
    args = parser.parse_args()
    
    # Determine results directory
    web_dir = os.path.join(args.results_dir, args.name, f'test_{args.epoch}')
    
    print(f"\n{'='*80}")
    print("🚀 VIRTUAL STAINING MODEL - TEST & EVALUATE")
    print(f"{'='*80}")
    print(f"Model:       {args.name}")
    print(f"Type:        {args.model}")
    print(f"Dataset:     {args.dataroot}")
    print(f"Output:      {web_dir}")
    print(f"GPU:         {args.gpu_ids}")
    print(f"Images:      {args.num_test}")
    print(f"Calculate FID: {'Yes' if args.calculate_fid else 'No'}")
    print(f"{'='*80}")
    
    success = True
    
    # Phase 1: Testing (Inference)
    if not args.skip_test:
        test_cmd = [
            'python', 'test.py',
            '--dataroot', args.dataroot,
            '--results_dir', args.results_dir,
            '--name', args.name,
            '--model', args.model,
            '--epoch', args.epoch,
            '--num_test', str(args.num_test),
            '--gpu_ids', args.gpu_ids,
            '--preprocess', args.preprocess,
            '--eval'
        ]
        
        success = run_command(test_cmd, "Phase 1: Testing (Inference)")
        if not success:
            print("\n❌ Testing failed. Exiting.")
            return 1
    else:
        print(f"\n⏭️  Skipping testing phase (using existing results in {web_dir})")
    
    # Phase 2: Evaluation
    if not args.skip_eval and success:
        eval_cmd = [
            'python', 'evaluate.py',
            '--results_dir', web_dir
        ]
        
        if args.calculate_fid:
            eval_cmd.append('--calculate_fid')
            eval_cmd.extend(['--fid_batch_size', str(args.fid_batch_size)])
        
        if args.num_workers is not None:
            eval_cmd.extend(['--num_workers', str(args.num_workers)])
        
        success = run_command(eval_cmd, "Phase 2: Evaluation")
        if not success:
            print("\n❌ Evaluation failed.")
            return 1
    else:
        print("\n⏭️  Skipping evaluation phase")
    
    # Print summary
    if not args.skip_eval and success:
        print_summary(web_dir)
    
    print("\n✨ All phases completed successfully!\n")
    return 0

if __name__ == '__main__':
    sys.exit(main())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Basic usage (test + evaluate, no FID):
# python test_and_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0

# Full evaluation with FID:
# python test_and_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --calculate_fid

# Evaluate existing results only:
# python test_and_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --skip_test \
#   --calculate_fid

# Test only (no evaluation):
# python test_and_evaluate.py \
#   --dataroot /path/to/dataset \
#   --name pix2pix \
#   --model pix2pix \
#   --gpu_ids 0 \
#   --skip_eval

# Custom configuration:
# python test_and_evaluate.py \
#   --dataroot /home/lzh/myCode/virtual_stain_dataset/mydataset/dataset_level1 \
#   --results_dir ./my_results \
#   --name my_pix2pix_exp \
#   --model pix2pix \
#   --epoch 200 \
#   --num_test 1000 \
#   --gpu_ids 0 \
#   --calculate_fid \
#   --fid_batch_size 64 \
#   --num_workers 32
