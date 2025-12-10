"""
Experiment runner with configuration file support.
"""

import yaml
import argparse
import subprocess
import sys
import os


def load_config(config_path):
    """Load experiment configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training(config):
    """Run training script with config parameters."""
    train_cmd = [
        'python', 'scripts/train_lora_vit.py',
        '--output_dir', config['output']['checkpoint_dir'],
        '--epochs', str(config['training']['epochs']),
        '--batch_size', str(config['training']['batch_size']),
        '--lr', str(config['training']['learning_rate']),
        '--weight_decay', str(config['training']['weight_decay']),
        '--prior_precision', str(config['training']['prior_precision']),
        '--lora_r', str(config['lora']['r']),
        '--lora_alpha', str(config['lora']['alpha']),
        '--lora_dropout', str(config['lora']['dropout']),
        '--seed', str(config['training']['seed'])
    ]
    
    print("Running training with command:")
    print(" ".join(train_cmd))
    print()
    
    result = subprocess.run(train_cmd)
    return result.returncode


def run_evaluation(config):
    """Run evaluation script with config parameters."""
    checkpoint_path = os.path.join(
        config['output']['checkpoint_dir'],
        'model_map.pt'
    )
    
    eval_cmd = [
        'python', 'scripts/eval_bayesian_lora.py',
        '--checkpoint', checkpoint_path,
        '--output_dir', config['output']['results_dir'],
        '--batch_size', str(config['evaluation']['batch_size']),
        '--num_samples', str(config['evaluation']['num_samples']),
        '--prior_precision', str(config['training']['prior_precision']),
        '--laplace_type', config['evaluation']['laplace_type'],
        '--n_bins', str(config['evaluation']['n_bins']),
        '--seed', str(config['training']['seed']),
        '--lora_r', str(config['lora']['r']),
        '--lora_alpha', str(config['lora']['alpha'])
    ]
    
    print("Running evaluation with command:")
    print(" ".join(eval_cmd))
    print()
    
    result = subprocess.run(eval_cmd)
    return result.returncode


def run_analysis(config):
    """Run analysis notebook."""
    analysis_cmd = [
        'jupyter', 'nbconvert',
        '--execute',
        '--to', 'notebook',
        '--inplace',
        'notebooks/analysis.ipynb'
    ]
    
    print("Running analysis notebook:")
    print(" ".join(analysis_cmd))
    print()
    
    result = subprocess.run(analysis_cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run experiments with config file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['train', 'eval', 'analyze', 'all'],
                        help='Which stage to run')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    print("Configuration loaded successfully\n")
    
    # Run requested stages
    if args.stage in ['train', 'all']:
        print("="*70)
        print("STAGE 1: TRAINING")
        print("="*70)
        returncode = run_training(config)
        if returncode != 0:
            print(f"\nTraining failed with code {returncode}")
            sys.exit(returncode)
        print("\n✓ Training completed successfully\n")
    
    if args.stage in ['eval', 'all']:
        print("="*70)
        print("STAGE 2: EVALUATION")
        print("="*70)
        returncode = run_evaluation(config)
        if returncode != 0:
            print(f"\nEvaluation failed with code {returncode}")
            sys.exit(returncode)
        print("\n✓ Evaluation completed successfully\n")
    
    if args.stage in ['analyze', 'all']:
        print("="*70)
        print("STAGE 3: ANALYSIS")
        print("="*70)
        returncode = run_analysis(config)
        if returncode != 0:
            print(f"\nAnalysis failed with code {returncode}")
            sys.exit(returncode)
        print("\n✓ Analysis completed successfully\n")
    
    print("="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {config['output']['results_dir']}")


if __name__ == '__main__':
    main()
