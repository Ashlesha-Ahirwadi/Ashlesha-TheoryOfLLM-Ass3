"""
Main training script for RLHF Assignment
Implements PPO, GRPO, and DPO for language model alignment
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.data_preparation import prepare_dataset
from src.reward_model import RewardModelTrainer
from src.ppo_trainer import PPOTrainer
from src.grpo_trainer import GRPOTrainer
from src.dpo_trainer import DPOTrainer
from src.evaluation import ModelEvaluator
from src.utils import set_seed, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="RLHF Assignment Training")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["prepare_data", "train_reward", "train_ppo", "train_grpo", "train_dpo", "evaluate", "all"],
                        help="Training mode")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf", help="Dataset name")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--data_subset_size", type=int, default=None, help="Use subset for quick testing")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name")
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to trained reward model")
    parser.add_argument("--policy_model_path", type=str, default=None, help="Path to policy model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # PPO/GRPO specific arguments
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL divergence coefficient")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size")
    
    # DPO specific arguments
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter")
    
    # Evaluation arguments
    parser.add_argument("--eval_prompts", type=int, default=100, help="Number of evaluation prompts")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU if available")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    
    print(f"Using device: {device}")
    print(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/models", exist_ok=True)
    os.makedirs(f"{args.output_dir}/results", exist_ok=True)
    os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
    
    # Mode: Prepare Data
    if args.mode in ["prepare_data", "all"]:
        print("\n" + "="*50)
        print("PART 1.1: Dataset Preparation")
        print("="*50)
        
        train_dataset, eval_dataset, dataset_stats = prepare_dataset(
            dataset_name=args.dataset_name,
            max_length=args.max_length,
            subset_size=args.data_subset_size,
            output_dir=args.output_dir
        )
        
        print(f"\nDataset statistics saved to {args.output_dir}/dataset_stats.json")
        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Mode: Train Reward Model
    if args.mode in ["train_reward", "all"]:
        print("\n" + "="*50)
        print("PART 1.2: Reward Model Training")
        print("="*50)
        
        reward_trainer = RewardModelTrainer(
            model_name=args.base_model,
            output_dir=f"{args.output_dir}/models/reward_model",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=device,
            max_length=args.max_length
        )
        
        reward_model, reward_metrics = reward_trainer.train(data_subset_size=args.data_subset_size)
        
        print(f"\nReward model saved to {args.output_dir}/models/reward_model")
        print(f"Validation accuracy: {reward_metrics['val_accuracy']:.4f}")
    
    # Mode: Train PPO
    if args.mode in ["train_ppo", "all"]:
        print("\n" + "="*50)
        print("PART 2.1: PPO Training")
        print("="*50)
        
        if args.reward_model_path is None:
            args.reward_model_path = f"{args.output_dir}/models/reward_model"
        
        ppo_trainer = PPOTrainer(
            model_name=args.base_model,
            reward_model_path=args.reward_model_path,
            output_dir=f"{args.output_dir}/models/ppo_model",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            kl_coef=args.kl_coef,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            num_epochs=args.num_epochs,
            device=device,
            max_length=args.max_length
        )
        
        ppo_model, ppo_metrics = ppo_trainer.train(data_subset_size=args.data_subset_size)
        
        print(f"\nPPO model saved to {args.output_dir}/models/ppo_model")
    
    # Mode: Train GRPO
    if args.mode in ["train_grpo", "all"]:
        print("\n" + "="*50)
        print("PART 2.2: GRPO Training")
        print("="*50)
        
        if args.reward_model_path is None:
            args.reward_model_path = f"{args.output_dir}/models/reward_model"
        
        grpo_trainer = GRPOTrainer(
            model_name=args.base_model,
            reward_model_path=args.reward_model_path,
            output_dir=f"{args.output_dir}/models/grpo_model",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            kl_coef=args.kl_coef,
            group_size=args.group_size,
            entropy_coef=args.entropy_coef,
            num_epochs=args.num_epochs,
            device=device,
            max_length=args.max_length
        )
        
        grpo_model, grpo_metrics = grpo_trainer.train(data_subset_size=args.data_subset_size)
        
        print(f"\nGRPO model saved to {args.output_dir}/models/grpo_model")
    
    # Mode: Train DPO
    if args.mode in ["train_dpo", "all"]:
        print("\n" + "="*50)
        print("PART 3: DPO Training")
        print("="*50)
        
        dpo_trainer = DPOTrainer(
            model_name=args.base_model,
            output_dir=f"{args.output_dir}/models/dpo_model",
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            beta=args.dpo_beta,
            num_epochs=args.num_epochs,
            device=device,
            max_length=args.max_length
        )
        
        dpo_model, dpo_metrics = dpo_trainer.train(data_subset_size=args.data_subset_size)
        
        print(f"\nDPO model saved to {args.output_dir}/models/dpo_model")
    
    # Mode: Evaluate
    if args.mode in ["evaluate", "all"]:
        print("\n" + "="*50)
        print("PART 4: Evaluation")
        print("="*50)
        
        evaluator = ModelEvaluator(
            base_model_name=args.base_model,
            reward_model_path=f"{args.output_dir}/models/reward_model",
            output_dir=f"{args.output_dir}/results",
            device=device
        )
        
        models_to_evaluate = {
            "base": args.base_model,
            "ppo": f"{args.output_dir}/models/ppo_model",
            "grpo": f"{args.output_dir}/models/grpo_model",
            "dpo": f"{args.output_dir}/models/dpo_model"
        }
        
        results = evaluator.evaluate_all_models(
            models_to_evaluate,
            num_prompts=args.eval_prompts
        )
        
        print(f"\nEvaluation results saved to {args.output_dir}/results")
        print("\nSummary:")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Training and evaluation complete!")
    print("="*50)

if __name__ == "__main__":
    main()