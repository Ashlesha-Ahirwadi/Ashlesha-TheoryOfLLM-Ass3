"""
Part 1.1: Dataset Preparation
Loads and preprocesses the Anthropic HH-RLHF dataset
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_dataset_distribution(dataset, output_dir):
    """Analyze and visualize dataset distribution and patterns"""
    
    print("\n--- Dataset Analysis ---")
    
    # Collect statistics
    stats = {
        "total_samples": len(dataset),
        "prompt_lengths": [],
        "chosen_lengths": [],
        "rejected_lengths": [],
    }
    
    for example in tqdm(dataset, desc="Analyzing dataset"):
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        
        stats["prompt_lengths"].append(len(prompt.split()))
        stats["chosen_lengths"].append(len(chosen.split()))
        stats["rejected_lengths"].append(len(rejected.split()))
    
    # Compute statistics
    stats_summary = {
        "total_samples": stats["total_samples"],
        "prompt_length": {
            "mean": float(np.mean(stats["prompt_lengths"])),
            "std": float(np.std(stats["prompt_lengths"])),
            "min": float(np.min(stats["prompt_lengths"])),
            "max": float(np.max(stats["prompt_lengths"])),
            "median": float(np.median(stats["prompt_lengths"]))
        },
        "chosen_length": {
            "mean": float(np.mean(stats["chosen_lengths"])),
            "std": float(np.std(stats["chosen_lengths"])),
            "min": float(np.min(stats["chosen_lengths"])),
            "max": float(np.max(stats["chosen_lengths"])),
            "median": float(np.median(stats["chosen_lengths"]))
        },
        "rejected_length": {
            "mean": float(np.mean(stats["rejected_lengths"])),
            "std": float(np.std(stats["rejected_lengths"])),
            "min": float(np.min(stats["rejected_lengths"])),
            "max": float(np.max(stats["rejected_lengths"])),
            "median": float(np.median(stats["rejected_lengths"]))
        }
    }
    
    # Print statistics
    print(f"\nTotal samples: {stats_summary['total_samples']}")
    print(f"\nPrompt length - Mean: {stats_summary['prompt_length']['mean']:.2f}, "
          f"Std: {stats_summary['prompt_length']['std']:.2f}, "
          f"Min: {stats_summary['prompt_length']['min']:.0f}, "
          f"Max: {stats_summary['prompt_length']['max']:.0f}")
    
    print(f"Chosen length - Mean: {stats_summary['chosen_length']['mean']:.2f}, "
          f"Std: {stats_summary['chosen_length']['std']:.2f}")
    
    print(f"Rejected length - Mean: {stats_summary['rejected_length']['mean']:.2f}, "
          f"Std: {stats_summary['rejected_length']['std']:.2f}")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(stats["prompt_lengths"], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title("Prompt Length Distribution")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(stats["chosen_lengths"], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title("Chosen Response Length Distribution")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    
    axes[2].hist(stats["rejected_lengths"], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_title("Rejected Response Length Distribution")
    axes[2].set_xlabel("Word Count")
    axes[2].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nDistribution plot saved to {output_dir}/dataset_distribution.png")
    
    return stats_summary

def preprocess_example(example, tokenizer, max_length=256):
    """Preprocess a single example"""
    
    # HH-RLHF format: chosen and rejected contain the full conversation
    # We need to extract the human prompt from the chosen response
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")
    
    # Extract prompt (everything before the first "Assistant:" if present)
    # Otherwise, use a portion of the chosen text as prompt
    if "\n\nAssistant:" in chosen:
        prompt = chosen.split("\n\nAssistant:")[0] + "\n\nAssistant:"
        chosen_response = chosen.split("\n\nAssistant:", 1)[1] if "\n\nAssistant:" in chosen else chosen
        rejected_response = rejected.split("\n\nAssistant:", 1)[1] if "\n\nAssistant:" in rejected else rejected
    else:
        # Fallback: use first 50 chars as prompt
        prompt = chosen[:50] if len(chosen) > 50 else chosen
        chosen_response = chosen
        rejected_response = rejected
    
    # Tokenize
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length // 2,
        padding=False,
        return_tensors=None
    )
    
    chosen_tokens = tokenizer(
        chosen,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    rejected_tokens = tokenizer(
        rejected,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "prompt_input_ids": prompt_tokens["input_ids"],
        "chosen_input_ids": chosen_tokens["input_ids"],
        "rejected_input_ids": rejected_tokens["input_ids"],
    }

def prepare_dataset(dataset_name="Anthropic/hh-rlhf", max_length=256, subset_size=None, output_dir="outputs"):
    """
    Load and prepare the HH-RLHF dataset
    
    Args:
        dataset_name: Name of the dataset to load
        max_length: Maximum sequence length
        subset_size: If specified, use only a subset of the data
        output_dir: Directory to save outputs
    
    Returns:
        train_dataset, eval_dataset, statistics
    """
    
    print(f"\nLoading dataset: {dataset_name}")
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
        train_data = dataset["train"]
        test_data = dataset["test"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative loading method...")
        # Alternative: load splits separately
        train_data = load_dataset(dataset_name, split="train")
        test_data = load_dataset(dataset_name, split="test")
    
    # Use subset if specified (for quick testing)
    if subset_size is not None:
        print(f"Using subset of {subset_size} samples")
        train_data = train_data.select(range(min(subset_size, len(train_data))))
        test_data = test_data.select(range(min(subset_size // 5, len(test_data))))
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Analyze dataset distribution
    dataset_stats = analyze_dataset_distribution(train_data, output_dir)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess datasets
    print("\nPreprocessing training data...")
    train_dataset = []
    for example in tqdm(train_data):
        try:
            # Skip examples with empty chosen/rejected
            if not example.get("chosen") or not example.get("rejected"):
                continue
            if len(example.get("chosen", "").strip()) == 0 or len(example.get("rejected", "").strip()) == 0:
                continue
            processed = preprocess_example(example, tokenizer, max_length)
            train_dataset.append(processed)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    print("Preprocessing test data...")
    eval_dataset = []
    for example in tqdm(test_data):
        try:
            # Skip examples with empty chosen/rejected
            if not example.get("chosen") or not example.get("rejected"):
                continue
            if len(example.get("chosen", "").strip()) == 0 or len(example.get("rejected", "").strip()) == 0:
                continue
            processed = preprocess_example(example, tokenizer, max_length)
            eval_dataset.append(processed)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    # Save statistics
    dataset_stats["train_samples"] = len(train_dataset)
    dataset_stats["eval_samples"] = len(eval_dataset)
    
    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Final train size: {len(train_dataset)}")
    print(f"Final eval size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, dataset_stats