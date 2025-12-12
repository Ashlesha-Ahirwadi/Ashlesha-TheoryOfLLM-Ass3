"""
Part 2.1: PPO-based RLHF Implementation (Memory Optimized)
Implements Proximal Policy Optimization for language model alignment
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from src.reward_model import RewardModel
import gc

class PPODataset(Dataset):
    """Dataset for PPO training"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Tokenize prompt
        encoding = self.tokenizer(
            example["prompt"],
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": example["prompt"]
        }

class PPOTrainer:
    """Trainer for PPO-based RLHF"""
    
    def __init__(self, model_name, reward_model_path, output_dir, 
                 learning_rate=1e-6, batch_size=1, kl_coef=0.05, 
                 clip_ratio=0.2, entropy_coef=0.01, num_epochs=1, 
                 device="cuda", max_length=256):
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.device = device
        self.max_length = max_length
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Fix for decoder-only models
        
        # Initialize policy model
        print(f"Initializing policy model from {model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Initialize reference model (frozen)
        print(f"Initializing reference model")
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Load reward model
        print(f"Loading reward model from {reward_model_path}")
        self.reward_model = RewardModel(model_name).to(device)
        checkpoint = torch.load(f"{reward_model_path}/reward_model.pt", map_location=device)
        self.reward_model.load_state_dict(checkpoint["model_state_dict"])
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # Training metrics
        self.training_history = {
            "reward": [],
            "kl_divergence": [],
            "policy_loss": [],
            "entropy": [],
            "total_loss": []
        }
    
    def generate_responses(self, prompts, max_new_tokens=50):
        """Generate responses from policy model"""
        self.policy_model.eval()
        
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                # Skip empty prompts
                if not prompt or len(prompt.strip()) == 0:
                    responses.append("")
                    continue
                
                try:
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                    
                    # Skip if input is too short
                    if input_ids.shape[1] < 2:
                        responses.append("")
                        continue
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    output = self.policy_model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        min_length=input_ids.shape[1] + 5  # Ensure at least 5 new tokens
                    )
                    
                    response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response)
                    
                    # Clear cache after each generation
                    del output, input_ids
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\nGeneration failed for prompt (len={len(prompt)}). Debug info:")
                    print(f" prompt repr: {repr(prompt[:100])}")
                    if 'input_ids' in locals():
                        print(f" input_ids shape: {input_ids.shape}  sample ids: {input_ids[0][:10].tolist()}")
                    print(f" error: {e}")
                    responses.append("")
                    torch.cuda.empty_cache()
                    continue
        
        self.policy_model.train()
        return responses
    
    def compute_kl_divergence(self, policy_logits, ref_logits, attention_mask):
        """Compute KL divergence between policy and reference model"""
        # Use smaller chunks to avoid OOM
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute KL in chunks
        kl = torch.zeros(policy_logits.size(0), policy_logits.size(1), device=self.device)
        chunk_size = 100  # Process 100 tokens at a time
        
        for i in range(0, policy_logits.size(-1), chunk_size):
            end_idx = min(i + chunk_size, policy_logits.size(-1))
            chunk_kl = (torch.exp(policy_logprobs[:, :, i:end_idx]) * 
                       (policy_logprobs[:, :, i:end_idx] - ref_logprobs[:, :, i:end_idx])).sum(dim=-1)
            kl += chunk_kl
        
        # Mask padding tokens
        if attention_mask is not None:
            kl = kl * attention_mask
            kl = kl.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            kl = kl.mean(dim=1)
        
        return kl.mean()
    
    def compute_entropy(self, logits, attention_mask):
        """Compute entropy of policy distribution"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute entropy in chunks
        entropy = torch.zeros(logits.size(0), logits.size(1), device=self.device)
        chunk_size = 100
        
        for i in range(0, logits.size(-1), chunk_size):
            end_idx = min(i + chunk_size, logits.size(-1))
            chunk_entropy = -(probs[:, :, i:end_idx] * log_probs[:, :, i:end_idx]).sum(dim=-1)
            entropy += chunk_entropy
        
        # Mask padding tokens
        if attention_mask is not None:
            entropy = entropy * attention_mask
            entropy = entropy.sum(dim=1) / attention_mask.sum(dim=1)
        else:
            entropy = entropy.mean(dim=1)
        
        return entropy.mean()
    
    def compute_ppo_loss(self, input_ids, attention_mask, old_logprobs, advantages):
        """Compute PPO loss with clipped surrogate objective"""
        # Forward pass through policy model
        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute log probabilities
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_logprobs = torch.gather(
            logprobs, 
            2, 
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        mask = attention_mask[:, 1:]
        target_logprobs = target_logprobs * mask
        new_logprobs = target_logprobs.sum(dim=1) / mask.sum(dim=1)
        
        # Compute ratio
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Handle NaN/Inf in ratio
        ratio = torch.clamp(ratio, 0.01, 100.0)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Get reference logits for KL (with gradient accumulation)
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits[:, :-1, :]
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(logits[:, :-1, :], ref_logits, mask)
        
        # Compute entropy
        entropy = self.compute_entropy(logits[:, :-1, :], mask)
        
        # Total loss
        total_loss = policy_loss + self.kl_coef * kl_div - self.entropy_coef * entropy
        
        # Clean up
        del outputs, logits, ref_outputs, ref_logits
        torch.cuda.empty_cache()
        
        return total_loss, policy_loss, kl_div, entropy
    
    def train_step(self, batch):
        """Single training step"""
        prompts = batch["prompt"]
        
        # Generate responses
        responses = self.generate_responses(prompts, max_new_tokens=50)
        
        # Create full sequences
        full_texts = [p + r for p, r in zip(prompts, responses)]
        encodings = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        full_input_ids = encodings["input_ids"].to(self.device)
        full_attention_mask = encodings["attention_mask"].to(self.device)
        
        # Compute rewards
        with torch.no_grad():
            rewards = self.reward_model(full_input_ids, full_attention_mask)
        
        # Compute old log probabilities
        with torch.no_grad():
            outputs = self.policy_model(input_ids=full_input_ids, attention_mask=full_attention_mask)
            logits = outputs.logits
            logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target_logprobs = torch.gather(
                logprobs, 
                2, 
                full_input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            mask = full_attention_mask[:, 1:]
            target_logprobs = target_logprobs * mask
            old_logprobs = target_logprobs.sum(dim=1) / mask.sum(dim=1)
            
            # Clean up
            del outputs, logits
            torch.cuda.empty_cache()
        
        # Advantages (using rewards as advantages for simplicity)
        advantages = rewards - rewards.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = advantages / (adv_std + 1e-8)
        else:
            advantages = advantages * 0.0  # If std is 0, set advantages to 0
        
        # Compute PPO loss
        total_loss, policy_loss, kl_div, entropy = self.compute_ppo_loss(
            full_input_ids, full_attention_mask, old_logprobs, advantages
        )
        
        metrics = {
            "reward": rewards.mean().item(),
            "kl_divergence": kl_div.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item()
        }
        
        # Clean up
        del full_input_ids, full_attention_mask, rewards, old_logprobs, advantages
        torch.cuda.empty_cache()
        
        return total_loss, metrics
    
    def train(self, data_subset_size=None):
        """Main training loop"""
        
        # Load data
        from src.data_preparation import prepare_dataset
        train_data, eval_data, _ = prepare_dataset(
            subset_size=data_subset_size,
            max_length=self.max_length
        )
        
        # Create dataset
        train_dataset = PPODataset(train_data, self.tokenizer, max_length=self.max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.learning_rate)
        num_training_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_metrics = {key: [] for key in ["reward", "kl_divergence", "policy_loss", "entropy", "total_loss"]}
            
            progress_bar = tqdm(train_loader, desc="Training PPO")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                try:
                    # Training step
                    total_loss, metrics = self.train_step(batch)
                    
                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Update metrics
                    for key, value in metrics.items():
                        epoch_metrics[key].append(value)
                    
                    progress_bar.set_postfix({
                        "reward": f"{metrics['reward']:.4f}",
                        "kl": f"{metrics['kl_divergence']:.4f}",
                        "loss": f"{metrics['total_loss']:.4f}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM at batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
                
                # Periodic cleanup
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Store epoch metrics
            for key in epoch_metrics:
                if len(epoch_metrics[key]) > 0:
                    avg_value = np.mean(epoch_metrics[key])
                    self.training_history[key].append(avg_value)
                    print(f"{key}: {avg_value:.4f}")
            
            # Save checkpoint
            self.save_model(epoch)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save training history
        with open(f"{self.output_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.policy_model, self.training_history
    
    def save_model(self, epoch=None):
        """Save model checkpoint"""
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        save_path = f"{self.output_dir}{suffix}"
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot training metrics"""
        if not any(self.training_history.values()):
            print("No training history to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Reward
        if self.training_history["reward"]:
            axes[0, 0].plot(self.training_history["reward"])
            axes[0, 0].set_title("Average Reward")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)
        
        # KL Divergence
        if self.training_history["kl_divergence"]:
            axes[0, 1].plot(self.training_history["kl_divergence"])
            axes[0, 1].set_title("KL Divergence")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("KL")
            axes[0, 1].grid(True)
        
        # Policy Loss
        if self.training_history["policy_loss"]:
            axes[1, 0].plot(self.training_history["policy_loss"])
            axes[1, 0].set_title("Policy Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True)
        
        # Entropy
        if self.training_history["entropy"]:
            axes[1, 1].plot(self.training_history["entropy"])
            axes[1, 1].set_title("Entropy")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Entropy")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves.png")