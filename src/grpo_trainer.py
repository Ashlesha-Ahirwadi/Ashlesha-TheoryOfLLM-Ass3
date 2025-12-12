"""
Part 2.2: GRPO Implementation (Memory Optimized)
Implements Group Relative Policy Optimization
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
import time
from src.reward_model import RewardModel
import gc

class GRPODataset(Dataset):
    """Dataset for GRPO training"""
    
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

class GRPOTrainer:
    """Trainer for GRPO"""
    
    def __init__(self, model_name, reward_model_path, output_dir, 
                 learning_rate=1e-6, batch_size=1, kl_coef=0.05, 
                 group_size=2, entropy_coef=0.01, num_epochs=1, 
                 device="cuda", max_length=256):
        self.model_name = model_name
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.device = device
        self.max_length = max_length
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
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
            "total_loss": [],
            "time_per_iteration": []
        }
    
    def generate_group_responses(self, prompt, group_size, max_new_tokens=50):
        """Generate multiple responses for a single prompt"""
        self.policy_model.eval()
        
        responses = []
        with torch.no_grad():
            # Skip empty prompts
            if not prompt or len(prompt.strip()) == 0:
                return [""] * group_size
            
            try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Skip if input is too short
                if input_ids.shape[1] < 2:
                    return [""] * group_size
                
                for _ in range(group_size):
                    torch.cuda.empty_cache()
                    
                    output = self.policy_model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        min_length=input_ids.shape[1] + 5
                    )
                    
                    response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response)
                    
                    del output
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nGroup generation failed for prompt (len={len(prompt)}). Error: {e}")
                return [""] * group_size
        
        self.policy_model.train()
        return responses
    
    def compute_group_advantages(self, rewards):
        """Compute advantages relative to group mean"""
        # Reshape to [batch_size, group_size]
        group_rewards = rewards.view(-1, self.group_size)
        
        # Compute mean reward per group
        group_means = group_rewards.mean(dim=1, keepdim=True)
        
        # Advantages are deviations from group mean
        advantages = group_rewards - group_means
        
        # Normalize advantages
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = advantages / (adv_std + 1e-8)
        else:
            advantages = advantages * 0.0
        
        return advantages.view(-1)
    
    def compute_kl_divergence(self, policy_logits, ref_logits, attention_mask):
        """Compute KL divergence between policy and reference model"""
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute KL in chunks
        kl = torch.zeros(policy_logits.size(0), policy_logits.size(1), device=self.device)
        chunk_size = 100
        
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
    
    def compute_grpo_loss(self, input_ids, attention_mask, advantages):
        """Compute GRPO loss with simplified policy gradient"""
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
        sequence_logprobs = target_logprobs.sum(dim=1) / mask.sum(dim=1)
        
        # GRPO policy gradient (no clipping)
        policy_loss = -(sequence_logprobs * advantages).mean()
        
        # Get reference logits for KL
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
        """Single training step with group sampling"""
        start_time = time.time()
        
        prompts = batch["prompt"]
        
        # Generate multiple responses per prompt
        all_responses = []
        all_prompts = []
        for prompt in prompts:
            responses = self.generate_group_responses(prompt, self.group_size, max_new_tokens=50)
            all_responses.extend(responses)
            all_prompts.extend([prompt] * self.group_size)
        
        # Create full sequences
        full_texts = [p + r for p, r in zip(all_prompts, all_responses)]
        encodings = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        full_input_ids = encodings["input_ids"].to(self.device)
        full_attention_mask = encodings["attention_mask"].to(self.device)
        
        # Compute rewards for all responses
        with torch.no_grad():
            rewards = self.reward_model(full_input_ids, full_attention_mask)
        
        # Compute group-based advantages
        advantages = self.compute_group_advantages(rewards)
        
        # Compute GRPO loss
        total_loss, policy_loss, kl_div, entropy = self.compute_grpo_loss(
            full_input_ids, full_attention_mask, advantages
        )
        
        time_taken = time.time() - start_time
        
        metrics = {
            "reward": rewards.mean().item(),
            "kl_divergence": kl_div.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "time": time_taken
        }
        
        # Clean up
        del full_input_ids, full_attention_mask, rewards, advantages
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
        train_dataset = GRPODataset(train_data, self.tokenizer, max_length=self.max_length)
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
            
            epoch_metrics = {key: [] for key in ["reward", "kl_divergence", "policy_loss", "entropy", "total_loss", "time"]}
            
            progress_bar = tqdm(train_loader, desc="Training GRPO")
            
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
            for key in ["reward", "kl_divergence", "policy_loss", "entropy", "total_loss"]:
                if len(epoch_metrics[key]) > 0:
                    avg_value = np.mean(epoch_metrics[key])
                    self.training_history[key].append(avg_value)
                    print(f"{key}: {avg_value:.4f}")
            
            # Store time metrics
            if len(epoch_metrics["time"]) > 0:
                avg_time = np.mean(epoch_metrics["time"])
                self.training_history["time_per_iteration"].append(avg_time)
                print(f"Average time per iteration: {avg_time:.2f}s")
            
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
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
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
            axes[0, 2].plot(self.training_history["policy_loss"])
            axes[0, 2].set_title("Policy Loss")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].grid(True)
        
        # Entropy
        if self.training_history["entropy"]:
            axes[1, 0].plot(self.training_history["entropy"])
            axes[1, 0].set_title("Entropy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Entropy")
            axes[1, 0].grid(True)
        
        # Time per iteration
        if self.training_history["time_per_iteration"]:
            axes[1, 1].plot(self.training_history["time_per_iteration"])
            axes[1, 1].set_title("Time per Iteration")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Time (s)")
            axes[1, 1].grid(True)
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves.png")