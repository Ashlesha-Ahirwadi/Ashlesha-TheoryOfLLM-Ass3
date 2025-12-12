"""
Part 3: Direct Preference Optimization (DPO) - Memory Optimized
Implements DPO without explicit reward modeling
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
import gc

class DPODataset(Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Tokenize chosen and rejected
        chosen_encoding = self.tokenizer(
            example["prompt"] + example["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            example["prompt"] + example["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

class DPOTrainer:
    """Trainer for Direct Preference Optimization"""
    
    def __init__(self, model_name, output_dir, learning_rate=1e-6, 
                 batch_size=1, beta=0.1, num_epochs=1, device="cuda", max_length=256):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = beta  # Temperature parameter for DPO
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
        
        # Training metrics
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
            "reward_margin": []
        }
    
    def compute_sequence_logprob(self, model, input_ids, attention_mask):
        """Compute log probability of a sequence"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and labels for next token prediction
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_logprobs = torch.gather(
            logprobs, 
            2, 
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = attention_mask[:, 1:]
        target_logprobs = target_logprobs * mask
        
        # Sum log probabilities over sequence
        sequence_logprob = target_logprobs.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Clean up
        del outputs, logits, logprobs
        
        return sequence_logprob
    
    def compute_dpo_loss(self, chosen_input_ids, chosen_attention_mask, 
                         rejected_input_ids, rejected_attention_mask):
        """Compute DPO loss"""
        
        # Compute log probabilities for policy model
        policy_chosen_logprob = self.compute_sequence_logprob(
            self.policy_model, chosen_input_ids, chosen_attention_mask
        )
        
        torch.cuda.empty_cache()
        
        policy_rejected_logprob = self.compute_sequence_logprob(
            self.policy_model, rejected_input_ids, rejected_attention_mask
        )
        
        # Compute log probabilities for reference model
        with torch.no_grad():
            ref_chosen_logprob = self.compute_sequence_logprob(
                self.ref_model, chosen_input_ids, chosen_attention_mask
            )
            
            torch.cuda.empty_cache()
            
            ref_rejected_logprob = self.compute_sequence_logprob(
                self.ref_model, rejected_input_ids, rejected_attention_mask
            )
        
        # Compute implicit rewards
        policy_chosen_reward = self.beta * (policy_chosen_logprob - ref_chosen_logprob)
        policy_rejected_reward = self.beta * (policy_rejected_logprob - ref_rejected_logprob)
        
        # Compute logits for preference
        logits = policy_chosen_reward - policy_rejected_reward
        
        # DPO loss: -log(sigmoid(logits))
        loss = -F.logsigmoid(logits).mean()
        
        # Compute accuracy
        accuracy = (logits > 0).float().mean()
        
        # Compute reward statistics
        reward_margin = (policy_chosen_reward - policy_rejected_reward).mean()
        
        return loss, accuracy, policy_chosen_reward.mean(), policy_rejected_reward.mean(), reward_margin
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.policy_model.train()
        
        total_loss = 0
        total_accuracy = 0
        total_chosen_rewards = 0
        total_rejected_rewards = 0
        total_reward_margin = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training DPO")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Compute DPO loss
                loss, accuracy, chosen_reward, rejected_reward, reward_margin = self.compute_dpo_loss(
                    chosen_input_ids, chosen_attention_mask,
                    rejected_input_ids, rejected_attention_mask
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_chosen_rewards += chosen_reward.item()
                total_rejected_rewards += rejected_reward.item()
                total_reward_margin += reward_margin.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy.item():.4f}",
                    "margin": f"{reward_margin.item():.4f}"
                })
                
                # Clean up
                del chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
                del loss, accuracy, chosen_reward, rejected_reward, reward_margin
                
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
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        avg_chosen_rewards = total_chosen_rewards / max(num_batches, 1)
        avg_rejected_rewards = total_rejected_rewards / max(num_batches, 1)
        avg_reward_margin = total_reward_margin / max(num_batches, 1)
        
        return avg_loss, avg_accuracy, avg_chosen_rewards, avg_rejected_rewards, avg_reward_margin
    
    def evaluate(self, eval_loader):
        """Evaluate model"""
        self.policy_model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_chosen_rewards = 0
        total_rejected_rewards = 0
        total_reward_margin = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                try:
                    # Move to device
                    chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                    rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                    
                    # Compute DPO loss
                    loss, accuracy, chosen_reward, rejected_reward, reward_margin = self.compute_dpo_loss(
                        chosen_input_ids, chosen_attention_mask,
                        rejected_input_ids, rejected_attention_mask
                    )
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    total_chosen_rewards += chosen_reward.item()
                    total_rejected_rewards += rejected_reward.item()
                    total_reward_margin += reward_margin.item()
                    num_batches += 1
                    
                    # Clean up
                    del chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM during eval, skipping batch...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        avg_chosen_rewards = total_chosen_rewards / max(num_batches, 1)
        avg_rejected_rewards = total_rejected_rewards / max(num_batches, 1)
        avg_reward_margin = total_reward_margin / max(num_batches, 1)
        
        return avg_loss, avg_accuracy, avg_chosen_rewards, avg_rejected_rewards, avg_reward_margin
    
    def train(self, data_subset_size=None):
        """Main training loop"""
        
        # Load data
        from src.data_preparation import prepare_dataset
        train_data, eval_data, _ = prepare_dataset(
            subset_size=data_subset_size,
            max_length=self.max_length
        )
        
        # Create datasets
        train_dataset = DPODataset(train_data, self.tokenizer, max_length=self.max_length)
        eval_dataset = DPODataset(eval_data, self.tokenizer, max_length=self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.batch_size * 2, 
            shuffle=False
        )
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=self.learning_rate)
        num_training_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_val_accuracy = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc, train_chosen, train_rejected, train_margin = self.train_epoch(
                train_loader, optimizer, scheduler
            )
            
            # Evaluate
            val_loss, val_acc, val_chosen, val_rejected, val_margin = self.evaluate(eval_loader)
            
            # Store metrics
            self.training_history["loss"].append(train_loss)
            self.training_history["accuracy"].append(train_acc)
            self.training_history["chosen_rewards"].append(train_chosen)
            self.training_history["rejected_rewards"].append(train_rejected)
            self.training_history["reward_margin"].append(train_margin)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Reward Margin: {train_margin:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model()
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                # Save anyway for each epoch
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
        
        # Loss
        if self.training_history["loss"]:
            axes[0, 0].plot(self.training_history["loss"])
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)
        
        # Accuracy
        if self.training_history["accuracy"]:
            axes[0, 1].plot(self.training_history["accuracy"])
            axes[0, 1].set_title("Training Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].grid(True)
        
        # Rewards
        if self.training_history["chosen_rewards"] and self.training_history["rejected_rewards"]:
            axes[1, 0].plot(self.training_history["chosen_rewards"], label="Chosen")
            axes[1, 0].plot(self.training_history["rejected_rewards"], label="Rejected")
            axes[1, 0].set_title("Implicit Rewards")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Reward")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Reward Margin
        if self.training_history["reward_margin"]:
            axes[1, 1].plot(self.training_history["reward_margin"])
            axes[1, 1].set_title("Reward Margin")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Margin")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves.png")