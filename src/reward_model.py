"""
Part 1.2: Reward Model Training (Memory Optimized)
Implements pairwise ranking reward model
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

class RewardModel(nn.Module):
    """Reward model that outputs a scalar reward for a given input"""
    
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.config = self.base_model.config
        
        # Add reward head
        self.reward_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Get reward from last token position
        if attention_mask is not None:
            # Get position of last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        # Compute reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        
        # Clean up
        del outputs, hidden_states
        
        return reward

class PreferenceDataset(Dataset):
    """Dataset for preference pairs"""
    
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
        }

class RewardModelTrainer:
    """Trainer for reward model"""
    
    def __init__(self, model_name, output_dir, learning_rate=1e-5, batch_size=2, 
                 num_epochs=1, device="cuda", max_length=256):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.max_length = max_length
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Initialize model
        print(f"Initializing reward model from {model_name}")
        self.model = RewardModel(model_name).to(device)
        
        # Training metrics
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "gradient_norms": []
        }
    
    def compute_loss(self, chosen_rewards, rejected_rewards):
        """
        Compute pairwise ranking loss
        L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
        """
        # Compute logits
        logits = chosen_rewards - rejected_rewards
        
        # Binary cross entropy loss (equivalent to -log(sigmoid(logits)))
        loss = -F.logsigmoid(logits).mean()
        
        # Compute accuracy
        accuracy = (logits > 0).float().mean()
        
        return loss, accuracy
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_gradient_norm = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Forward pass
                chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                
                torch.cuda.empty_cache()
                
                rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                
                # Compute loss
                loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_gradient_norm += total_norm
                num_batches += 1
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy.item():.4f}"
                })
                
                # Clean up
                del chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
                del chosen_rewards, rejected_rewards, loss, accuracy
                
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
        avg_gradient_norm = total_gradient_norm / max(num_batches, 1)
        
        return avg_loss, avg_accuracy, avg_gradient_norm
    
    def evaluate(self, eval_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        all_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
                try:
                    # Move to device
                    chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                    rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                    
                    # Forward pass
                    chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                    rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                    
                    # Compute loss
                    loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards)
                    
                    # Store examples for error analysis
                    correct = (chosen_rewards > rejected_rewards).cpu().numpy()
                    for i in range(len(chosen_rewards)):
                        all_examples.append({
                            "chosen_reward": float(chosen_rewards[i].item()),
                            "rejected_reward": float(rejected_rewards[i].item()),
                            "correct": bool(correct[i])
                        })
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    num_batches += 1
                    
                    # Clean up
                    del chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask
                    del chosen_rewards, rejected_rewards
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM during eval at batch {batch_idx}, skipping...")
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
        
        return avg_loss, avg_accuracy, all_examples
    
    def error_analysis(self, examples, eval_dataset):
        """Perform error analysis on incorrect predictions"""
        
        # Find incorrect predictions
        incorrect_examples = [ex for ex in examples if not ex["correct"]]
        
        print(f"\n--- Error Analysis ---")
        print(f"Total examples: {len(examples)}")
        print(f"Incorrect predictions: {len(incorrect_examples)}")
        print(f"Error rate: {len(incorrect_examples) / max(len(examples), 1) * 100:.2f}%")
        
        # Analyze first 20+ incorrect examples
        analysis_examples = incorrect_examples[:min(25, len(incorrect_examples))]
        
        error_report = {
            "total_examples": len(examples),
            "incorrect_count": len(incorrect_examples),
            "error_rate": len(incorrect_examples) / max(len(examples), 1),
            "sample_errors": []
        }
        
        for i, ex in enumerate(analysis_examples):
            error_report["sample_errors"].append({
                "example_id": i,
                "chosen_reward": ex["chosen_reward"],
                "rejected_reward": ex["rejected_reward"],
                "reward_difference": ex["chosen_reward"] - ex["rejected_reward"]
            })
        
        # Save error analysis
        with open(f"{self.output_dir}/error_analysis.json", "w") as f:
            json.dump(error_report, f, indent=2)
        
        print(f"Error analysis saved to {self.output_dir}/error_analysis.json")
        
        return error_report
    
    def train(self, data_subset_size=None):
        """Main training loop"""
        
        # Load data
        from src.data_preparation import prepare_dataset
        train_data, eval_data, _ = prepare_dataset(
            subset_size=data_subset_size,
            max_length=self.max_length
        )
        
        # Create datasets
        train_dataset = PreferenceDataset(train_data, self.tokenizer, max_length=self.max_length)
        eval_dataset = PreferenceDataset(eval_data, self.tokenizer, max_length=self.max_length)
        
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
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
            train_loss, train_acc, grad_norm = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Evaluate
            val_loss, val_acc, eval_examples = self.evaluate(eval_loader)
            
            # Store metrics
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_accuracy"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_acc)
            self.training_history["gradient_norms"].append(grad_norm)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Gradient Norm: {grad_norm:.4f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.save_model()
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                # Save checkpoint for this epoch
                self.save_model(epoch)
        
        # Error analysis
        _, _, final_eval_examples = self.evaluate(eval_loader)
        self.error_analysis(final_eval_examples, eval_data)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save training history
        with open(f"{self.output_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.model, {
            "val_accuracy": best_val_accuracy,
            "train_history": self.training_history
        }
    
    def save_model(self, epoch=None):
        """Save model checkpoint"""
        checkpoint_name = "reward_model.pt"
        if epoch is not None:
            checkpoint_name = f"reward_model_epoch{epoch}.pt"
            
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "training_history": self.training_history
        }, f"{self.output_dir}/{checkpoint_name}")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}/{checkpoint_name}")
    
    def plot_training_curves(self):
        """Plot training metrics"""
        if not any(self.training_history.values()):
            print("No training history to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        if self.training_history["train_loss"] and self.training_history["val_loss"]:
            axes[0, 0].plot(self.training_history["train_loss"], label="Train")
            axes[0, 0].plot(self.training_history["val_loss"], label="Val")
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if self.training_history["train_accuracy"] and self.training_history["val_accuracy"]:
            axes[0, 1].plot(self.training_history["train_accuracy"], label="Train")
            axes[0, 1].plot(self.training_history["val_accuracy"], label="Val")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Gradient norms
        if self.training_history["gradient_norms"]:
            axes[1, 0].plot(self.training_history["gradient_norms"])
            axes[1, 0].set_title("Gradient Norms")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Norm")
            axes[1, 0].grid(True)
        
        # Hide empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/training_curves.png")