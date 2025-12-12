"""
RLHF Assignment Implementation
Implements PPO, GRPO, and DPO for language model alignment
"""

__version__ = "1.0.0"
__author__ = "RLHF Assignment"

from .data_preparation import prepare_dataset
from .reward_model import RewardModel, RewardModelTrainer
from .ppo_trainer import PPOTrainer
from .grpo_trainer import GRPOTrainer
from .dpo_trainer import DPOTrainer
from .evaluation import ModelEvaluator
from .utils import set_seed, setup_logging

__all__ = [
    "prepare_dataset",
    "RewardModel",
    "RewardModelTrainer",
    "PPOTrainer",
    "GRPOTrainer",
    "DPOTrainer",
    "ModelEvaluator",
    "set_seed",
    "setup_logging"
]