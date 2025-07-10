# the main loop in the fine-tuning process
import os
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from ft_src.sft_trainer import CustomSFTTrainer
from ft_src.sft_dataset import format_data, collate_fn
from ft_src.model import Model
import torch

def train_loop(cfg: DictConfig):
    # 1. load the model
    vlm_model = Model(cfg.model.model_id)
    print(f"Model {cfg.model.model_id} loaded successfully, The model dytype is {vlm_model.model.dtype}.")

    # 2. load the data
    dataset = load_dataset(cfg.dataset.dataset_id, split='train')
    dataset = [format_data(sample) for sample in dataset]
    print(f"Dataset {cfg.dataset.dataset_id} loaded successfully with {len(dataset)} samples.")

    # 3. set the trainer
    trainer = CustomSFTTrainer(vlm_model, cfg)
    # begin the training
    trainer.create_trainer(dataset)
    print("Trainer created successfully.")
    trainer.train()

if __name__ == "__main__":
    cfg = OmegaConf.load("cfg/qwen2_5-vl.yaml")
    train_loop(cfg)
