# the main loop in the fine-tuning process
import os
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from ft_src.sft_trainer import CustomSFTTrainer
from ft_src.sft_dataset import format_data, collate_fn, generate_description
from ft_src.model import Model
import torch

def test_example(cfg: DictConfig):
    # 1. load the model
    vlm_model = Model(cfg.model.model_id)
    print(f"Model {cfg.model.model_id} loaded successfully, The model dytype is {vlm_model.model.dtype}.")

    # 2. load the data
    dataset = load_dataset(cfg.dataset.dataset_id, split='train')
    data_sample = dataset[0]

    original_hddl = generate_description(data_sample, vlm_model.model, vlm_model.processor)
    print(f"Original HDDL: {original_hddl}")

    # load the lora model
    vlm_model.model.load_adapter(cfg.trainer.output_dir)
    ft_hddl = generate_description(data_sample, vlm_model.model, vlm_model.processor)
    print(f"Fine-tuned HDDL: {ft_hddl}")

    #

if __name__ == "__main__":
    cfg = OmegaConf.load("cfg/qwen2_5-vl.yaml")
    test_example(cfg)
