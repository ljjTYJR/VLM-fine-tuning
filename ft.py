# the main loop in the fine-tuning process
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from ft_src.sft_trainer import CustomSFTTrainer
from ft_src.sft_dataset import format_data, collate_fn_ref_ids
from ft_src.model import Model
import argparse
from ft_src.sft_dataset import collate_fn_ref_ids

def train_loop(cfg: DictConfig):
    # 1. load the model
    vlm_model = Model(cfg['model'])
    print("Model and tokenizer loaded successfully.")

    # 2. load the data
    dataset = load_dataset(cfg.dataset.dataset_id, split='train')
    dataset = [format_data(sample, cfg.dataset.prompt_index) for sample in dataset]
    print(f"Dataset {cfg.dataset.dataset_id} loaded successfully with {len(dataset)} samples.")
    # 3. set the trainer
    trainer = CustomSFTTrainer(vlm_model, cfg)
    # begin the training
    trainer.create_trainer(dataset)
    print("Trainer created successfully.")
    trainer.train()

if __name__ == "__main__":
    cfg = OmegaConf.load("cfg/qwen2_5-vl_train.yaml")

    # trainer args
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer.per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--trainer.num_train_epochs", type=int, default=3)
    args = parser.parse_args()
    cli_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    print(f"CLI config: {cli_config}")
    # merge the CLI config with the main config
    cfg = OmegaConf.merge(cfg, cli_config)

    print(f"Final config: {OmegaConf.to_yaml(cfg)}")
    train_loop(cfg)
