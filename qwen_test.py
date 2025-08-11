# the main loop in the fine-tuning process
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from qwen_ft_src.sft_trainer import CustomSFTTrainer
from qwen_ft_src.sft_dataset import format_data, collate_fn_ref_ids
from qwen_ft_src.model import Model
import argparse
from qwen_ft_src.sft_dataset import collate_fn_ref_ids

def test_batch_processing(cfg: DictConfig):
    # 1. load the model
    vlm_model = Model(cfg['model'])
    print("Model and tokenizer loaded successfully.")

    # 2. load the data
    dataset = load_dataset(cfg.dataset.dataset_id, split='train')
    dataset = [format_data(sample, cfg.dataset.prompt_index) for sample in dataset]

    # sample data
    samples = [dataset[0], dataset[1], dataset[2]]
    # test sample method
    collate_fn_ref_ids(samples, vlm_model.processor)
    return

if __name__ == "__main__":
    # trainer args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="cfg/qwen2_5-vl_train_0.yaml", help="Path to the config file")
    parser.add_argument("--trainer.num_train_epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    cli_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    print(f"CLI config: {cli_config}")
    # merge the CLI config with the main config
    cfg = OmegaConf.load(args.config_file)
    print(f"Base config: {OmegaConf.to_yaml(cfg)}")
    cfg = OmegaConf.merge(cfg, cli_config)

    print(f"Final config: {OmegaConf.to_yaml(cfg)}")
    test_batch_processing(cfg)
