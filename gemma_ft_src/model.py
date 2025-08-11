# load the pre-trained model
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from omegaconf import OmegaConf
import argparse

model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

class Model:
    def __init__(self, cfg):
        self.model_params = cfg
        self.load_model()

    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_params['pretrained_model_name_or_path'],
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_params['processor'], use_fast=True
        )
        print("Load model params successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="cfg/gemma3-vl_train_0.yaml", help="Path to the config file")
    args = parser.parse_args()
    cli_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, cli_config)
    model = Model(cfg)
