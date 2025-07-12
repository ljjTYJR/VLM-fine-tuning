# load the pre-trained model
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from omegaconf import OmegaConf

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=False, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

quantization_map = {
    "bnb_config": bnb_config,
}

class Model:
    def __init__(self, cfg):
        self.model_params = OmegaConf.to_container(cfg, resolve=True)
        self.model_params['torch_dtype'] = dtype_map[cfg['torch_dtype']]
        # if exist "quantization_config"
        if "quantization_config" in self.model_params:
            self.model_params['quantization_config'] = quantization_map[cfg['quantization_config']]
        print(f"Model parameters: {self.model_params}")

        self.load_model()

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            **self.model_params
        )
        min_pixels = 224*224
        max_pixels = 640*320
        self.processor = AutoProcessor.from_pretrained(
            # todo: make it configurable
            self.model_params['pretrained_model_name_or_path'],
            use_fast=True,
            # min_pixels=min_pixels,
            # max_pixels=max_pixels
        )

if __name__ == "__main__":
    vlm_model = Model("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model and tokenizer loaded successfully.")