# load the pre-trained model
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

class Model:
    def __init__(self, model_id: "Qwen/Qwen2.5-7B-Instruct-1M"):
        self.model_id = model_id
        self.load_model()

    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2"
        )
        # min_pixels = 224*224
        # max_pixels = 640*360
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            use_fast=True,
        )

if __name__ == "__main__":
    vlm_model = Model("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model and tokenizer loaded successfully.")