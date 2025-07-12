The demo of fine-tuning a VLLM model with a custom dataset.

# Preparation
- The model: Qwen2.5-VL 7B model.
- The dataset: A custom dataset, which from image to generate the pddl file: https://huggingface.co/datasets/shuooru/image-hddl-dataset


# Reference
## Fine-tuning the model
- https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
- https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
- Qwen2.5-VL 7B model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

## Converting the model to ollama
- https://github.com/ggml-org/llama.cpp/discussions/2948
- https://github.com/ollama/ollama/blob/main/docs/import.md#Importing-a-fine-tuned-adapter-from-Safetensors-weights