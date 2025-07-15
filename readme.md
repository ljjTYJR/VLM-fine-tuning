The demo of fine-tuning a VLLM model with a custom dataset.

# Preparation
- The model: Qwen2.5-VL 7B model.
- The dataset: A custom dataset, which from image to generate the pddl file: https://huggingface.co/datasets/shuooru/image-hddl-dataset

## Alvis usage
- load needed modules: `module load virtualenv/20.26.2-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a SciPy-bundle/2024.05-gfbf-2024a`
- bind the virtual environment: `virtualenv --system-site-packages my_env`
- activate the virtual environment: `source my_env/bin/activate`
- The working directory: `cd /mimer/NOBACKUP/groups/naiss2025-22-933`
- Go to the qwen directory and run the training: `python ft.py --config_file='cfg/qwen2_5-vl_train_0.yaml' --trainer.num_train_epochs=100`

# Reference
## Fine-tuning the model
- https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
- https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
- Qwen2.5-VL 7B model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

## Converting the model to ollama
- https://github.com/ggml-org/llama.cpp/discussions/2948
- https://github.com/ollama/ollama/blob/main/docs/import.md#Importing-a-fine-tuned-adapter-from-Safetensors-weights