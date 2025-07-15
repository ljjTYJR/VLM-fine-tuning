#!/bin/bash
# This script is used to run the fine-tuning process for the Qwen2.5-VL model.
python ft.py --config_file='cfg/qwen2_5-vl_train_0.yaml' --trainer.num_train_epochs=100
# python ft.py --config_file='cfg/qwen2_5-vl_train_1.yaml' --trainer.num_train_epochs=100
# python ft.py --config_file='cfg/qwen2_5-vl_train_2.yaml' --trainer.num_train_epochs=100
# python ft.py --config_file='cfg/qwen2_5-vl_train_3.yaml' --trainer.num_train_epochs=100