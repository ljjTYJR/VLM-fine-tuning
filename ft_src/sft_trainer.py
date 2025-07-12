from trl import SFTTrainer
from transformers import TrainingArguments, PreTrainedTokenizerBase
from peft import LoraConfig
from ft_src.sft_dataset import collate_fn
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from functools import partial


class CustomSFTTrainer:
    def __init__(self, model, cfg):
        # get model
        self.model = model.model
        self.processor = model.processor
        self.out_dir = cfg.trainer.output_dir
        self.cfg = cfg

    def create_trainer(self, dataset):
        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
        )
        # the SFTConfig comes from
        # https://www.philschmid.de/fine-tune-multimodal-llms-with-trl#3-create-and-prepare-the-multimodal-dataset
        sft_args= SFTConfig(
            output_dir=self.out_dir,                # directory to save and repository id
            num_train_epochs=3,                     # number of training epochs
            per_device_train_batch_size=4,          # batch size per device during training
            gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
            gradient_checkpointing=True,            # use gradient checkpointing to save memory
            optim="adamw_torch_fused",              # use fused adamw optimizer
            logging_steps=5,                         # log every 5 steps
            save_strategy="epoch",                  # save checkpoint every epoch
            learning_rate=2e-4,                     # learning rate, based on QLoRA paper
            bf16=True,                              # use bfloat16 precision
            tf32=True,                              # use tf32 precision
            max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
            lr_scheduler_type="constant",           # use constant learning rate scheduler
            push_to_hub=True,                       # push model to hub
            report_to="tensorboard",                # report metrics to tensorboard
            gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
            dataset_text_field="", # need a dummy field for collator
            dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
        )
        sft_args.remove_unused_columns=False
        sft_args.per_device_train_batch_size = self.cfg.trainer.per_device_train_batch_size

        collate_fn_wrap = partial(
            collate_fn,
            processor=self.processor,
        )
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_args,
            train_dataset=dataset,
            data_collator=collate_fn_wrap,
            processing_class=self.processor.tokenizer,
            peft_config=peft_config,
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.out_dir)