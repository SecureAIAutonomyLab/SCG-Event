from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from datasets import load_dataset
from trl import DPOTrainer
import wandb

dataset_name = "CASIE"
train_name = f"Llama3-{dataset_name}--DPO"

#wandb.init(project="CASIE-DPO", name=f"{train_name}")

dataset = load_dataset("json", data_files=f"pref_data/Llama3-{dataset_name}-pref-data.jsonl", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"outputs_instruct/Llama3-{dataset_name}-Merge",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
)

training_args = TrainingArguments(
    per_device_train_batch_size = 16,
    num_train_epochs = 1,
    learning_rate = 5e-6,
    bf16 = True,
    logging_steps = 1,
    save_strategy="epoch",
    logging_strategy="steps",
    optim = "adamw_8bit",
    lr_scheduler_type = "cosine",
    output_dir = f"outputs_dpo/{train_name}",
    #report_to="wandb"
)

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=2048,
    max_prompt_length=1024
)

dpo_trainer.train()

