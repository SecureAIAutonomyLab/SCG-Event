import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

model_path = '../../../models/Mistral-7B-v0.2-hf'

dataset = load_dataset("json", data_files="../../data/m2e2/m2e2-instruct.jsonl", split='train')

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# Define a function to process each sample
# def process_text(sample):
#     text = sample["text"]
#     # Remove <s> at the beginning and </s> at the end, add llama3 eos
#     if text.startswith("<s>") and text.endswith("</s>"):
#         text = text[3:-4] + tokenizer.eos_token
#     sample["text"] = text
#     return sample

# Apply the function to the dataset
# print("Adding Llama3 EoS Tokens...\n")
# dataset = dataset.map(process_text)

training_args = TrainingArguments(
    output_dir="../outputs/Mistral-M2E2-FullFT",
    per_device_train_batch_size=1,
    lr_scheduler_type="cosine",
    bf16=True,
    warmup_ratio=0.1,
    optim="adamw_8bit",
    num_train_epochs=6,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    gradient_accumulation_steps=1,
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=512,
    args=training_args,
    dataset_text_field="text"
)

trainer.train()
