import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from unsloth import FastLanguageModel
import wandb

# Initialize wandb
wandb.init(project="CASIE-Instruct", name="Llama3-CASIE")

model_path = '../../../models/llama-3-8b-Instruct'

dataset = load_dataset("json", data_files="../../data/casie/casie-instruct.jsonl", split='train')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False, # Use 4bit quantization to reduce memory usage. Can be False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 256,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
)


### Main takeaway from process_text function is ADD the tokenizer.eos token to the end of all of your samples!
# Define a function to process each sample
def process_text(sample):
    text = sample["text"]
    # Remove <s> at the beginning and </s> at the end
    if text.startswith("<s>") and text.endswith("</s>"):
        text = text[3:-4] + tokenizer.eos_token
    sample["text"] = text
    return sample

# Apply the function to the dataset
print("Adding Llama3 EoS Tokens...\n")
dataset = dataset.map(process_text)

training_args = TrainingArguments(
    output_dir="../outputs/Llama3-CASIE",
    report_to="wandb",
    per_device_train_batch_size=32,
    lr_scheduler_type="cosine",
    bf16=True,
    warmup_ratio=0.1,
    optim="adamw_8bit",
    num_train_epochs=6,
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    args=training_args,
    dataset_text_field="text"
)

trainer.train()
