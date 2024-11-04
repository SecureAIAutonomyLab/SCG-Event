from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# load in base model
model_name = "../../models/llama-3-8b-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


###################### load in lora adapter and merge each model individually with the original base model
lora_adapter_name = "../event_instruct/outputs/Llama3-CASIE/checkpoint-198"
lora_model = PeftModel.from_pretrained(model, lora_adapter_name)

merged_model = lora_model.merge_and_unload()

merged_model_name = "outputs_instruct/Llama3-CASIE-Merge"
merged_model = merged_model.to(torch.bfloat16)
merged_model.save_pretrained(merged_model_name, use_safetensors=True, torch_dtype=torch.bfloat16)
tokenizer.save_pretrained(merged_model_name)


