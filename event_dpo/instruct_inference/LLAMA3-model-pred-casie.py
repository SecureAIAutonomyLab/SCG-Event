import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from tqdm import tqdm

test_name = "casie"
base_name = "Llama3"
train_names = ["CASIE"]

# Load the test set data from the JSONL file
test_data = []
with open(f"../../data/{test_name}/{test_name}-dev.jsonl", "r") as file:
    for line_num, line in enumerate(file, start=1):
        try:
            test_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line {line_num} due to JSON decoding error: {str(e)}")
            continue

# Iterate over each train_name
for train_name in train_names:
    parent_folder = f"../../event_instruct/outputs/{base_name}-{train_name}"
    
    # Get the list of checkpoint folders
    checkpoint_folders = [folder for folder in os.listdir(parent_folder) if folder.startswith("checkpoint-")]
    checkpoint_folders.sort(key=lambda x: int(x.split("-")[-1]))

    # Iterate over each checkpoint folder
    for epoch, ckpt_name in enumerate(checkpoint_folders, start=1):
        print(f"Processing checkpoint: {ckpt_name} for model: {train_name} (Epoch {epoch})")
        if epoch in [1, 2, 3, 4, 5]:
            continue
        # Load the fine-tuned model and tokenizer
        model_name = f"{parent_folder}/{ckpt_name}"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        text_streamer = TextStreamer(tokenizer)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Open the JSONL file for writing
        with open(f"Llama3-{train_name}-dev-responses.jsonl", "w") as outfile:
            # Iterate over each randomly selected test instance
            for instance in tqdm(test_data, desc=f"Processing test instances for epoch {epoch} of model {train_name}"):
                text = instance["text"]
                golden_labels = instance["events"]
                
                # Construct the query with the event types list
                system_prompt = "As an event detection expert, explore the given text to identify and classify event triggers that indicate events. A trigger is a key word or phrase that most explicitly identifies an event happening. For each identified trigger, provide the trigger word or phrase and the corresponding event type. Output each unique trigger-event type pair only once. Event triggers should be selected only from the words present in the given text. Remember that there should not be more identified triggers of the same word/phrase than exist in the given text."
                
                messages = [
                    {"role": "user", "content": f"[INST]{system_prompt}\n\nText: {text}[/INST]"}
                ]

                model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                model.eval()
                
                with torch.no_grad():
                    generated_text = tokenizer.decode(model.generate(model_input, max_new_tokens=300, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators)[0][model_input.shape[-1]:], skip_special_tokens=True)

                # Process the generated output to extract event triggers and types
                lines = generated_text.strip().split("\n")
                generated_labels = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("Trigger:"):
                        parts = line.split(",")
                        if len(parts) == 2:
                            trigger = parts[0].replace("Trigger:", "").strip()
                            event_type = parts[1].replace("Event Type:", "").strip().rstrip(';')  # Remove trailing semicolon
                            if trigger.lower() == "none":
                                generated_labels.append({"trigger": "None", "event_type": "None"})
                            elif trigger.lower() in text.lower():
                                generated_labels.append({"trigger": trigger, "event_type": event_type})

                # Create a dictionary to store the model response and golden labels
                response_data = {
                    "text": text,
                    "golden_labels": golden_labels,
                    "model_labels": generated_labels,
                    "model_response": generated_text,
                    "messages": messages
                }

                # Write the response data to the JSONL file
                json.dump(response_data, outfile)
                outfile.write("\n")