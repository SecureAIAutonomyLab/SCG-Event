import json
import jsonlines
import anthropic
import os
from tqdm import tqdm
import random

# Initialize Anthropic client
client = anthropic.Anthropic(api_key='INSERT_API_KEY_HERE')
model_name = "claude-3-5-sonnet-20240620"

def modify_text(text, triggers, system_prompt):
    trigger_list = ", ".join(f'"{t}"' for t in triggers)
    user_prompt = f"""Please modify the following text.
    The text contains the following trigger phrases that MUST be preserved exactly as they are: {trigger_list}. 
    Ensure these triggers remain in the text unchanged and in their original context.

    Original text:
    {text}"""

    message = client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0.6,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )

    return message.content[0].text.strip()

def check_triggers(original_text, modified_text, triggers):
    for trigger in triggers:
        if trigger not in modified_text:
            return False
    return True

def modify_dataset(input_file, output_file, sample_size):
    system_prompt = """
    You are an AI assistant tasked with modifying text for event detection datasets to make variations of the original text. Your job is to change only entities, locations, dates, times, and similar specific details in the given text. Do not alter the overall structure, events, or meaning of the text. Maintain the same writing style and tone. Your output should be the modified text only, without any explanations or additional comments. You must only change entities, locations, dates, and times and keep everything else about the original text exactly the same as the original text given. Ensure that the modified text contains exactly all of the user provided trigger words/phrases from the original text. Only modify the entities, locations, dates, times, and similar specific details in the given text.

    Rules:
    1. Change names of people, organizations, and locations.
    2. Modify dates and times, but keep them realistic and consistent with the events described.
    3. Alter specific numbers (e.g., ages, quantities) slightly, but keep them plausible.
    4. Do not change the events, their types, or their triggers.
    5. Maintain the same paragraph structure and quotations (if any).
    6. Ensure the modified text remains coherent and logical.
    7. Keep all original trigger phrases intact and in the same context. These phrases are critical and must not be altered in any way.
    8. Other than the modifications, keep all other input text the same. We are just trying to make slight variations of the original text. You MUST output the rest of the text EXACTLY as given otherwise.
    9. If a sentence or phrase does not contain any entities, locations, dates, or times to be changed, leave it completely unchanged.
    10. Ensure that you include all of the triggers that are given by the user for the original text in the modified text in their  original context.

    Remember, the goal is to create a subtle variation of the original text while preserving its core structure and meaning. Be extremely careful not to alter anything beyond the specific elements mentioned in the rules.
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the input file
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Print the keys of the first item to understand the structure
    if data:
        print(f"Keys in the first item of {os.path.basename(input_file)}: {list(data[0].keys())}")

    successful_modifications = 0
    processed_indices = set()
    
    with open(output_file, 'w') as f:
        pbar = tqdm(total=sample_size, desc=f"Processing {os.path.basename(input_file)}")
        
        while successful_modifications < sample_size and len(processed_indices) < len(data):
            # Select a random unprocessed item
            available_indices = set(range(len(data))) - processed_indices
            if not available_indices:
                break
            
            index = random.choice(list(available_indices))
            processed_indices.add(index)
            item = data[index]

            if 'text' not in item:
                print(f"Skipping an item without 'text' key")
                continue

            original_text = item['text']
            triggers = []
            if 'events' in item:
                triggers = [event.get('trigger', '') for event in item['events'] if 'trigger' in event]
            
            skip_trigger_check = (len(triggers) == 1 and triggers[0].lower() == 'none')

            max_attempts = 20
            for attempt in range(max_attempts):
                modified_text = modify_text(original_text, triggers, system_prompt)
                
                if skip_trigger_check or check_triggers(original_text, modified_text, triggers):
                    # Create a new item with the same structure as the original, but with modified text and original text
                    new_item = item.copy()
                    new_item['original_text'] = original_text
                    new_item['text'] = modified_text
                    json.dump(new_item, f)
                    f.write('\n')
                    f.flush()  # Flush the file buffer to ensure data is written to disk
                    print(f"Successfully wrote item to {output_file}")  # Debug print
                    successful_modifications += 1
                    pbar.update(1)
                    break
                
                if attempt == max_attempts - 1:
                    print(f"Failed to preserve triggers after {max_attempts} attempts. Skipping this instance.")

        pbar.close()

    print(f"Modified dataset saved to {output_file}")
    print(f"Successfully modified {successful_modifications} out of {sample_size} requested instances.")

def process_all_datasets(input_base_path, output_base_path, sample_size):
    datasets = ['m2e2', 'maven', 'mlee']
    
    # Ensure the output base directory exists
    os.makedirs(output_base_path, exist_ok=True)
    
    for dataset in datasets:
        input_file = os.path.join(input_base_path, dataset, f"{dataset}-test.jsonl")
        
        if not os.path.exists(input_file):
            print(f"Warning: Test file not found for dataset {dataset}: {input_file}")
            continue
        
        output_file = os.path.join(output_base_path, f"{dataset}-test-modified.jsonl")
        
        print(f"Processing file: {input_file}")
        print(f"Output will be saved to: {output_file}")
        modify_dataset(input_file, output_file, sample_size)

# Usage
input_base_path = '../data'
output_base_path = 'modified_data'
sample_size = 50
process_all_datasets(input_base_path, output_base_path, sample_size)