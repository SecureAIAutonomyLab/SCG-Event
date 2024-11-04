import json
import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoModel
from unsloth import FastLanguageModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

test_names = ["mee", "mlee", "phee"]
train_names = ["CASIE", "FewEvent", "Genia2011", "M2E2", "MAVEN", "MEE", "MLEE", "PHEE"]
n_shots = 6
shot_name = f"{n_shots}shot"

def get_unique_event_types(jsonl_file):
    event_types = set()
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            events = data.get('events', [])
            for event in events:
                event_type = event.get('event_type')
                if event_type:
                    event_types.add(event_type)
    return ', '.join(event_types)

def get_latest_checkpoint_folder(path):
    checkpoints = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {path}")
    latest_checkpoint = max(checkpoints, key=lambda d: int(d.split('-')[-1]))
    return os.path.join(path, latest_checkpoint)

# Loop over all possible train/test combinations
for test_name in test_names:
    for train_name in train_names:
        print(f"Processing train_name: {train_name}, test_name: {test_name}")

        # Load the test set data from the JSONL file
        test_data = []
        with open(f"../../data/{test_name}/{test_name}-test.jsonl", "r") as file:
            for line_num, line in enumerate(file, start=1):
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_num} due to JSON decoding error: {str(e)}")
                    continue

        # Get the path to the latest checkpoint folder
        model_base_path = f"../outputs/Llama3-{train_name}-Alpaca/"
        latest_checkpoint = get_latest_checkpoint_folder(model_base_path)
        model_name = latest_checkpoint

        # Load the fine-tuned model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        text_streamer = TextStreamer(tokenizer)

        # Load train examples with embeddings
        train_examples = []
        with open(f"../../data/{test_name}/{test_name}-train-rag.jsonl", "r") as file:
            for line in file:
                train_examples.append(json.loads(line))

        # Load sentence transformer model for embedding
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        event_types = get_unique_event_types(f"../../data/{test_name}/{test_name}-test.jsonl")

        print(f"Possible Event Types from {test_name}: \n\n{event_types}\n\n")

        # Open the JSONL file for writing
        with open(f"../../eval_event/OOD-experiments-all/{train_name}_train-{test_name}_test-RAG-{shot_name}.jsonl", "w") as outfile:
            # Iterate over each test instance
            for instance in tqdm(test_data, desc="Processing test instances"):
                text = instance["text"]
                golden_labels = instance["events"]

                # Perform inference on the text
                query = f"[INST]As an event detection expert, explore the given text to identify and classify event triggers that indicate events. A trigger is a key word or phrase that most explicitly identifies an event happening. For each identified trigger, provide the trigger word or phrase and the corresponding event type. Event triggers should be selected only from the words present in the given text.\n\nNow you must only consider from the given list of event types when classifying event triggers. You are required to choose from the given list of event types. When classifying the event trigger, only classify from the event types in this given list. Remember that you must always choose an event type from the list of given event types.\n\nHere is the given list of event types that you must choose from: {event_types}\n\nText: "

                # Embed the test instance text
                test_embedding = embedder.encode(text)

                # Calculate cosine similarity with train set embeddings
                sim_scores = []
                for train_ex in train_examples:
                    train_embedding = train_ex["all-MiniLM-L6-v2_embedding"]
                    sim_scores.append(util.cos_sim(test_embedding, train_embedding))

                # Get indices of top n most similar train examples  
                top_indices = torch.topk(torch.tensor(sim_scores), n_shots).indices

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                messages = []

                # Format relevant train examples into messages
                for idx in top_indices:
                    example = train_examples[idx]
                    example_text = example["text"]
                    example_events = example["events"]
                    example_answer = ""
                    for event in example_events:
                        example_answer += f"Trigger: {event['trigger']}, Event Type: {event['event_type']}\n"
                    messages.append({"role": "user", "content": f"{query}{example_text}[/INST]"})
                    messages.append({"role": "assistant", "content": example_answer})

                messages.append({"role": "user", "content": f"{query}{text}[/INST]"})

                model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
                model.eval()

                with torch.no_grad():
                    generated_text = tokenizer.decode(model.generate(model_input, max_new_tokens=300, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators)[0][model_input.shape[-1]:], skip_special_tokens=True)

                # Process the generated output to extract event triggers and types
                lines = generated_text.strip().split("\n")
                generated_labels = []
                for line in lines:
                    if line.startswith("Trigger:"):
                        parts = line.split(",")
                        if len(parts) == 2:
                            trigger = parts[0].replace("Trigger:", "").strip()
                            event_type = parts[1].replace("Event Type:", "").strip().rstrip(';')  # Remove trailing semicolon
                            if trigger.lower() in text.lower():
                                generated_labels.append({"trigger": trigger, "event_type": event_type})

                # Create a dictionary to store the model response and golden labels
                response_data = {
                    "text": text,
                    "golden_labels": golden_labels,
                    "model_labels": generated_labels,
                    "messages": messages
                }

                # Write the response data to the JSONL file
                json.dump(response_data, outfile)
                outfile.write("\n")
