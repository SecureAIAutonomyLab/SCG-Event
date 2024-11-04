import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def get_top_similar_samples(input_text, train_file, top_k):
    # Load the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Compute the embedding for the input text
    input_embedding = model.encode([input_text])
    
    # Read the train.jsonl file and compute cosine similarities
    similarities = []
    samples = []
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            sample_embedding = sample['all-MiniLM-L6-v2_embedding']
            similarity = cosine_similarity(input_embedding, [sample_embedding])[0][0]
            similarities.append((i, similarity))
            samples.append(sample)
    
    # Sort the samples by cosine similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top-k most similar samples
    top_samples = [samples[idx] for idx, _ in similarities[:top_k]]
    return top_samples


def prompt_generation(dataset, model_name, shot_list, input_text):
    # Read the dataset-specific train file and gather unique event types
    event_types = set()
    data_train_path = os.path.join(dataset, f"{dataset}_train.jsonl")
    with open(data_train_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'events' in data:
                for event in data['events']:
                    if 'event_type' in event:
                        event_types.add(event['event_type'])

    # Convert the set of event types to a list
    event_types_list = list(event_types)

    # Format the event types list as a string representation
    event_types_str = str(event_types_list)

    task_description = f"You are an event extractor designed to check for the presence of a specific event in a sentence and to locate the single corresponding event trigger. Task Description: Your task is to analyze the provided text to identify the event trigger. An event trigger is a key word in the text that most explicitly conveys the occurence of the event. Following this, classify the event trigger into the correct event type from the provided list. You should only classify the event trigger into one of the event types that is listed in the List of Event Types. Only respond with the identified Trigger and the corresponding Event Type of the given text and nothing else. Do not explain your decision and do not describe your reasoning, only respond with the Trigger and Event Type. \n\nList of Event Types: {event_types_str}"

    if shot_list == 'zero-shot':
        if model_name == '../../models/Mistral-7B-Instruct-v0.2':
            messages = [
                {"role": "user", "content": f'{task_description}\n\n You should output in this format: Trigger: trigger, Event Type: event type. Separate each pair of trigger and event type with a newline.'}
            ]
        else:
            messages = [
                {"role": "system", "content": f'{task_description}\n\n You should output in this format: Trigger: trigger, Event Type: event type. Separate each pair of trigger and event type with a newline.'}
            ]
        return messages

    if shot_list == 'six-shot':
        # Read the dataset-specific train file and gather few-shot examples
        few_shot_examples = []
        data_train_path = os.path.join(dataset, f"{dataset}_train.jsonl")
        with open(data_train_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'text' in data and 'events' in data:
                    example_events = []
                    for event in data['events']:
                        if 'trigger' in event and 'event_type' in event:
                            example_events.append((event['trigger'], event['event_type']))
                    if example_events:
                        few_shot_examples.append((data['text'], example_events))
                    if len(few_shot_examples) == 6:
                        break

        if model_name == '../../models/Mistral-7B-Instruct-v0.2':
            messages = [
                {"role": "user", "content": f"{task_description}\n\n{few_shot_examples[0][0]}"}
            ]
            few_shot_a = ""
            for trigger, event_type in few_shot_examples[0][1]:
                few_shot_a += f"Trigger: {trigger}, Event Type: {event_type}\n"
            messages.append({"role": "assistant", "content": few_shot_a.strip()})
            few_shot_examples = few_shot_examples[1:]
        else:
            messages = [
                {"role": "system", "content": task_description}
            ]

        for i, example in enumerate(few_shot_examples, start=1):
            few_shot_q = example[0]
            few_shot_a = ""
            for trigger, event_type in example[1]:
                few_shot_a += f"Trigger: {trigger}, Event Type: {event_type}\n"

            messages.extend([
                {"role": "user", "content": few_shot_q},
                {"role": "assistant", "content": few_shot_a.strip()}
            ])

        return messages
        
    if shot_list == 'three-shot':
        # Read the dataset-specific train file and gather few-shot examples
        few_shot_examples = []
        data_train_path = os.path.join(dataset, f"{dataset}_train.jsonl")
        with open(data_train_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'text' in data and 'events' in data:
                    example_events = []
                    for event in data['events']:
                        if 'trigger' in event and 'event_type' in event:
                            example_events.append((event['trigger'], event['event_type']))
                    if example_events:
                        few_shot_examples.append((data['text'], example_events))
                    if len(few_shot_examples) == 3:
                        break

        if model_name == '../../models/Mistral-7B-Instruct-v0.2':
            messages = [
                {"role": "user", "content": f"{task_description}\n\n{few_shot_examples[0][0]}"}
            ]
            few_shot_a = ""
            for trigger, event_type in few_shot_examples[0][1]:
                few_shot_a += f"Trigger: {trigger}, Event Type: {event_type}\n"
            messages.append({"role": "assistant", "content": few_shot_a.strip()})
            few_shot_examples = few_shot_examples[1:]
        else:
            messages = [
                {"role": "system", "content": task_description}
            ]

        for i, example in enumerate(few_shot_examples, start=1):
            few_shot_q = example[0]
            few_shot_a = ""
            for trigger, event_type in example[1]:
                few_shot_a += f"Trigger: {trigger}, Event Type: {event_type}\n"

            messages.extend([
                {"role": "user", "content": few_shot_q},
                {"role": "assistant", "content": few_shot_a.strip()}
            ])

        return messages
    
    if shot_list == 'three-shot_RAG':
        # Retrieve the top-3 most similar samples using cosine similarity
        data_train_path = os.path.join(dataset, f"{dataset}_train_processed.jsonl")
        few_shot_examples = get_top_similar_samples(input_text, data_train_path, top_k=3)

        if model_name == '../../models/Mistral-7B-Instruct-v0.2':
            messages = [
                {"role": "user", "content": f"{task_description}\n\n{few_shot_examples[0]['text']}"}
            ]
            few_shot_a = ""
            for event in few_shot_examples[0]['events']:
                if 'trigger' in event and 'event_type' in event:
                    few_shot_a += f"Trigger: {event['trigger']}, Event Type: {event['event_type']}\n"
            messages.append({"role": "assistant", "content": few_shot_a.strip()})
            few_shot_examples = few_shot_examples[1:]
        else:
            messages = [
                {"role": "system", "content": task_description}
            ]

        for example in few_shot_examples:
            few_shot_q = example['text']
            few_shot_a = ""
            for event in example['events']:
                if 'trigger' in event and 'event_type' in event:
                    few_shot_a += f"Trigger: {event['trigger']}, Event Type: {event['event_type']}\n"

            messages.extend([
                {"role": "user", "content": few_shot_q},
                {"role": "assistant", "content": few_shot_a.strip()}
            ])

        return messages
    
    if shot_list == 'six-shot_RAG':
        # Retrieve the top-3 most similar samples using cosine similarity
        data_train_path = os.path.join(dataset, f"{dataset}_train_processed.jsonl")
        few_shot_examples = get_top_similar_samples(input_text, data_train_path, top_k=6)

        if model_name == '../../models/Mistral-7B-Instruct-v0.2':
            messages = [
                {"role": "user", "content": f"{task_description}\n\n{few_shot_examples[0]['text']}"}
            ]
            few_shot_a = ""
            for event in few_shot_examples[0]['events']:
                if 'trigger' in event and 'event_type' in event:
                    few_shot_a += f"Trigger: {event['trigger']}, Event Type: {event['event_type']}\n"
            messages.append({"role": "assistant", "content": few_shot_a.strip()})
            few_shot_examples = few_shot_examples[1:]
        else:
            messages = [
                {"role": "system", "content": task_description}
            ]

        for example in few_shot_examples:
            few_shot_q = example['text']
            few_shot_a = ""
            for event in example['events']:
                if 'trigger' in event and 'event_type' in event:
                    few_shot_a += f"Trigger: {event['trigger']}, Event Type: {event['event_type']}\n"

            messages.extend([
                {"role": "user", "content": few_shot_q},
                {"role": "assistant", "content": few_shot_a.strip()}
            ])

        return messages

    else:
        return print("Undefined shot prompt")



def LLM_text_generation(dataset, model_name, shot_list, input_text, model, tokenizer):
    query = prompt_generation(dataset, model_name, shot_list, input_text)

    if model_name == '../../models/Mistral-7B-Instruct-v0.2' and shot_list == 'zero-shot':
        query[0]["content"] += f"\n\n{input_text.strip()}"
    else:
        query.extend([{"role": "user", "content": input_text.strip()}])

    messages = query

    model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    model.eval()
    
    if model_name == '../../models/Meta-Llama-3-8B-Instruct':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        with torch.no_grad():
            generated_text = tokenizer.decode(model.generate(model_input, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators)[0][model_input.shape[-1]:], skip_special_tokens=True)
 
        return generated_text

    else:
        with torch.no_grad():
            generated_text = tokenizer.decode(model.generate(model_input, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)[0][model_input.shape[-1]:], skip_special_tokens=True)
        
 
        return generated_text



# for loop #1 iterate through models_list (model from models_list)
# for loop #2 iterate through dataset_list (dataset from dataset_list)
# for loop #3 iterate through shot_list (shot from shot_list)

# dataset_list = ['casie', 'fewevent', 'genia2011', 'm2e2', 'maven', 'mee-en', 'mlee', 'phee']

# models_list = ['../../models/Mistral-7B-Instruct-v0.2', '../../models/Llama-2-7b-chat-hf', '../../models/Meta-Llama-3-8B-Instruct']

# shot_list = ['zero-shot', 'three-shot']


dataset_list = ['casie', 'fewevent', 'genia2011', 'm2e2', 'maven', 'mee-en', 'mlee', 'phee']

models_list = ['../../models/Mistral-7B-Instruct-v0.2', '../../models/Meta-Llama-3-8B-Instruct']

shot_list = ['three-shot_RAG', 'six-shot_RAG']


test_data = []
for model_type in models_list:
    for dataset in dataset_list:
        for shot in shot_list:
            # Convert model_type to a simplified model name
            if model_type == '../../models/Mistral-7B-Instruct-v0.2':
                model_name_converted = 'mistral_7B_instruct'
            elif model_type == '../../models/Llama-2-7b-chat-hf':
                model_name_converted = 'llama2_7B_instruct'
            elif model_type == '../../models/Meta-Llama-3-8B-Instruct':
                model_name_converted = 'llama3_8B_instruct'
            else:
                model_name_converted = model_type.replace('/', '_').replace('.', '_')

            test_data = []
            with open(f"{dataset}/{dataset}_test.jsonl", "r") as file:
                for line_num, line in enumerate(file, start=1):
                    try:
                        test_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping line {line_num} due to JSON decoding error: {str(e)}")
                        continue

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_type,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(model)

            with open(f"{dataset}_{model_name_converted}_{shot}.jsonl", "w") as outfile:
                # Iterate over each test instance
                for index, instance in enumerate(tqdm(test_data, desc="Processing test instances"), start=1):
                    if index > 250:
                        break

                    text = instance["text"]
                    golden_labels = instance["events"]
                    # ###
                    # # Check the token length before generating the response
                    # query = prompt_generation(dataset, model_type, shot)

                    # if model_type == '../../models/Mistral-7B-Instruct-v0.2' and shot_list == 'zero-shot':
                    #     query[0]["content"] += f"\n\n{text.strip()}"
                    # else:
                    #     query.extend([{"role": "user", "content": text.strip()}])

                    # messages = query

                    # model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                    # token_length = model_input.shape[-1]
                    # max_token_length = model.config.max_position_embeddings - 256
                    # print(token_length)
                    # if token_length > max_token_length:
                    #     print(f"Skipping sample {index} due to token length exceeding the maximum allowed length.")
                    #     continue
                    # ###           
                    model_response = LLM_text_generation(dataset, model_type, shot, text, model, tokenizer)
                    generated_labels = []
                    response_lines = model_response.split("\n")
                    for line in response_lines:
                        line = line.strip()
                        if line.startswith("Trigger:"):
                            parts = line.split(",")
                            if len(parts) == 2:
                                trigger = parts[0].replace("Trigger:", "").strip()
                                event_type = parts[1].replace("Event Type:", "").strip().rstrip(';')  # Remove trailing semicolon
                                generated_labels.append({"trigger": trigger, "event_type": event_type})

                    # Create a dictionary to store the model response and golden labels
                    response_data = {
                        "text": text,
                        "golden_labels": golden_labels,
                        "model_labels": generated_labels,
                        "model_responses": model_response
                    }

                    # Write the response data to the JSONL file
                    json.dump(response_data, outfile)
                    outfile.write("\n")