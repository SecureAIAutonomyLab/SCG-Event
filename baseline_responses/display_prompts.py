import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from tqdm import tqdm





def prompt_generation(dataset, model_name, shot_list):
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

    else:
        return print("Undefined shot prompt")

# Specify the model and shot list
model_name = '../../models/Meta-Llama-3-8B-Instruct'
shot_list = ['six-shot']
dataset_list = ['casie', 'fewevent', 'genia2011', 'm2e2', 'maven', 'mee-en', 'mlee', 'phee']


# Iterate through the datasets
for dataset in dataset_list:
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Shot List: {shot_list[0]}")

    # Generate the six-shot prompt
    messages = prompt_generation(dataset, model_name, shot_list[0])

    # Print the prompt
    print("Prompt:")
    for message in messages:
        print(f"Role: {message['role']}")
        print(f"Content: {message['content']}")
        print()

    print("---")