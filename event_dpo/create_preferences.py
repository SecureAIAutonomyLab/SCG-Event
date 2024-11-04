import json
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel

def generate_instruction_variations():
    instructions = [
        "As an event detection assistant, your task is to identify the event triggers in the given text. An event trigger is a word or phrase that most explicitly describes the event happening in the text. Classify each trigger into its corresponding event type from the predefined set.",
        "In your role as an event detection assistant, find the event triggers which are words or phrases that most explicitly describe the events occurring in the text. Categorize each trigger into its respective event type.",
        "You are an event detection assistant. Locate the event triggers, which are words or phrases that most clearly express the events in the text. Determine the corresponding event type for each trigger from the provided categories and output the trigger words along with their associated types.",
        "Analyze the given text and pinpoint the event triggers which are specific words or phrases that most explicitly indicate the occurrence of events. As an event detection assistant, classify each event trigger into one of the predefined event types and list the triggers along with their assigned types.",
        "In your capacity as an event detection assistant, examine the provided passage and identify the event triggers, which are key terms that most explicitly signify events taking place. For each event trigger found, specify the relevant word(s) and label it with the appropriate event type from the given set.",
        "Read through the text and spot the event triggers which are expressions that most unambiguously represent events. As an event detection assistant, extract these event triggers and match them with their respective event types based on the predefined categories.",
        "You are tasked with being an event detection assistant. Scan the given text to locate event triggers, which are words or phrases that most clearly denote events. For each detected event trigger, determine its event type from the provided list and output the trigger along with its corresponding type.",
        "Go through the passage and recognize the event triggers which are terms that most explicitly indicate the presence of events. In your role as an event detection assistant, classify these event triggers into their relevant event types and generate a list containing the triggers and their assigned categories.",
        "As an event detection assistant, identify the event triggers in the text, which are keywords that most unambiguously suggest the occurrence of specific events. Map each event trigger to one of the predefined event types and create an output featuring the triggers and their associated types.",
        "Inspect the given text for event triggers which are words or phrases that most explicitly signal events. As an event detection assistant, categorize each discovered event trigger into its appropriate event type and produce a result that includes the trigger expressions and their corresponding types.",
        "You are an event detection assistant. Detect the presence of event triggers which are words or phrases that most clearly describe events within the provided text. For each trigger identified, establish its event type based on the predefined categories and present the trigger along with its assigned type.",
        "Examine the passage to uncover event triggers, which are terms that most unambiguously indicate events. In your capacity as an event detection assistant, assign each event trigger to one of the given event types and generate an output that lists the triggers and their respective categories.",
        "As an event detection assistant, analyze the text to find event triggers which are specific words or phrases that most explicitly suggest the existence of events. Determine the event type for each trigger based on the provided categories and create a result that showcases the triggers alongside their corresponding types.",
        "Your role is to be an event detection assistant. Study the given text and isolate the event triggers, which are expressions that most clearly imply the occurrence of events. Sort each event trigger into its designated event type and compile a list of the triggers with their assigned types.",
        "Act as an event detection assistant and scrutinize the passage for event triggers which are indicators that most explicitly denote events. Identify the event triggers and align them with their appropriate event types based on the predefined categories. Present your findings as a list of triggers and their corresponding types.",
        "As an event detection assistant, your objective is to pinpoint the event triggers in the text, which are words or phrases that most unambiguously signify events. Classify each event trigger into one of the given event types and generate an output that displays the triggers alongside their associated types.",
        "In your function as an event detection assistant, review the provided text and highlight the event triggers which are terms that most clearly denote events. Assign each event trigger to its relevant event type and produce a result that showcases the triggers and their corresponding categories.",
        "You are an event detection assistant tasked with identifying event triggers which are words or phrases that most explicitly describe events within the given text. Determine the event type for each trigger based on the predefined set and create an output that lists the triggers along with their assigned types.",
        "As an event detection assistant, evaluate the passage to discover event triggers, which are expressions that most unambiguously indicate events. Categorize each event trigger into one of the provided event types and compile a list that includes the triggers and their respective types.",
        "In your role as an event detection assistant, examine the text to locate event triggers which are specific words or phrases that most explicitly imply events. Map each event trigger to its appropriate event type based on the given categories and generate a result that presents the triggers and their corresponding types."
    ]
    return instructions


def create_preference_dataset(input_file, output_file):
    instruction_variations = generate_instruction_variations()

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            text = data['text']
            golden_labels = data['golden_labels']
            model_labels = data['model_labels']

            if golden_labels != model_labels:
                instruction = random.choice(instruction_variations)
                prompt = f"[INST]{instruction}\n\nText: {text}[/INST]"
                chosen = "; ".join([f"\n\nTrigger: {label['trigger']}, Event Type: {label['event_type']}" for label in golden_labels]) + "<|eot_id|>"
                rejected = "; ".join([f"\n\nTrigger: {label['trigger']}, Event Type: {label['event_type']}" for label in model_labels]) + "<|eot_id|>"

                preference_data = {
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                }
                f_out.write(json.dumps(preference_data) + '\n')

# # Usage
input_file = 'instruct_inference/Llama3-MAVEN-dev-responses.jsonl'
output_file = 'pref_data/Llama3-MAVEN-pref-data.jsonl'
create_preference_dataset(input_file, output_file)

# names = ["M2E2", "CASIE", "MLEE", "MAVEN", "FewEvent"]
# for name in names:
#     input_file = f'instruct_inference/Llama3-{name}-dev-responses.jsonl'
#     output_file = f'pref_data/Llama3-{name}--pref-data.jsonl'
#     create_preference_dataset(input_file, output_file)