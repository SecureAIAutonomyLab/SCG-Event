import json
import random
import os

def generate_instruction_variations():
    instructions = [
        "As an event detection assistant, your task is to identify the event types present in the given text. Classify the text into its corresponding event type(s) from the predefined set.",
        "In your role as an event detection assistant, categorize the text into its respective event type(s).",
        "You are an event detection assistant. Determine the corresponding event type(s) for the provided text from the given categories and output the associated types.",
        "Analyze the given text and, as an event detection assistant, classify it into one or more of the predefined event types.",
        "In your capacity as an event detection assistant, examine the provided passage and specify the relevant event type(s) from the given set.",
        "Read through the text and, as an event detection assistant, match it with the appropriate event type(s) based on the predefined categories.",
        "You are tasked with being an event detection assistant. Scan the given text to determine its event type(s) from the provided list.",
        "Go through the passage and, in your role as an event detection assistant, classify it into the relevant event type(s) and generate a list containing the assigned categories.",
        "As an event detection assistant, identify the event types in the text. Map the text to one or more of the predefined event types and create an output featuring the associated types.",
        "Inspect the given text and, as an event detection assistant, categorize it into its appropriate event type(s) and produce a result that includes the corresponding types.",
        "You are an event detection assistant. Detect the presence of event types within the provided text. Establish the text's event type(s) based on the predefined categories and present the assigned types.",
        "Examine the passage and, in your capacity as an event detection assistant, assign it to one or more of the given event types and generate an output that lists the respective categories.",
        "As an event detection assistant, analyze the text to determine its event type(s) based on the provided categories and create a result that showcases the corresponding types.",
        "Your role is to be an event detection assistant. Study the given text and sort it into its designated event type(s) and compile a list of the assigned types.",
        "Act as an event detection assistant and scrutinize the passage to identify its appropriate event type(s) based on the predefined categories. Present your findings as a list of corresponding types.",
        "As an event detection assistant, your objective is to classify the text into one or more of the given event types and generate an output that displays the associated types.",
        "In your function as an event detection assistant, review the provided text and assign it to its relevant event type(s) and produce a result that showcases the corresponding categories.",
        "You are an event detection assistant tasked with determining the event type(s) for the given text based on the predefined set and creating an output that lists the assigned types.",
        "As an event detection assistant, evaluate the passage to categorize it into one or more of the provided event types and compile a list that includes the respective types.",
        "In your role as an event detection assistant, examine the text to map it to its appropriate event type(s) based on the given categories and generate a result that presents the corresponding types."
    ]
    return instructions

def process_casie_dataset(input_file, output_file):
    instruction_variations = generate_instruction_variations()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            data = json.loads(line)
            text = data['text']
            event_types = [event_mention['event_type'] for event_mention in data['event_mentions']]

            instruction = random.choice(instruction_variations)
            input_text = f"Text: {text}"

            if event_types:
                output_text = '\n'.join(f"Event Type: {event_type}" for event_type in event_types)
            else:
                output_text = "Event Type: None"

            instruction_data_point = {
                'text': f"<s>[INST]{instruction}\n\n{input_text}[/INST]\n\n{output_text}</s>"
            }
            json.dump(instruction_data_point, outfile)
            outfile.write('\n')

input_file_path = 'split1/train.json'
output_file_path = 'fewevent-instruct-ET.jsonl'
process_casie_dataset(input_file_path, output_file_path)