import json
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import re

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def evaluate_event_detection(jsonl_file):
    identification_tp = identification_fp = identification_fn = 0
    classification_tp = classification_fp = classification_fn = 0
    event_types = set()
    golden_event_types = []
    model_event_types = []

    with open(jsonl_file, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSON decode error in {jsonl_file} at line {line_number}: {line}")
                continue

            golden_labels = data['golden_labels']
            model_labels = data['model_labels']
            golden_triggers = set(label['trigger'] for label in golden_labels)
            try:
                model_triggers = set(label['trigger'] for label in model_labels)
            except TypeError:
                print(f"Error processing line: {line}")
                print("Skipping this line and continuing with the next one.")
                continue

            identification_tp += len(golden_triggers & model_triggers)
            identification_fp += len(model_triggers - golden_triggers)
            identification_fn += len(golden_triggers - model_triggers)

            golden_label_dict = {label['trigger']: label['event_type'] for label in golden_labels}
            model_label_dict = {label['trigger']: label['event_type'] for label in model_labels}

            for trigger in golden_triggers & model_triggers:
                golden_event_type = golden_label_dict[trigger]
                model_event_type = model_label_dict[trigger]
                event_types.add(golden_event_type)
                event_types.add(model_event_type)
                golden_event_types.append(golden_event_type)
                model_event_types.append(model_event_type)
                if golden_event_type == model_event_type:
                    classification_tp += 1
                else:
                    classification_fp += 1
                    classification_fn += 1

            classification_fp += len(model_triggers - golden_triggers)
            classification_fn += len(golden_triggers - model_triggers)

    identification_precision, identification_recall, identification_f1 = calculate_metrics(
        identification_tp, identification_fp, identification_fn
    )
    classification_precision, classification_recall, classification_f1 = calculate_metrics(
        classification_tp, classification_fp, classification_fn
    )

    return {
        "file_name": os.path.basename(jsonl_file),
        "trigger_identification": {
            # "precision": identification_precision,
            # "recall": identification_recall,
            "f1_score": identification_f1
        },
        "trigger_classification": {
            # "precision": classification_precision,
            # "recall": classification_recall,
            "f1_score": classification_f1
        }
    }

dataset_trains = ["CASIE", "FewEvent", "M2E2", "MAVEN", "MLEE"]
datasets_tests = [ds.lower() for ds in dataset_trains]

for train in dataset_trains:
    test_file = f"{train.lower()}/Gemma-{train}-alpaca-epoch6.jsonl"
    try:
        # Run the evaluation function
        result = evaluate_event_detection(test_file)
        print(result)
        print("\n\n")
    except FileNotFoundError:
        # If the file doesn't exist, print a message and continue
        print(f"File not found: {test_file}\n\n")
        continue
    print("========================================================================")

