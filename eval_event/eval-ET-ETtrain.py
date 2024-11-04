import json
from collections import defaultdict
import os

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def evaluate_event_classification(jsonl_file):
    classification_tp = classification_fp = classification_fn = 0
    event_types = set()
    golden_event_types = []
    model_event_types = []
    event_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with open(jsonl_file, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSON decode error in {jsonl_file} at line {line_number}: {line}")
                continue

            golden_labels = data['golden_labels']
            model_labels = data['model_labels']

            if isinstance(golden_labels, list) and all(isinstance(label, str) for label in golden_labels):
                golden_event_types.extend(golden_labels)
            else:
                print(f"Invalid format in golden_labels at line {line_number}: {golden_labels}")
                continue

            if isinstance(model_labels, list) and all(isinstance(label, str) for label in model_labels):
                model_event_types.extend(model_labels)
            else:
                print(f"Invalid format in model_labels at line {line_number}: {model_labels}")
                continue

            event_types.update(golden_event_types)
            event_types.update(model_event_types)

    golden_event_type_counts = defaultdict(int)
    model_event_type_counts = defaultdict(int)

    for event_type in golden_event_types:
        golden_event_type_counts[event_type] += 1

    for event_type in model_event_types:
        model_event_type_counts[event_type] += 1

    for event_type in event_types:
        tp = min(golden_event_type_counts[event_type], model_event_type_counts[event_type])
        fp = max(0, model_event_type_counts[event_type] - golden_event_type_counts[event_type])
        fn = max(0, golden_event_type_counts[event_type] - model_event_type_counts[event_type])

        event_type_metrics[event_type]["tp"] = tp
        event_type_metrics[event_type]["fp"] = fp
        event_type_metrics[event_type]["fn"] = fn

        classification_tp += tp
        classification_fp += fp
        classification_fn += fn

    classification_precision, classification_recall, classification_f1 = calculate_metrics(
        classification_tp, classification_fp, classification_fn
    )

    event_type_results = {}
    for event_type, metrics in event_type_metrics.items():
        precision, recall, f1 = calculate_metrics(metrics["tp"], metrics["fp"], metrics["fn"])
        event_type_results[event_type] = {
            "f1_score": f1
        }

    return {
        "file_name": os.path.basename(jsonl_file),
        "event_classification": {
            "f1_score": classification_f1
        },
    }

dataset_trains = ["CASIE", "FewEvent", "M2E2", "MAVEN", "MLEE"]
datasets_tests = [ds.lower() for ds in dataset_trains]

for train in dataset_trains:
    test_file = f"{train.lower()}/Llama3-{train}-ET.jsonl"
    try:
        result = evaluate_event_classification(test_file)
        print(result)
        print("\n\n")
    except FileNotFoundError:
        print(f"File not found: {test_file}\n\n")
        continue
    print("========================================================================")
