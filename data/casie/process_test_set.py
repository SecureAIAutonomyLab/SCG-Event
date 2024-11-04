import json

def process_casie_test_set(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            data = json.loads(line)
            text = data['text']
            events = []

            for event_mention in data['event_mentions']:
                trigger = event_mention['trigger']['text']
                event_type = event_mention['event_type']
                events.append({"trigger": trigger, "event_type": event_type})

            if not events:
                events.append({"trigger": "None", "event_type": "None"})

            test_instance = {
                "text": text,
                "events": events
            }
            json.dump(test_instance, outfile)
            outfile.write('\n')

# Example usage
input_file_path = 'split1/test.json'
output_file_path = 'casie-test.jsonl'
process_casie_test_set(input_file_path, output_file_path)