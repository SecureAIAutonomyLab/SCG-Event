import os
import json
from sentence_transformers import SentenceTransformer

def process_jsonl_files(folder_path):
    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_processed.jsonl")

            # Open the input and output files
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                # Iterate over each line in the input file
                for line in infile:
                    # Parse the JSON object
                    data = json.loads(line)

                    # Get the text field
                    text = data['text']

                    # Compute the embedding
                    embedding = model.encode(text).tolist()

                    # Add the new field to the JSON object
                    data['all-MiniLM-L6-v2_embedding'] = embedding

                    # Write the updated JSON object to the output file
                    json.dump(data, outfile)
                    outfile.write('\n')

            print(f"Processed {filename} and saved as {output_file}")

# Example usage
folder_path = './phee/'
process_jsonl_files(folder_path)