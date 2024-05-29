import json
import pandas as pd

# File path to the JSONL file
file_path = 'fine_tuned_testing_data_expression.jsonl'

# The sentences to check
sentences_to_check = [
    "This sentence contains both fully and partially curatable data as well as terms related to curation.",
    "This sentence contains both fully and partially curatable data.",
    "This sentence contains fully curatable data and terms related to curation.",
    "This sentence contains fully curatable data and does not contain partially curatable data.",
    "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.",
    "This sentence does not contain fully curatable data but it does contain partially curatable data.",
    "This sentence does not contain fully or partially curatable data but does contain terms related to curation.",
    "This sentence does not contain fully or partially curatable data or terms related to curation."
]

# Load the JSONL data
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Check which sentences are in the file
sentences_in_file = {sentence: False for sentence in sentences_to_check}

for entry in data:
    for message in entry['messages']:
        if message['role'] == 'assistant' and message['content'] in sentences_in_file:
            sentences_in_file[message['content']] = True

# Set display options to avoid clipping
pd.set_option('display.max_colwidth', None)

# Create a DataFrame to display the results
df = pd.DataFrame(list(sentences_in_file.items()), columns=['Sentence', 'In File'])
print(df)
