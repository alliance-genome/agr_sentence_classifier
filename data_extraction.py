import pandas as pd
import json

# Configuration variables
source_filter = ['GOLD', '1000']
fully_curable_filter = [0, 1]
partially_curable_filter = [0, 1]
related_language_filter = [0, 1]
training_or_testing_filter = 'TRAINING'
type_of_data = 'expression data'

# File paths
input_file_path = 'expression_sentence_datasets.tsv'
output_file_path = 'fine_tuned_training_data_expression.jsonl'

# Load the TSV file
df = pd.read_csv(input_file_path, sep='\t')

# Filter the DataFrame based on the configuration
filtered_df = df[
    (df['SOURCE'].isin(source_filter)) &
    (df['FULLY_CURATABLE'].isin(fully_curable_filter)) &
    (df['PARTIALLY_CURATABLE'].isin(partially_curable_filter)) &
    (df['RELATED_LANGUAGE'].isin(related_language_filter)) &
    (df['TRAINING_OR_TESTING'] == training_or_testing_filter)
]

# Function to create the assistant content based on curatable values
def create_assistant_content(row):
    fully = row['FULLY_CURATABLE'] == 1
    partially = row['PARTIALLY_CURATABLE'] == 1
    related = row['RELATED_LANGUAGE'] == 1
    
    if fully and partially:
        if related:
            return "This sentence contains both fully and partially curatable data as well as terms related to curation."
        else:
            return "This sentence contains both fully and partially curatable data."
    elif fully and not partially:
        if related:
            return "This sentence contains fully curatable data and terms related to curation."
        else:
            return "This sentence contains fully curatable data and does not contain partially curatable data."
    elif not fully and partially:
        if related:
            return "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."
        else:
            return "This sentence does not contain fully curatable data but it does contain partially curatable data."
    elif not fully and not partially:
        if related:
            return "This sentence does not contain fully or partially curatable data but does contain terms related to curation."
        else:
            return "This sentence does not contain fully or partially curatable data or terms related to curation."

# Create the structured data
structured_data = []
for _, row in filtered_df.iterrows():
    entry = {
        "messages": [
            {"role": "system", "content": f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."},
            {"role": "user", "content": row['SENTENCE']},
            {"role": "assistant", "content": create_assistant_content(row)}
        ]
    }
    structured_data.append(entry)

# Save the structured data to a JSONL file
with open(output_file_path, 'w') as output_file:
    for entry in structured_data:
        output_file.write(json.dumps(entry) + '\n')

print(f"Structured data has been saved to {output_file_path}")
