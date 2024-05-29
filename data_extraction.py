import pandas as pd
import json

# Configuration variables
source_filter = ['GOLD', '1000']
fully_curable_filter = [0, 1]
partially_curable_filter = [0, 1]
related_language_filter = [0, 1]

type_of_data = 'protein kinase activity'
type_of_data_filename = type_of_data.replace(' ', '_')

# File paths
input_file_path = 'kinaseact_sentence_datasets.tsv'
training_output_file_path = f'fine_tuned_training_data_expression_{type_of_data_filename}.jsonl'
testing_output_file_path = f'fine_tuned_testing_data_expression_{type_of_data_filename}.jsonl'

# Load the TSV file
df = pd.read_csv(input_file_path, sep='\t')

# Function to create the assistant content based on curatable values
def create_assistant_content(row):
    fully = row['FULLY_CURATABLE'] == 1
    partially = row['PARTIALLY_CURATABLE'] == 1
    related = row['RELATED_LANGUAGE'] == 1
    
    if fully and partially:
        if related:
            return "This sentence contains both fully and partially curatable data as well as terms related to curation."
        else:
            raise ValueError(f"Unexpected combination: fully and partially curatable but not related. Row: {row}")
    elif fully and not partially:
        raise ValueError(f"Unexpected combination: fully curatable but not partially curatable. Row: {row}")
    elif not fully and partially:
        if related:
            return "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."
        else:
            raise ValueError(f"Unexpected combination: not fully but partially curatable and not related. Row: {row}")
    elif not fully and not partially:
        if related:
            return "This sentence does not contain fully or partially curatable data but does contain terms related to curation."
        else:
            return "This sentence does not contain fully or partially curatable data or terms related to curation."
    else:
        raise ValueError(f"Unexpected combination: Row: {row}")

# Function to filter the DataFrame and create structured data
def filter_and_create_structured_data(df, training_or_testing_filter):
    filtered_df = df[
        (df['SOURCE'].isin(source_filter)) &
        (df['FULLY_CURATABLE'].isin(fully_curable_filter)) &
        (df['PARTIALLY_CURATABLE'].isin(partially_curable_filter)) &
        (df['RELATED_LANGUAGE'].isin(related_language_filter)) &
        (df['TRAINING_OR_TESTING'] == training_or_testing_filter)
    ]
    
    structured_data = []
    for _, row in filtered_df.iterrows():
        try:
            assistant_content = create_assistant_content(row)
        except ValueError as e:
            print(e)
            continue

        entry = {
            "messages": [
                {"role": "system", "content": f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."},
                {"role": "user", "content": row['SENTENCE']},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        structured_data.append(entry)
    
    return structured_data

# Generate training data
training_data = filter_and_create_structured_data(df, 'TRAINING')

# Save the training data to a JSONL file
with open(training_output_file_path, 'w') as output_file:
    for entry in training_data:
        output_file.write(json.dumps(entry) + '\n')

print(f"Training data has been saved to {training_output_file_path}")

# Generate testing data
testing_data = filter_and_create_structured_data(df, 'TESTING')

# Save the testing data to a JSONL file
with open(testing_output_file_path, 'w') as output_file:
    for entry in testing_data:
        output_file.write(json.dumps(entry) + '\n')

print(f"Testing data has been saved to {testing_output_file_path}")
