import pandas as pd
import json

# Load the JSONL file with all possible prompts
jsonl_file_path = 'fine_tuned_testing_data_expression_gene_expression.jsonl'
prompts = []

with open(jsonl_file_path, 'r') as file:
    for line in file:
        prompts.append(json.loads(line))

print(f"Total prompts in JSONL file: {len(prompts)}")

# Load the TSV file with results
tsv_file_path = 'classification_results_gene_expression_4.tsv'
results_df = pd.read_csv(tsv_file_path, sep='\t')

print(f"Total results in TSV file: {len(results_df)}")

# Extract prompts from the results TSV
completed_prompts = list(results_df['sentence'])

print(f"Total unique sentences in TSV file: {len(completed_prompts)}")

# Check if any prompts are missing in the results
missing_prompts = []
for prompt in prompts:
    sentence = prompt['messages'][1]['content']
    if sentence not in completed_prompts:
        missing_prompts.append(sentence)

print(f"Total missing prompts: {len(missing_prompts)}")

if missing_prompts:
    print("List of missing prompts:")
    for mp in missing_prompts:
        print(mp)

# Check for early termination by comparing number of entries and order in JSONL and TSV files
early_termination = False
if len(results_df) < len(prompts):
    print("Potential early termination detected based on number of entries.")
    
    # Check if the entries match in order up to the length of the TSV file
    for i in range(len(results_df)):
        if prompts[i]['messages'][1]['content'] != completed_prompts[i]:
            early_termination = True
            print(f"Mismatch found at index {i}:")
            print(f"  JSONL prompt: {prompts[i]['messages'][1]['content']}")
            print(f"  TSV result: {completed_prompts[i]}")
            break
    if not early_termination:
        print("Entries match in order up to the length of the TSV file.")
else:
    print("No early termination based on number of entries.")

if early_termination:
    print("Early termination detected due to order mismatch.")
else:
    print("No early termination detected.")

# Check for duplicates in the TSV file
duplicates = results_df[results_df.duplicated(subset=['sentence'], keep=False)]

if not duplicates.empty:
    print("Duplicate entries found in TSV file:")
    print(duplicates)
else:
    print("No duplicate entries found in TSV file.")

print("Debugging complete.")
