import pandas as pd

# Define file path
type_of_data = 'gene expression' # Ensure this matches the file naming convention in your main script
type_of_data_filename = type_of_data.replace(' ', '_')  
output_file_path = f'classification_results_{type_of_data_filename}.tsv'

# Load the TSV file
df = pd.read_csv(output_file_path, sep='\t')

# Helper function to calculate precision, recall, and F1 score
def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    return precision, recall, f1_score, accuracy

# Initialize counters for the three classes
metrics = {
    "fully_curatable": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    "curatable": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    "language_related": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
}

# Initialize counters for combined metrics
combined_tp = 0
combined_fp = 0
combined_fn = 0
combined_tn = 0

# Define the conditions for each class
fully_curatable_condition = "This sentence contains both fully and partially curatable data as well as terms related to curation."
partially_curatable_condition = "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."
language_related_condition = "This sentence does not contain fully or partially curatable data but does contain terms related to curation."
not_curatable_condition = "This sentence does not contain fully or partially curatable data or terms related to curation."

# Calculate true positives, false positives, false negatives, and true negatives for each class
for index, row in df.iterrows():
    expected = row['expected_response']
    assistant = row['assistant_response']

    # Fully curatable
    if fully_curatable_condition in expected:
        if fully_curatable_condition in assistant:
            metrics["fully_curatable"]["tp"] += 1
        else:
            metrics["fully_curatable"]["fn"] += 1
    else:
        if fully_curatable_condition in assistant:
            metrics["fully_curatable"]["fp"] += 1
        else:
            metrics["fully_curatable"]["tn"] += 1

    # Curatable (fully or partially)
    if fully_curatable_condition in expected or partially_curatable_condition in expected:
        if fully_curatable_condition in assistant or partially_curatable_condition in assistant:
            metrics["curatable"]["tp"] += 1
        else:
            metrics["curatable"]["fn"] += 1
    else:
        if fully_curatable_condition in assistant or partially_curatable_condition in assistant:
            metrics["curatable"]["fp"] += 1
        else:
            metrics["curatable"]["tn"] += 1

    # Language related
    if language_related_condition in expected:
        if language_related_condition in assistant:
            metrics["language_related"]["tp"] += 1
        else:
            metrics["language_related"]["fn"] += 1
    else:
        if language_related_condition in assistant:
            metrics["language_related"]["fp"] += 1
        else:
            metrics["language_related"]["tn"] += 1

    # Combined metrics
    if expected == not_curatable_condition and assistant == not_curatable_condition:
        combined_tn += 1
    elif row['result_category'] == 'correct':
        if row['classification'] in ['true_positive', 'true_negative']:
            combined_tp += 1
        else:
            combined_tn += 1
    else:
        if row['classification'] == 'false_positive':
            combined_fp += 1
        else:
            combined_fn += 1

# Calculate and print the metrics for each class
for key in metrics:
    tp, fp, fn, tn = metrics[key]["tp"], metrics[key]["fp"], metrics[key]["fn"], metrics[key]["tn"]
    precision, recall, f1_score, accuracy = calculate_metrics(tp, fp, fn, tn)
    print(f"{key.replace('_', ' ').title()}:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1_score:.2f}")
    print(f"  True Positives: {tp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print()

# Calculate and print combined metrics
combined_precision, combined_recall, combined_f1_score, combined_accuracy = calculate_metrics(combined_tp, combined_fp, combined_fn, combined_tn)

print("Combined Metrics:")
print(f"  Accuracy: {combined_accuracy:.2f}")
print(f"  Precision: {combined_precision:.2f}")
print(f"  Recall: {combined_recall:.2f}")
print(f"  F1 Score: {combined_f1_score:.2f}")
print(f"  True Positives: {combined_tp}")
print(f"  True Negatives: {combined_tn}")
print(f"  False Positives: {combined_fp}")
print(f"  False Negatives: {combined_fn}")
print()

# Save the results to a new TSV file
output_metrics_file_path = f'additional_metrics_{type_of_data}.tsv'
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
metrics_df.to_csv(output_metrics_file_path, sep='\t', index=True)
