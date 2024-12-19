#!/usr/bin/env python

import pandas as pd
import os
import sys
import argparse
import statistics

# ------------------------------
# Configuration
# ------------------------------

# Define the data types and their corresponding TSV files for runs 1 to 5
DATA_TYPES = {
    "Gene Expression": [
        "final_classification_results_gene_expression_run1.tsv",
        "final_classification_results_gene_expression_run2.tsv",
        "final_classification_results_gene_expression_run3.tsv",
        "final_classification_results_gene_expression_run4.tsv",
        "final_classification_results_gene_expression_run5.tsv"
    ],
    "Protein Kinase": [
        "final_classification_results_protein_kinase_activity_run1.tsv",
        "final_classification_results_protein_kinase_activity_run2.tsv",
        "final_classification_results_protein_kinase_activity_run3.tsv",
        "final_classification_results_protein_kinase_activity_run4.tsv",
        "final_classification_results_protein_kinase_activity_run5.tsv"
    ]
}

# Define tasks with their corresponding response phrases
TASKS = {
    "Task1_Fully_Curatable": [
        "This sentence only contains fully curatable data."
    ],
    "Task2_Fully_or_Partially_Curatable": [
        "This sentence only contains fully curatable data.",
        "This sentence only contains partially curatable data."
    ],
    "Task3_Fully_Partially_or_Language_Related": [
        "This sentence only contains fully curatable data.",
        "This sentence only contains partially curatable data.",
        "This sentence is not fully or partially curatable, but it contains terms related to the datatype."
    ]
}

# Define exact response phrase to label mapping
RESPONSE_LABELS = {
    "This sentence only contains fully curatable data.": "fully curated",
    "This sentence only contains partially curatable data.": "partially curated",
    "This sentence is not fully or partially curatable, but it contains terms related to the datatype.": "language related"
}

# ------------------------------
# Helper Functions
# ------------------------------

def calculate_metrics(tp, fp, fn):
    """
    Calculate precision, recall, and F1-score.

    Args:
        tp (int): True Positives
        fp (int): False Positives
        fn (int): False Negatives

    Returns:
        tuple: precision, recall, f1_score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def determine_task(response, task_responses):
    """
    Determine if the response belongs to the task based on predefined responses.

    Args:
        response (str): The response string to check.
        task_responses (list): List of response strings associated with the task.

    Returns:
        bool: True if response matches any in task_responses, else False.
    """
    return response.strip() in task_responses

def get_response_label(response):
    """
    Map a response phrase to its corresponding label using exact matching.

    Args:
        response (str): The response string.

    Returns:
        str: The label corresponding to the response.
    """
    return RESPONSE_LABELS.get(response.strip(), "unknown")

def process_tsv_file(file_path, verbose=False):
    """
    Process a single TSV file and calculate metrics for each task.

    Args:
        file_path (str): Path to the TSV file.
        verbose (bool): If True, print detailed logs.

    Returns:
        tuple: run_metrics (dict), counts (dict), negative_count (int)
    """
    # Initialize metrics and counts
    metrics = {task: {"TP": 0, "FP": 0, "FN": 0} for task in TASKS.keys()}
    counts = {task: {response: 0 for response in TASKS[task]} for task in TASKS.keys()}
    negative_count = 0
    skipped_rows = 0

    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Processing file: {file_path} with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return metrics, counts, negative_count
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty.")
        return metrics, counts, negative_count
    except pd.errors.ParserError as e:
        print(f"Error parsing {file_path}: {e}")
        return metrics, counts, negative_count

    for index, row in df.iterrows():
        expected = row.get('expected_response', '').strip()
        assistant = row.get('assistant_response', '').strip()

        if not expected or not assistant:
            print(f"Row {index+1}: Missing expected or assistant response. Skipping.")
            skipped_rows += 1
            continue

        matched_any_task = False

        # Iterate through each task to update counts
        for task_name, responses in TASKS.items():
            expected_in_task = determine_task(expected, responses)
            assistant_in_task = determine_task(assistant, responses)

            if expected_in_task:
                # Identify which response phrase it matched
                counts[task_name][expected] += 1
                matched_any_task = True

            if expected_in_task and assistant_in_task:
                metrics[task_name]["TP"] += 1
            elif not expected_in_task and assistant_in_task:
                metrics[task_name]["FP"] += 1
            elif expected_in_task and not assistant_in_task:
                metrics[task_name]["FN"] += 1

            if verbose:
                classification = ""
                if expected_in_task and assistant_in_task:
                    classification = "TP"
                elif not expected_in_task and assistant_in_task:
                    classification = "FP"
                elif expected_in_task and not assistant_in_task:
                    classification = "FN"
                print(f"Row {index+1}:")
                print(f"  Sentence: \"{row.get('sentence', '').strip()}\"")
                print(f"  Expected: \"{expected}\"")
                print(f"  Assistant: \"{assistant}\"")
                print(f"  Task: {task_name}")
                print(f"  Classification: {classification}")
                print("-" * 60)

        if not matched_any_task:
            negative_count += 1

    if skipped_rows > 0:
        print(f"Total Skipped Rows in {file_path}: {skipped_rows}")

    # Calculate metrics for each task
    run_metrics = {}
    for task_name, counts_task in metrics.items():
        tp = counts_task["TP"]
        fp = counts_task["FP"]
        fn = counts_task["FN"]
        precision, recall, f1_score = calculate_metrics(tp, fp, fn)
        run_metrics[task_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score
        }

    # Informative Printout after processing the file
    print(f"\nCompleted processing {file_path}:")
    for task, stats in run_metrics.items():
        print(f"  {task.replace('_', ' ')} - Precision: {stats['Precision']:.3f}, Recall: {stats['Recall']:.3f}, F1 Score: {stats['F1_Score']:.3f}")
    print(f"  Negative Examples (Do not fit any task): {negative_count}")
    print("=" * 60 + "\n")

    return run_metrics, counts, negative_count

def calculate_mean_std(metrics_per_run, counts_per_run, negative_counts, data_type, verbose=False):
    """
    Calculate sum counts and mean and standard deviation for each metric across runs.

    Args:
        metrics_per_run (list): List of metrics dictionaries per run.
        counts_per_run (list): List of counts dictionaries per run.
        negative_counts (list): List of negative counts per run.
        data_type (str): The data type being processed.
        verbose (bool): If True, print detailed logs.

    Returns:
        dict: Summary of sum counts for each task's response phrases and averaged metrics.
    """
    summary = {}
    total_negative = sum(negative_counts)
    total_runs = len(metrics_per_run)

    for task_name in TASKS.keys():
        response_counts = {response: 0 for response in TASKS[task_name]}
        for run_counts in counts_per_run:
            for response_phrase in TASKS[task_name]:
                response_counts[response_phrase] += run_counts[task_name][response_phrase]

        # Calculate metrics mean and std dev
        precisions = [run_metrics[task_name]["Precision"] for run_metrics in metrics_per_run]
        recalls = [run_metrics[task_name]["Recall"] for run_metrics in metrics_per_run]
        f1_scores = [run_metrics[task_name]["F1_Score"] for run_metrics in metrics_per_run]

        mean_p = statistics.mean(precisions) if precisions else 0
        std_p = statistics.stdev(precisions) if len(precisions) > 1 else 0
        mean_r = statistics.mean(recalls) if recalls else 0
        std_r = statistics.stdev(recalls) if len(recalls) > 1 else 0
        mean_f1 = statistics.mean(f1_scores) if f1_scores else 0
        std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0

        summary[task_name] = {
            "Response_Counts": response_counts,
            "Entries_count_total": sum(response_counts.values()),
            "Precision_mean": round(mean_p, 3),
            "Precision_std_dev": round(std_p, 3),
            "Recall_mean": round(mean_r, 3),
            "Recall_std_dev": round(std_r, 3),
            "F1_Score_mean": round(mean_f1, 3),
            "F1_Score_std_dev": round(std_f1, 3)
        }

    # Calculate averages for negative examples
    mean_negatives = statistics.mean(negative_counts) if negative_counts else 0
    std_negatives = statistics.stdev(negative_counts) if len(negative_counts) > 1 else 0

    summary["Negative_Examples"] = {
        "Entries_count_mean": round(mean_negatives, 1),
        "Entries_count_std_dev": round(std_negatives, 1)
    }

    return summary

def print_metrics_summary(metrics_summary, data_type):
    """
    Print the metrics summary in a clear and structured manner.

    Args:
        metrics_summary (dict): Summary of metrics per task.
        data_type (str): The data type being processed.
    """
    print(f"\n{'='*60}")
    print(f"Metrics Summary for Data Type: {data_type}")
    print(f"{'='*60}")
    for task_name, stats in metrics_summary.items():
        if task_name != "Negative_Examples":
            task_display_name = task_name.replace('_', ' ').title()
            combined_types = len(TASKS[task_name])
            print(f"\n{task_display_name}:")
            print(f"  Combined {combined_types} types of responses to form {task_display_name}:")
            for response_phrase, count in stats["Response_Counts"].items():
                label = get_response_label(response_phrase)
                print(f"    Combined {count} entries from {label}.")
            print(f"  Entries Count Total: {stats['Entries_count_total']}")
            print(f"  Precision: {stats['Precision_mean']} (± {stats['Precision_std_dev']})")
            print(f"  Recall:    {stats['Recall_mean']} (± {stats['Recall_std_dev']})")
            print(f"  F1 Score:  {stats['F1_Score_mean']} (± {stats['F1_Score_std_dev']})")
        else:
            print(f"\nNegative Examples (Do not fit any task):")
            print(f"  Entries Count - Mean: {stats['Entries_count_mean']}, Std Dev: {stats['Entries_count_std_dev']}")
    print(f"{'='*60}\n")

def save_metrics_summary(metrics_summary, data_type):
    """
    Save the metrics summary to a TSV file.

    Args:
        metrics_summary (dict): Summary of metrics per task.
        data_type (str): The data type being processed.
    """
    rows = []
    for task_name, stats in metrics_summary.items():
        if task_name != "Negative_Examples":
            row = {
                "Task": task_name.replace('_', ' ').title(),
                "Precision_mean": stats["Precision_mean"],
                "Precision_std_dev": stats["Precision_std_dev"],
                "Recall_mean": stats["Recall_mean"],
                "Recall_std_dev": stats["Recall_std_dev"],
                "F1_Score_mean": stats["F1_Score_mean"],
                "F1_Score_std_dev": stats["F1_Score_std_dev"],
                "Entries_count_total": stats["Entries_count_total"]
            }
            # Add counts from each response type as separate columns
            for response_phrase, count in stats["Response_Counts"].items():
                label = get_response_label(response_phrase)
                column_name = f"Combined_{label.replace(' ', '_')}_count"
                row[column_name] = count
            rows.append(row)
        else:
            row = {
                "Task": "Negative Examples",
                "Precision_mean": "-",
                "Precision_std_dev": "-",
                "Recall_mean": "-",
                "Recall_std_dev": "-",
                "F1_Score_mean": "-",
                "F1_Score_std_dev": "-",
                "Entries_count_total": "-"
            }
            # Add counts from each response type as separate columns (none for negative examples)
            row["Combined_fully_curated_count"] = "-"
            row["Combined_partially_curated_count"] = "-"
            row["Combined_language_related_count"] = "-"
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    data_type_filename = data_type.replace(' ', '_')
    output_metrics_file_path = f'final_metrics_summary_{data_type_filename}.tsv'
    df_summary.to_csv(output_metrics_file_path, sep='\t', index=False)
    print(f"Successfully saved metrics summary to {output_metrics_file_path}\n")

def setup_logging():
    """
    Placeholder function since we're using print statements.
    """
    pass

def main():
    """
    Main function to parse arguments and process TSV files.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Precision, Recall, and F1 Score for classification tasks.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode for detailed output.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        print("Verbose mode enabled.\n")

    # Initialize metrics_per_data_type
    metrics_per_data_type = {data_type: [] for data_type in DATA_TYPES.keys()}
    counts_per_data_type = {data_type: [] for data_type in DATA_TYPES.keys()}
    negative_counts_per_data_type = {data_type: [] for data_type in DATA_TYPES.keys()}

    # Process each data type
    for data_type, files in DATA_TYPES.items():
        print(f"\n{'='*60}")
        print(f"Processing Data Type: {data_type}")
        print(f"{'='*60}\n")

        for run_number, file_path in enumerate(files, 1):
            print(f"Processing Run {run_number}: {file_path}")
            run_metrics, counts, negative_count = process_tsv_file(file_path, verbose)
            metrics_per_data_type[data_type].append(run_metrics)
            counts_per_data_type[data_type].append(counts)
            negative_counts_per_data_type[data_type].append(negative_count)

        # Calculate sum counts and mean/std dev for the data type
        if metrics_per_data_type[data_type]:
            metrics_summary = calculate_mean_std(
                metrics_per_run=metrics_per_data_type[data_type],
                counts_per_run=counts_per_data_type[data_type],
                negative_counts=negative_counts_per_data_type[data_type],
                data_type=data_type,
                verbose=verbose
            )
            print_metrics_summary(metrics_summary, data_type)
            save_metrics_summary(metrics_summary, data_type)
        else:
            print(f"No valid runs processed for {data_type}.")

    print("All metrics have been calculated and saved successfully.")

if __name__ == "__main__":
    main()
