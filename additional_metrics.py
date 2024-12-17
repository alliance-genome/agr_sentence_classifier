import pandas as pd
import glob
import os
import sys
import statistics
import argparse
import logging

# ------------------------------
# Configuration
# ------------------------------

TSV_PATTERN = 'classification_results_*_run*.tsv'

TASKS = {
    "task1_fully_curatable": "This sentence contains both fully and partially curatable data as well as terms related to curation.",
    "task2_partially_curatable": "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.",
    "task3_language_related": "This sentence does not contain fully or partially curatable data but does contain terms related to curation.",
    "task4_not_curatable": "This sentence does not contain fully or partially curatable data or terms related to curation."
}

# ------------------------------
# Helper Functions
# ------------------------------

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def determine_task(response):
    response = response.strip().lower()
    for task_name, exact_phrase in TASKS.items():
        if exact_phrase.lower() in response:
            return task_name
    return "Unknown"

def process_tsv_file(file_path, verbose=False):
    counts = {task: {"TP": 0, "FP": 0, "FN": 0} for task in TASKS.keys()}
    skipped_rows = 0
    try:
        df = pd.read_csv(file_path, sep='\t')
        logging.info(f"Processing file: {file_path} with {len(df)} entries.")
    except FileNotFoundError:
        logging.error(f"Error: File {file_path} not found.")
        return counts
    except pd.errors.EmptyDataError:
        logging.error(f"Error: File {file_path} is empty.")
        return counts
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return counts

    for index, row in df.iterrows():
        expected = row['expected_response']
        assistant = row['assistant_response']
        expected_task = determine_task(expected)
        predicted_task = determine_task(assistant)

        if expected_task == "Unknown" or predicted_task == "Unknown":
            logging.warning(f"Row {index+1}: Skipped due to Unknown task.")
            skipped_rows += 1
            continue

        for task_name in TASKS.keys():
            if expected_task == task_name:
                if predicted_task == task_name:
                    counts[task_name]["TP"] += 1
                    classification = "TP"
                else:
                    counts[task_name]["FN"] += 1
                    classification = "FN"
            else:
                if predicted_task == task_name:
                    counts[task_name]["FP"] += 1
                    classification = "FP"
                # No need to track TN as it's not used

            if verbose:
                logging.info(f"Task: {task_name.replace('_', ' ').title()}")
                logging.info(f"  Expected: \"{expected}\"")
                logging.info(f"  Assistant: \"{assistant}\"")
                logging.info(f"  Classification: {classification}")
                logging.info("-" * 60)

    if skipped_rows > 0:
        logging.info(f"Total Skipped Rows in {file_path}: {skipped_rows}")

    return counts

def calculate_run_metrics(counts):
    run_metrics = {}
    for task_name, count in counts.items():
        tp = count["TP"]
        fp = count["FP"]
        fn = count["FN"]
        precision, recall, f1_score = calculate_metrics(tp, fp, fn)
        run_metrics[task_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score
        }
    return run_metrics

def calculate_mean_std(metrics_per_run, data_type, verbose=False):
    summary = {}
    for task_name in TASKS.keys():
        precisions = [run_metrics[task_name]["Precision"] for run_metrics in metrics_per_run]
        recalls = [run_metrics[task_name]["Recall"] for run_metrics in metrics_per_run]
        f1_scores = [run_metrics[task_name]["F1_Score"] for run_metrics in metrics_per_run]

        if verbose and len(metrics_per_run) > 0:
            logging.info(f"\nDetailed Math for {task_name.replace('_',' ').title()} in {data_type}:")
            for i, run_metrics_dict in enumerate(metrics_per_run, 1):
                tp = run_metrics_dict[task_name]["TP"]
                fp = run_metrics_dict[task_name]["FP"]
                fn = run_metrics_dict[task_name]["FN"]

                # Log per-run counts
                logging.info(f"Run {i}: TP={tp}, FP={fp}, FN={fn}")

                # Precision math
                if tp + fp > 0:
                    logging.info(f"  Precision (Run {i}) = TP / (TP + FP) = {tp} / ({tp}+{fp}) = {tp/(tp+fp):.3f}")
                else:
                    logging.info(f"  Precision (Run {i}) = 0 (no positive predictions)")

                # Recall math
                if tp + fn > 0:
                    logging.info(f"  Recall (Run {i}) = TP / (TP + FN) = {tp} / ({tp}+{fn}) = {tp/(tp+fn):.3f}")
                else:
                    logging.info(f"  Recall (Run {i}) = 0 (no actual positives)")

                # F1 Score math
                precision_val = run_metrics_dict[task_name]["Precision"]
                recall_val = run_metrics_dict[task_name]["Recall"]
                if (precision_val + recall_val) > 0:
                    logging.info(f"  F1 Score (Run {i}) = 2 * Precision * Recall / (Precision + Recall) = 2 * {precision_val:.3f} * {recall_val:.3f} / ({precision_val:.3f}+{recall_val:.3f}) = {f1_scores[i-1]:.3f}")
                else:
                    logging.info(f"  F1 Score (Run {i}) = 0 (no positive predictions or actual positives)")

            # Compute mean and std dev for each metric
            # Precision
            if len(precisions) > 0:
                mean_p = statistics.mean(precisions)
                logging.info(f"\nPrecision Mean = Average of {precisions} = {mean_p:.3f}")
                if len(precisions) > 1:
                    std_p = statistics.stdev(precisions)
                    logging.info(f"Precision Std Dev = Stdev of {precisions} = {std_p:.3f}")
                else:
                    std_p = 0
                    logging.info("Only one run, no Std Dev for Precision.")
            else:
                mean_p = 0
                std_p = 0
                logging.info("No precision values.")

            # Recall
            if len(recalls) > 0:
                mean_r = statistics.mean(recalls)
                logging.info(f"\nRecall Mean = Average of {recalls} = {mean_r:.3f}")
                if len(recalls) > 1:
                    std_r = statistics.stdev(recalls)
                    logging.info(f"Recall Std Dev = Stdev of {recalls} = {std_r:.3f}")
                else:
                    std_r = 0
                    logging.info("Only one run, no Std Dev for Recall.")
            else:
                mean_r = 0
                std_r = 0
                logging.info("No recall values.")

            # F1 Score
            if len(f1_scores) > 0:
                mean_f1 = statistics.mean(f1_scores)
                logging.info(f"\nF1 Score Mean = Average of {f1_scores} = {mean_f1:.3f}")
                if len(f1_scores) > 1:
                    std_f1 = statistics.stdev(f1_scores)
                    logging.info(f"F1 Score Std Dev = Stdev of {f1_scores} = {std_f1:.3f}")
                else:
                    std_f1 = 0
                    logging.info("Only one run, no Std Dev for F1.")
            else:
                mean_f1 = 0
                std_f1 = 0
                logging.info("No F1 values.")

        else:
            # If not verbose or no runs
            mean_p = statistics.mean(precisions) if precisions else 0
            std_p = statistics.stdev(precisions) if len(precisions) > 1 else 0
            mean_r = statistics.mean(recalls) if recalls else 0
            std_r = statistics.stdev(recalls) if len(recalls) > 1 else 0
            mean_f1 = statistics.mean(f1_scores) if f1_scores else 0
            std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0

        summary[task_name] = {
            "Precision_mean": round(mean_p, 3),
            "Precision_std_dev": round(std_p, 3),
            "Recall_mean": round(mean_r, 3),
            "Recall_std_dev": round(std_r, 3),
            "F1_Score_mean": round(mean_f1, 3),
            "F1_Score_std_dev": round(std_f1, 3)
        }

    return summary

def print_metrics_summary(metrics_summary, data_type):
    logging.info(f"\nMetrics Summary for Data Type: {data_type}")
    print(f"\nMetrics Summary for Data Type: {data_type}")
    for task_name, stats in metrics_summary.items():
        task_display_name = task_name.replace('_', ' ').title()
        logging.info(f"\n{task_display_name}:")
        print(f"\n{task_display_name}:")
        logging.info(f"  Precision: {stats['Precision_mean']} (± {stats['Precision_std_dev']})")
        logging.info(f"  Recall: {stats['Recall_mean']} (± {stats['Recall_std_dev']})")
        logging.info(f"  F1 Score: {stats['F1_Score_mean']} (± {stats['F1_Score_std_dev']})")

        print(f"  Precision: {stats['Precision_mean']} (± {stats['Precision_std_dev']})")
        print(f"  Recall: {stats['Recall_mean']} (± {stats['Recall_std_dev']})")
        print(f"  F1 Score: {stats['F1_Score_mean']} (± {stats['F1_Score_std_dev']})")
    logging.info("")
    print("")

def save_metrics_summary(metrics_summary, data_type):
    rows = []
    for task_name, stats in metrics_summary.items():
        row = {
            "Task": task_name.replace('_', ' ').title(),
            "Precision_mean": stats["Precision_mean"],
            "Precision_std_dev": stats["Precision_std_dev"],
            "Recall_mean": stats["Recall_mean"],
            "Recall_std_dev": stats["Recall_std_dev"],
            "F1_Score_mean": stats["F1_Score_mean"],
            "F1_Score_std_dev": stats["F1_Score_std_dev"]
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    output_metrics_file_path = f'additional_metrics_summary_{data_type.replace(" ", "_")}.tsv'
    df_summary.to_csv(output_metrics_file_path, sep='\t', index=False)
    logging.info(f"Successfully saved metrics summary to {output_metrics_file_path}\n")
    print(f"Successfully saved metrics summary to {output_metrics_file_path}\n")

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler('metrics_calculation.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Calculate Precision, Recall, and F1 Score for classification tasks.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode for detailed output.')
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        logging.info("Verbose mode enabled.\n")
        print("Verbose mode enabled.\n")

    tsv_files = glob.glob(os.path.join('.', TSV_PATTERN))

    if not tsv_files:
        logging.error("No TSV files found matching the pattern.")
        print("No TSV files found matching the pattern.")
        sys.exit(1)

    data_types = {}
    for file_path in tsv_files:
        filename = os.path.basename(file_path)
        try:
            parts = filename.split('_')
            run_part = parts[-1]  # 'run1.tsv'
            run_number = run_part.split('.')[0]  # 'run1'
            data_type_parts = parts[2:-1]
            data_type = ' '.join(data_type_parts).title()
            logging.debug(f"Extracted data_type: {data_type} from filename: {filename}")
            if data_type not in data_types:
                data_types[data_type] = []
            data_types[data_type].append(file_path)
        except Exception as e:
            logging.error(f"Error parsing filename {filename}: {e}")
            print(f"Error parsing filename {filename}: {e}")
            continue

    metrics_per_data_type = {data_type: [] for data_type in data_types.keys()}

    for data_type, files in data_types.items():
        logging.info(f"\nProcessing Data Type: {data_type}")
        print(f"\nProcessing Data Type: {data_type}")

        for file_path in files:
            counts = process_tsv_file(file_path, verbose)
            run_metrics = calculate_run_metrics(counts)
            metrics_per_data_type[data_type].append(run_metrics)

        if metrics_per_data_type[data_type]:
            metrics_summary = calculate_mean_std(metrics_per_data_type[data_type], data_type, verbose=verbose)
            print_metrics_summary(metrics_summary, data_type)
            save_metrics_summary(metrics_summary, data_type)
        else:
            logging.warning(f"No valid runs processed for {data_type}.")
            print(f"No valid runs processed for {data_type}.")

    logging.info("All metrics have been calculated and saved successfully.")
    print("All metrics have been calculated and saved successfully.")

if __name__ == "__main__":
    main()
