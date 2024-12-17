import json
import random
import pandas as pd
from openai import OpenAI  # Using the OpenAI client as originally provided
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sys
from tqdm import tqdm
import os
import concurrent.futures
import threading
import logging
import time

# ------------------------------
# Configuration for Multiple Data Types
# ------------------------------
DATA_TYPES = [
    {
        'type_of_data': 'gene expression',
        'input_file': 'expression_test.jsonl',
        'model_name': 'o1-preview-2024-09-12'
    },
    {
        'type_of_data': 'protein kinase activity',
        'input_file': 'kinase_test.jsonl',
        'model_name': 'o1-preview-2024-09-12'
    }
]

# ------------------------------
# Get the API key and verbosity flags from the command line arguments
# ------------------------------
def parse_arguments():
    if len(sys.argv) not in [2, 3, 4, 5, 6]:
        print("Usage: python script_name.py [-v] [-s] <your_openai_api_key>")
        print("Options:")
        print("  -v    Enable verbose mode to print detailed output.")
        print("  -s    Run a random subset of 10 entries from the testing data.")
        print("Arguments:")
        print("  <your_openai_api_key>    Your OpenAI API key for authentication.")
        sys.exit(1)
    
    verbose = False
    subset = False
    api_key_index = 1
    
    if len(sys.argv) >= 3:
        if '-v' in sys.argv:
            verbose = True
            api_key_index += 1
        if '-s' in sys.argv:
            subset = True
            api_key_index += 1
    
    if api_key_index >= len(sys.argv):
        print("Error: Missing OpenAI API key.")
        sys.exit(1)
    
    api_key = sys.argv[api_key_index]
    return api_key, verbose, subset

# ------------------------------
# Initialize the OpenAI client
# ------------------------------
def initialize_openai_client(api_key):
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

# ------------------------------
# Define the classification mapping function
# ------------------------------
def classify_sentence(content):
    expected_responses = {
        "This sentence contains both fully and partially curatable data as well as terms related to curation.": "curatable",
        "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.": "curatable",
        "This sentence does not contain fully or partially curatable data but does contain terms related to curation.": "not_curatable",
        "This sentence does not contain fully or partially curatable data or terms related to curation.": "not_curatable"
    }

    if content in expected_responses:
        return {"result": expected_responses[content]}
    else:
        raise ValueError(f"Unexpected response: {content}")

# ------------------------------
# Load the JSONL file
# ------------------------------
def load_jsonl(file_path):
    testing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                testing_data.append(json.loads(line))
        print(f"Loaded {len(testing_data)} entries from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as jde:
        print(f"JSON decode error in file {file_path}: {jde}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        sys.exit(1)
    return testing_data

# ------------------------------
# Utility function to make chat completion requests with retry logic
# ------------------------------
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(client, messages, model):
    try:
        # Sending only user messages and expecting assistant response
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response
    except Exception as e:
        logging.error(f"Exception during API call: {e}")
        raise e  # Let tenacity handle the retry

# ------------------------------
# Function to test the performance of the fine-tuned model
# ------------------------------
def test_model_concurrent(client, testing_data, model_name, assistant_description, verbose, run_number, data_type_filename, max_workers=10):
    """
    Processes API requests concurrently using ThreadPoolExecutor.

    For the o1-preview model:
    - No system messages allowed.
    - No function calls, no tools.
    - Only user and assistant messages.
    """
    correct = 0
    total = len(testing_data)
    unexpected_responses = []
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    results = []
    lock = threading.Lock()

    def process_entry(entry):
        nonlocal correct, true_positives, true_negatives, false_positives, false_negatives

        # Extract the sentence to classify from the JSONL data
        # Typically, entry['messages'][1]['content'] is the user sentence
        sentence = entry['messages'][1]['content']
        expected_response = entry["messages"][-1]["content"]

        # Merge instructions (assistant_description) and the sentence into one user message
        user_message = f"""
{assistant_description}

Please classify the following sentence according to the guidelines above.
You must return exactly one of the following four responses and nothing else:

1. "This sentence contains both fully and partially curatable data as well as terms related to curation."
2. "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."
3. "This sentence does not contain fully or partially curatable data but does contain terms related to curation."
4. "This sentence does not contain fully or partially curatable data or terms related to curation."

Sentence: "{sentence}"
"""

        # For o1, no system messages. Only user.
        messages = [
            {"role": "user", "content": user_message.strip()}
        ]

        try:
            completion = chat_completion_request(client, messages, model_name)
            if completion is None:
                raise ValueError("No response received from the API.")

            content = completion.choices[0].message.content.strip()

            response_category = classify_sentence(content)
            expected_category = classify_sentence(expected_response)

            with lock:
                result = {
                    "sentence": sentence,
                    "expected_response": expected_response,
                    "assistant_response": content,
                    "result_category": "correct" if response_category == expected_category else "incorrect",
                    "classification": ""
                }

                if response_category == expected_category:
                    correct += 1
                    if response_category["result"] == "curatable":
                        true_positives += 1
                        result["classification"] = "true_positive"
                    else:
                        true_negatives += 1
                        result["classification"] = "true_negative"
                else:
                    if verbose:
                        print(f"Sentence: {sentence}")
                        print(f"Expected: {expected_response}")
                        print(f"Got: {content}")
                        print("-" * 50)

                    if response_category["result"] == "curatable":
                        false_positives += 1
                        result["classification"] = "false_positive"
                    else:
                        false_negatives += 1
                        result["classification"] = "false_negative"

                    if content not in [entry["messages"][-1]["content"] for entry in testing_data]:
                        unexpected_responses.append(content)
                        if verbose:
                            print(f"Unexpected response: {content}")

                results.append(result)

        except ValueError as ve:
            with lock:
                if verbose:
                    print(f"ValueError: {ve}")
        except Exception as e:
            with lock:
                if verbose:
                    print(f"Exception: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_entry, testing_data), total=total, desc=f"Processing Run {run_number}", unit="sentence"))

    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if verbose:
        print(f"\nRun {run_number} Metrics:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print(f"True Positives: {true_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")

    # Save results to TSV file
    output_file_path = f'classification_results_{data_type_filename}_run{run_number}.tsv'
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, sep='\t', index=False)
    print(f"Successfully saved TSV file: {output_file_path}")

    # Save unexpected responses
    unexpected_responses_file_path = f'unexpected_responses_{data_type_filename}_run{run_number}.txt'
    with open(unexpected_responses_file_path, 'w', encoding='utf-8') as file:
        for response in unexpected_responses:
            file.write(response + '\n')
    print(f"Successfully saved unexpected responses file: {unexpected_responses_file_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Parse command-line arguments
    api_key, verbose, subset = parse_arguments()

    # Initialize OpenAI client
    client = initialize_openai_client(api_key)

    NUM_RUNS = 5
    MAX_WORKERS = 50

    logging.basicConfig(
        filename='testing_model_parallel.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO if not verbose else logging.DEBUG
    )

    for data_type_config in DATA_TYPES:
        type_of_data = data_type_config['type_of_data']
        type_of_data_filename = type_of_data.replace(' ', '_')
        input_file = data_type_config['input_file']
        model_name = data_type_config['model_name']
        
        # Instead of sending as system (not allowed), we just keep these instructions in the user message
        assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

        print(f"\n{'='*60}")
        print(f"Preparing to process data type: {type_of_data}")
        print(f"Input JSONL file: {input_file}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load testing data
        testing_data = load_jsonl(input_file)

        # Select a random subset if requested
        if subset:
            if len(testing_data) < 10:
                print(f"Warning: Requested subset size of 10, but only {len(testing_data)} entries are available.")
                subset_size = len(testing_data)
            else:
                subset_size = 10
            testing_data = random.sample(testing_data, subset_size)
            print(f"Selected a random subset of {len(testing_data)} entries for testing.")

        # Perform multiple test runs
        for run in range(1, NUM_RUNS + 1):
            print(f"\n--- Starting Test Run {run} for {type_of_data} ---")
            logging.info(f"Starting Test Run {run} for {type_of_data}")

            metrics = test_model_concurrent(
                client=client,
                testing_data=testing_data,
                model_name=model_name,
                assistant_description=assistant_description,
                verbose=verbose,
                run_number=run,
                data_type_filename=type_of_data_filename,
                max_workers=MAX_WORKERS
            )

            print(f"Completed Test Run {run} for {type_of_data}.\n")
            logging.info(f"Completed Test Run {run} for {type_of_data}. Metrics: {metrics}")

    print("\nAll test runs have been completed successfully.")
    logging.info("All test runs have been completed successfully.")

if __name__ == "__main__":
    main()
