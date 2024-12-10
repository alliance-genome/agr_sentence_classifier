import json
import random
import pandas as pd
import openai  # Correct import
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sys
from tqdm import tqdm
import os
import logging

# ------------------------------
# Configuration for Multiple Data Types
# ------------------------------
DATA_TYPES = [
    {
        'type_of_data': 'gene expression',
        'input_file': 'expression_test.jsonl',
        'model_name': 'ft:gpt-3.5-turbo-0125:alliance-of-genome-resources:expression-eighth:Ad1RsqXE'
    },
    {
        'type_of_data': 'protein kinase activity',
        'input_file': 'kinase_test.jsonl',
        'model_name': 'ft:gpt-3.5-turbo-0125:alliance-of-genome-resources:kinase-eighth:Ad1UasK8'
    }
]

# ------------------------------
# Setup Logging
# ------------------------------
logging.basicConfig(
    filename='testing_model.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------------------
# Get the API key and verbosity flags from the command line arguments
# ------------------------------
def parse_arguments():
    if len(sys.argv) < 2:
        print("Usage: python script_name.py [-v] [-s] <your_openai_api_key>")
        print("Options:")
        print("  -v    Enable verbose mode to print detailed output.")
        print("  -s    Run a random subset of 10 entries from the testing data.")
        print("Arguments:")
        print("  <your_openai_api_key>    Your OpenAI API key for authentication.")
        sys.exit(1)
    
    verbose = False
    subset = False
    api_key = None

    args = sys.argv[1:]
    for arg in args:
        if arg == '-v':
            verbose = True
        elif arg == '-s':
            subset = True
        else:
            api_key = arg
    
    if not api_key:
        print("Error: Missing OpenAI API key.")
        sys.exit(1)
    
    return api_key, verbose, subset

# ------------------------------
# Initialize the OpenAI client
# ------------------------------
def initialize_openai_client(api_key):
    try:
        openai.api_key = api_key
        # Test the API key by making a simple request
        openai.Engine.list()
        logging.info("OpenAI client initialized successfully.")
        if verbose:
            print("OpenAI client initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Error initializing OpenAI client: {e}")
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

# ------------------------------
# Define the function that returns the structured response
# ------------------------------
def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_sentence",
                "description": "Classify the sentence based on curatable data types.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "enum": [
                                "This sentence contains both fully and partially curatable data as well as terms related to curation.",
                                "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.",
                                "This sentence does not contain fully or partially curatable data but does contain terms related to curation.",
                                "This sentence does not contain fully or partially curatable data or terms related to curation."
                            ],
                            "description": "The classification result of the sentence."
                        }
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            }
        }
    ]

# ------------------------------
# Load the JSONL file
# ------------------------------
def load_jsonl(file_path, verbose=False):
    testing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                testing_data.append(json.loads(line))
        print(f"Loaded {len(testing_data)} entries from {file_path}.")
        logging.info(f"Loaded {len(testing_data)} entries from {file_path}.")
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decode error in file {file_path}: {jde}")
        print(f"JSON decode error in file {file_path}: {jde}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {file_path}: {e}")
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        sys.exit(1)
    return testing_data

# ------------------------------
# Utility function to make chat completion requests with retry logic
# ------------------------------
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools, model):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=tools,
            function_call={"name": "classify_sentence"}
        )
        return response
    except Exception as e:
        logging.warning(f"Attempt failed with exception: {e}. Retrying...")
        raise e  # Trigger retry

# ------------------------------
# Define the function that maps assistant response to classification
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
# Function to test the performance of the fine-tuned model
# ------------------------------
def test_model(client, tools, testing_data, model_name, assistant_description, verbose, run_number, data_type_filename):
    correct = 0
    total = len(testing_data)
    unexpected_responses = []
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    results = []

    # Initialize tqdm with dynamic_ncols to ensure proper display
    with tqdm(total=total, desc=f"Processing Run {run_number}", unit="sentence", dynamic_ncols=True) as pbar:
        for entry in testing_data:
            user_message = f"Please classify the content of this sentence in terms of its possibility of curation: {entry['messages'][1]['content']}."
            messages = [
                {"role": "system", "content": assistant_description},
                {"role": "user", "content": user_message},
            ]
            expected_response = entry.get("messages", [{}])[-1].get("content", "")
    
            # Retry logic for unexpected responses
            retry_count = 0
            max_retries = 20
            while retry_count < max_retries:
                try:
                    if verbose:
                        print(f"Run {run_number}, Entry {retry_count+1}: Sending API request.")
                    completion = chat_completion_request(messages, tools, model_name)
                    
                    if completion is None:
                        logging.warning(f"Run {run_number}, Entry {retry_count+1}: Received no response. Retrying...")
                        retry_count += 1
                        continue
    
                    if verbose:
                        print(f"Run {run_number}, Entry {retry_count+1}: Received response.")
                        print(f"Completion Response: {json.dumps(completion, indent=2)}")
    
                    tool_calls = completion.choices[0].message.get("function_call")
                    if not tool_calls:
                        raise ValueError("No function call was made by the model.")
                        
                    function_call_response = json.loads(tool_calls.get("arguments", "{}"))
                    content = function_call_response.get("content", "")
                    
                    if verbose:
                        print(f"Run {run_number}, Entry {retry_count+1}: Function call response: {content}")
    
                    # Use the classify_sentence function to categorize the response
                    response_category = classify_sentence(content)
                    expected_category = classify_sentence(expected_response)
    
                    result = {
                        "sentence": entry['messages'][1]['content'],
                        "expected_response": expected_response,
                        "assistant_response": content,
                        "result_category": "correct" if response_category == expected_category else "incorrect",
                        "classification": ""
                    }
    
                    if response_category["result"] == expected_category["result"]:
                        correct += 1
                        if response_category["result"] == "curatable":
                            true_positives += 1
                            result["classification"] = "true_positive"
                        else:
                            true_negatives += 1
                            result["classification"] = "true_negative"
                    else:
                        if verbose:
                            print(f"Run {run_number}, Entry {retry_count+1}: Mismatch detected.")
                            print(f"Sentence: {messages[1]['content']}")
                            print(f"Expected: {expected_response}")
                            print(f"Got: {content}")
                            print("-" * 50)
    
                        if response_category["result"] == "curatable":
                            false_positives += 1
                            result["classification"] = "false_positive"
                        else:
                            false_negatives += 1
                            result["classification"] = "false_negative"
    
                        if content not in [resp["expected_response"] for resp in testing_data]:
                            unexpected_responses.append(content)
                            if verbose:
                                print(f"Run {run_number}, Entry {retry_count+1}: Unexpected response: {content}")
    
                    results.append(result)
                    pbar.update(1)
                    break  # Exit the retry loop if a valid response is received
                except ValueError as e:
                    if verbose:
                        print(f"Run {run_number}, Entry {retry_count+1}: ValueError encountered: {e}")
                    logging.warning(f"Run {run_number}, Entry {retry_count+1}: ValueError encountered: {e}")
                    retry_count += 1  # Increment retry count on unexpected responses
                    continue  # Retry on unexpected responses
                except Exception as e:
                    if verbose:
                        print(f"Run {run_number}, Entry {retry_count+1}: An unexpected error occurred: {e}")
                    logging.error(f"Run {run_number}, Entry {retry_count+1}: An unexpected error occurred: {e}")
                    retry_count += 1  # Increment retry count on unexpected responses
                    continue  # Retry on other exceptions
    
            # Check if no valid response was received after retries
            if retry_count == max_retries:
                print(f"Run {run_number}, Entry {retry_count}: Failed to get a valid response for sentence: {entry['messages'][1]['content']}")
                logging.error(f"Run {run_number}, Entry {retry_count}: Failed to get a valid response for sentence: {entry['messages'][1]['content']}")
                pbar.update(1)  # Still update the progress bar
                continue  # Skip to next entry
    
    # Calculate metrics (optional)
    accuracy = (correct / total) * 100 if total > 0 else 0
    precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
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
    logging.info(f"Successfully saved TSV file: {output_file_path}")
    if verbose:
        print(f"Successfully saved TSV file: {output_file_path}")

    # Save unexpected responses to a file
    unexpected_responses_file_path = f'unexpected_responses_{data_type_filename}_run{run_number}.txt'
    with open(unexpected_responses_file_path, 'w', encoding='utf-8') as file:
        for response in unexpected_responses:
            file.write(response + '\n')
    logging.info(f"Successfully saved unexpected responses file: {unexpected_responses_file_path}")
    if verbose:
        print(f"Successfully saved unexpected responses file: {unexpected_responses_file_path}")

    # Return metrics (optional, for further processing)
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
    initialized = initialize_openai_client(api_key)
    if not initialized:
        print("Failed to initialize OpenAI client.")
        sys.exit(1)

    # Get tools definition
    tools = get_tools()

    # Define number of test runs
    NUM_RUNS = 5

    # Loop through each data type
    for data_type_config in DATA_TYPES:
        type_of_data = data_type_config['type_of_data']
        type_of_data_filename = type_of_data.replace(' ', '_')
        input_file = data_type_config['input_file']
        model_name = data_type_config['model_name']
        assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

        print(f"\n{'='*60}")
        print(f"Preparing to process data type: {type_of_data}")
        print(f"Input JSONL file: {input_file}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load testing data
        testing_data = load_jsonl(input_file, verbose=verbose)

        # Select a random subset if the -s flag is set
        if subset:
            if len(testing_data) < 10:
                print(f"Warning: Requested subset size of 10, but only {len(testing_data)} entries are available.")
                subset_size = len(testing_data)
            else:
                subset_size = 10
            testing_data = random.sample(testing_data, subset_size)
            print(f"Selected a random subset of {len(testing_data)} entries for testing.")
            logging.info(f"Selected a random subset of {len(testing_data)} entries for testing.")

        # Perform multiple test runs
        for run in range(1, NUM_RUNS + 1):
            print(f"\n--- Starting Test Run {run} for {type_of_data} ---")
            logging.info(f"Starting Test Run {run} for {type_of_data}")
            metrics = test_model(
                client=None,  # Not needed anymore
                tools=tools,
                testing_data=testing_data,
                model_name=model_name,
                assistant_description=assistant_description,
                verbose=verbose,
                run_number=run,
                data_type_filename=type_of_data_filename
            )
            print(f"Completed Test Run {run} for {type_of_data}.\n")
            logging.info(f"Completed Test Run {run} for {type_of_data}. Metrics: {metrics}")

    print("\nAll test runs have been completed successfully.")
    logging.info("All test runs have been completed successfully.")

if __name__ == "__main__":
    main()
