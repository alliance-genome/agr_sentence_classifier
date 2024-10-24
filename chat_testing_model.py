import json
import random
import pandas as pd
import requests
import sys
import time
from tqdm import tqdm
import argparse
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Sentence Classification Script')
parser.add_argument('--api_key', help='Your OpenAI API key for authentication.')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode to print detailed output.')
parser.add_argument('-s', '--subset', action='store_true', help='Run a random subset of 10 entries from the testing data.')
args = parser.parse_args()

verbose = args.verbose
subset = args.subset
api_key = args.api_key

# Configuration variables
type_of_data = 'gene expression'
type_of_data_filename = type_of_data.replace(' ', '_')
model_name = 'o1-preview-2024-09-12'  # Replace with your model name

assistant_description = f"This assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

# File paths
testing_output_file_path = 'fine_tuned_testing_data_expression_gene_expression.jsonl'
output_file_path = f'classification_results_{type_of_data_filename}_{model_name}.tsv'
unexpected_responses_file_path = f'unexpected_responses_{type_of_data_filename}_{model_name}.txt'

# Expected responses and their categories
expected_responses = {
    "1": "curatable",
    "2": "curatable",
    "3": "not_curatable",
    "4": "not_curatable",
    "This sentence contains both fully and partially curatable data as well as terms related to curation": "curatable",
    "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation": "curatable",
    "This sentence does not contain fully or partially curatable data but does contain terms related to curation": "not_curatable",
    "This sentence does not contain fully or partially curatable data or terms related to curation": "not_curatable"
}

# Mapping from numbers to full sentences (for reference)
number_to_sentence = {
    "1": "This sentence contains both fully and partially curatable data as well as terms related to curation",
    "2": "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation",
    "3": "This sentence does not contain fully or partially curatable data but does contain terms related to curation",
    "4": "This sentence does not contain fully or partially curatable data or terms related to curation"
}

# Load the JSONL file
testing_data = []
with open(testing_output_file_path, 'r') as file:
    for line in file:
        testing_data.append(json.loads(line))

# Select a random subset if the -s flag is set
if subset:
    testing_data = random.sample(testing_data, 10)

# Function to normalize content
def normalize_content(content):
    content = content.strip()
    content = content.rstrip('.:')
    content = content.strip('"\'')
    # Remove any numbering like "1." or "1)"
    content = re.sub(r'^\d+[\.\)]\s*', '', content)
    # Extract number if content is like "Option 2", "Answer: 3", etc.
    match = re.match(r'^(?:Option\s*|Answer:\s*)?(\d)', content, re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        # Handle multi-line responses by taking the first line
        content = content.splitlines()[0]
    # Convert numerical strings to integers and back to strings
    if content.isdigit():
        content = str(int(content))
    return content

# Function to classify the assistant's response
def classify_sentence(content):
    content = normalize_content(content)
    if content in expected_responses:
        return {"result": expected_responses[content]}
    else:
        raise ValueError(f"Unexpected response: {content}")

# Function to check if the model supports the 'system' role
def supports_system_role(model_name):
    # List of models that do not support the 'system' role
    models_without_system_role = ['o1-mini', 'o1-preview']
    for model_prefix in models_without_system_role:
        if model_name.startswith(model_prefix):
            return False
    return True

# Function to test the model
def test_model(testing_data, model_name, verbose):
    correct = 0
    total = len(testing_data)
    unexpected_responses = []
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    results = []

    # Rate limiting variables
    rate_limit_per_minute = 60  # Default rate limit
    if '1o' in model_name.lower():
        rate_limit_per_minute = 25  # Set rate limit to 25 requests per minute
        if verbose:
            print(f"Model '{model_name}' detected as '1o' model. Rate limit set to {rate_limit_per_minute} requests per minute.")
    min_time_between_requests = 60 / rate_limit_per_minute  # Minimum time between requests in seconds
    last_request_timestamp = None

    # Set up the API endpoint and headers
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Check if the model supports the 'system' role
    model_supports_system = supports_system_role(model_name)

    for entry in tqdm(testing_data, desc="Processing", unit="sentence"):
        sentence = entry['messages'][1]['content']
        
        # Build the user message
        if model_supports_system:
            user_message_content = f"Please classify the content of this sentence in terms of its possibility of curation: \"{sentence}\".\n\n" \
                                   f"Your response should be exactly one of the following four options (you may respond with just the number or the full sentence):\n" \
                                   f"1. This sentence contains both fully and partially curatable data as well as terms related to curation.\n" \
                                   f"2. This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.\n" \
                                   f"3. This sentence does not contain fully or partially curatable data but does contain terms related to curation.\n" \
                                   f"4. This sentence does not contain fully or partially curatable data or terms related to curation.\n\n" \
                                   f"Please respond with exactly one of these options, either the number or the full sentence, without any additional text."
            # Messages with 'system' role
            messages = [
                {"role": "system", "content": assistant_description},
                {"role": "user", "content": user_message_content},
            ]
        else:
            # Include assistant description in the user's message
            user_message_content = f"{assistant_description}\n\n" \
                                   f"Please classify the content of this sentence in terms of its possibility of curation: \"{sentence}\".\n\n" \
                                   f"Your response should be exactly one of the following four options (you may respond with just the number or the full sentence):\n" \
                                   f"1. This sentence contains both fully and partially curatable data as well as terms related to curation.\n" \
                                   f"2. This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.\n" \
                                   f"3. This sentence does not contain fully or partially curatable data but does contain terms related to curation.\n" \
                                   f"4. This sentence does not contain fully or partially curatable data or terms related to curation.\n\n" \
                                   f"Please respond with exactly one of these options, either the number or the full sentence, without any additional text."
            # Messages without 'system' role
            messages = [
                {"role": "user", "content": user_message_content},
            ]

        expected_response = entry["messages"][-1]["content"]

        # Normalize expected_response
        expected_response_normalized = normalize_content(expected_response)

        # Retry logic for unexpected responses
        retry_count = 0
        max_retries = 5
        assistant_content = ""
        while retry_count < max_retries:
            try:
                # Rate limiting
                if last_request_timestamp is not None:
                    elapsed_time = time.time() - last_request_timestamp
                    sleep_time = min_time_between_requests - elapsed_time
                    if sleep_time > 0:
                        if verbose:
                            print(f"Sleeping for {sleep_time:.2f} seconds to respect rate limit.")
                        time.sleep(sleep_time)
                last_request_timestamp = time.time()

                # Prepare the payload for the API request
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 1   # Set temperature to 0 for deterministic output
                }

                # Make the API request
                response = requests.post(url, headers=headers, json=payload)

                # Check for errors
                if response.status_code != 200:
                    print(f"Error with model '{model_name}': {response.status_code} {response.reason}")
                    print(response.text)
                    sys.exit(1)

                # Parse the response
                response_data = response.json()
                assistant_content = response_data['choices'][0]['message']['content'].strip()

                if verbose:
                    print("\nCompletion response received:")
                    print(json.dumps(response_data, indent=2))
                    print(f"Assistant's raw response: {repr(assistant_content)}")

                # Normalize assistant_content
                assistant_content_normalized = normalize_content(assistant_content)

                # Check if assistant_content is a valid response
                classification_result = classify_sentence(assistant_content_normalized)

                # Use the classify_sentence function to categorize the response
                response_category = classification_result
                expected_category = classify_sentence(expected_response_normalized)

                result = {
                    "sentence": sentence,
                    "expected_response": expected_response,
                    "assistant_response": assistant_content,
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
                        print(f"Got: {assistant_content}")
                        print("-" * 50)

                    if response_category["result"] == "curatable":
                        false_positives += 1
                        result["classification"] = "false_positive"
                    else:
                        false_negatives += 1
                        result["classification"] = "false_negative"

                    if assistant_content not in [entry["messages"][-1]["content"] for entry in testing_data]:
                        unexpected_responses.append(assistant_content)
                        if verbose:
                            print(f"Unexpected response: {assistant_content}")

                results.append(result)
                break  # Exit the retry loop if a valid response is received

            except ValueError as e:
                if verbose:
                    print(f"An error occurred while processing the completion: {e}")
                    print(f"Assistant's response: {assistant_content}")
                    print(f"Retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1  # Increment retry count on unexpected responses
                # Modify the prompt to reinforce the expected format
                messages.append({
                    "role": "user",
                    "content": "Please make sure your response is exactly one of the four provided options, either the number or the full sentence, without any additional text."
                })
                continue  # Retry on unexpected responses

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)  # Halt on exceptions

        # Check if no valid response was received after retries
        if retry_count == max_retries:
            print(f"Failed to get a valid response for sentence: {sentence}")
            unexpected_responses.append(f"Failed after {max_retries} retries: {sentence}")
            results.append({
                "sentence": sentence,
                "expected_response": expected_response,
                "assistant_response": "No valid response",
                "result_category": "failed",
                "classification": "failed"
            })

    # Save results to TSV file
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, sep='\t', index=False)

    # Save unexpected responses to a file
    with open(unexpected_responses_file_path, 'w') as file:
        for response in unexpected_responses:
            file.write(response + '\n')

    # Print summary statistics
    accuracy = correct / total * 100 if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print("\nSummary Statistics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

# Test the model
test_model(testing_data, model_name, verbose)
