import json
import random
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sys
from tqdm import tqdm

# Get the API key and verbosity flag from the command line arguments
if len(sys.argv) not in [2, 3, 4]:
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

api_key = sys.argv[api_key_index]

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Configuration variables
type_of_data = 'gene expression'
type_of_data_filename = type_of_data.replace(' ', '_')
model_name = 'ft:gpt-4o-2024-08-06:alliance-of-genome-resources:expression-first:A4SuCivz'  # Replace with your fine-tuned model name
assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

# File paths
testing_output_file_path = 'fine_tuned_testing_data_expression_gene_expression.jsonl'
output_file_path = f'classification_results_{type_of_data_filename}_gpt4o-1.tsv'
unexpected_responses_file_path = f'unexpected_responses_{type_of_data_filename}_gpt4o-1.txt'

# Define the functions parameter for the request
functions = [
    {
        "name": "classify_sentence",
        "description": "Classify the sentence based on curatable data types.",
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
            "required": ["content"]
        }
    }
]

# Load the JSONL file
testing_data = []
with open(testing_output_file_path, 'r') as file:
    for line in file:
        testing_data.append(json.loads(line))

# Select a random subset if the -s flag is set
if subset:
    testing_data = random.sample(testing_data, 10)

# Utility function to make chat completion requests with retry logic
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions, model=model_name):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call={"name": "classify_sentence"}
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        sys.exit(1)  # Halt on exceptions

# Define the function that returns the structured response
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

# Function to test the performance of the fine-tuned model
def test_model(testing_data, model_name, verbose):
    correct = 0
    total = len(testing_data)
    unexpected_responses = []
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    results = []

    for entry in tqdm(testing_data, desc="Processing", unit="sentence"):
        user_message = f"Please classify the content of this sentence in terms of its possibility of curation: {entry['messages'][1]['content']}."
        messages = [
            {"role": "system", "content": assistant_description},
            {"role": "user", "content": user_message},
        ]
        expected_response = entry["messages"][-1]["content"]

        # Retry logic for unexpected responses
        retry_count = 0
        while retry_count < 20:
            completion = chat_completion_request(messages=messages, functions=functions)
            if completion is None:
                retry_count += 1
                continue

            try:
                if verbose:
                    print("Completion response received.")
                    completion_dict = completion.to_dict()  # Convert the response to a dictionary
                    print(json.dumps(completion_dict, indent=2))  # Print the full completion object

                # Access the function call response
                tool_calls = completion.choices[0].message.tool_calls  # Adjusted access method
                if not tool_calls:
                    raise ValueError("No function call was made by the model.")

                function_call_response = json.loads(tool_calls[0]["function"]["arguments"])
                content = function_call_response["content"]

                if verbose:
                    print(f"Function call response: {content}")

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

                    if content not in [entry["messages"][-1]["content"] for entry in testing_data]:
                        unexpected_responses.append(content)
                        if verbose:
                            print(f"Unexpected response: {content}")

                results.append(result)
                break  # Exit the retry loop if a valid response is received
            except ValueError as e:
                if verbose:
                    print(f"An error occurred while processing the completion: {e}")
                    print(f"Messages: {json.dumps(messages, indent=2)}")
                retry_count += 1  # Increment retry count on unexpected responses
                continue  # Retry on unexpected responses
            except Exception as e:
                if verbose:
                    print(f"An error occurred while processing the completion: {e}")
                    print(f"Messages: {json.dumps(messages, indent=2)}")
                sys.exit(1)  # Halt on exceptions

        # Check if no valid response was received after retries
        if retry_count == 20:
            print(f"Failed to get a valid response for sentence: {entry['messages'][1]['content']}")
            sys.exit(1)  # Exit the program if no valid response is received

    # Save results to TSV file
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, sep='\t', index=False)

    # Save unexpected responses to a file
    with open(unexpected_responses_file_path, 'w') as file:
        for response in unexpected_responses:
            file.write(response + '\n')

# Test the fine-tuned model
test_model(testing_data, model_name, verbose)
