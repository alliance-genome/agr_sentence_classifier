import json
import random
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
type_of_data = 'expression data'
model_name = 'ft:gpt-3.5-turbo-0125:alliance-of-genome-resources:expression:9Tt6sT60'  # Replace with your fine-tuned model name
assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}. Please classify the content of this sentence in terms of its possibility of curation."

# File paths
testing_output_file_path = 'fine_tuned_testing_data_expression.jsonl'

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
def chat_completion_request(messages, model=model_name):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        sys.exit(1)  # Halt on exceptions

# Define expected responses
expected_responses = {
    "This sentence contains both fully and partially curatable data as well as terms related to curation.": "curatable",
    "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation.": "curatable",
    "This sentence does not contain fully or partially curatable data but does contain terms related to curation.": "not_curatable",
    "This sentence does not contain fully or partially curatable data or terms related to curation.": "not_curatable"
}

# Function to test the performance of the fine-tuned model
def test_model(testing_data, model_name, verbose):
    correct = 0
    total = len(testing_data)
    unexpected_responses = []
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for entry in tqdm(testing_data, desc="Processing", unit="sentence"):
        user_message = f"Please classify the content of this sentence in terms of its possibility of curation: {entry['messages'][1]['content']}"
        messages = [
            {"role": "system", "content": assistant_description},
            {"role": "user", "content": user_message}
        ]
        expected_response = entry["messages"][-1]["content"]

        completion = chat_completion_request(messages=messages)

        if completion is None:
            continue

        try:
            if verbose:
                print("Completion response received.")
                completion_dict = completion.to_dict()  # Convert the response to a dictionary
                print(json.dumps(completion_dict, indent=2))  # Print the full completion object

            assistant_response = completion.choices[0].message.content.strip()
            if verbose:
                print(f"Assistant response: {assistant_response}")

            expected_category = expected_responses.get(expected_response)
            actual_category = expected_responses.get(assistant_response)

            if assistant_response == expected_response:
                correct += 1
                if actual_category == "curatable":
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if verbose:
                    print(f"Sentence: {messages[1]['content']}")
                    print(f"Expected: {expected_response}")
                    print(f"Got: {assistant_response}")
                    print("-" * 50)

                if actual_category == "curatable":
                    false_positives += 1
                else:
                    false_negatives += 1

                if assistant_response not in [entry["messages"][-1]["content"] for entry in testing_data]:
                    unexpected_responses.append(assistant_response)
                    if verbose:
                        print(f"Unexpected response: {assistant_response}")
        except Exception as e:
            print(f"An error occurred while processing the completion: {e}")
            print(f"Messages: {json.dumps(messages, indent=2)}")
            sys.exit(1)  # Halt on exceptions

    accuracy = correct / total * 100
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    if unexpected_responses:
        print("\nUnexpected responses:")
        for response in unexpected_responses:
            print(response)

# Test the fine-tuned model
test_model(testing_data, model_name, verbose)
