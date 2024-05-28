from openai import OpenAI

client = OpenAI(api_key=api_key)
import json
import sys

# Get the API key from the command line arguments
if len(sys.argv) != 2:
    print("Usage: python script_name.py <your_openai_api_key>")
    sys.exit(1)

api_key = sys.argv[1]

# Set your OpenAI API key

# Configuration variables
type_of_data = 'expression data'
model_name = 'ft:gpt-3.5-turbo-0125:alliance-of-genome-resources:expression:9Tt6sT60'  # Replace with your fine-tuned model name
assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

# File paths
testing_output_file_path = 'fine_tuned_testing_data_expression.jsonl'

# Load the JSONL file
testing_data = []
with open(testing_output_file_path, 'r') as file:
    for line in file:
        testing_data.append(json.loads(line))

# Define the function that returns the structured response
def classify_sentence(content):
    fully = "fully curatable data" in content
    partially = "partially curatable data" in content
    related = "terms related to curation" in content

    if fully and partially:
        if related:
            return json.dumps({"result": "This sentence contains both fully and partially curatable data as well as terms related to curation."})
        else:
            return json.dumps({"result": "This sentence contains both fully and partially curatable data."})
    elif fully and not partially:
        if related:
            return json.dumps({"result": "This sentence contains fully curatable data and terms related to curation."})
        else:
            return json.dumps({"result": "This sentence contains fully curatable data and does not contain partially curatable data."})
    elif not fully and partially:
        if related:
            return json.dumps({"result": "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."})
        else:
            return json.dumps({"result": "This sentence does not contain fully curatable data but it does contain partially curatable data."})
    elif not fully and not partially:
        if related:
            return json.dumps({"result": "This sentence does not contain fully or partially curatable data but does contain terms related to curation."})
        else:
            return json.dumps({"result": "This sentence does not contain fully or partially curatable data or terms related to curation."})

# Define the functions for the API
functions = [
    {
        "name": "classify_sentence",
        "description": "Classify the sentence based on curatable data types.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the sentence to classify."
                }
            },
            "required": ["content"]
        }
    }
]

# Function to test the performance of the fine-tuned model
def test_model(testing_data, model_name):
    correct = 0
    total = len(testing_data)

    for entry in testing_data:
        messages = [
            {"role": "system", "content": assistant_description},
            {"role": "user", "content": entry["messages"][1]["content"]}
        ]
        expected_response = entry["messages"][-1]["content"]

        response = client.chat.completions.create(model=model_name,
        messages=messages,
        functions=functions,
        function_call={"name": "classify_sentence"})

        function_call_response = json.loads(response.choices[0].message.content).result
        assistant_response = response.choices[0].message.content.strip()

        if function_call_response == expected_response:
            correct += 1
        else:
            print(f"Sentence: {messages[1]['content']}")
            print(f"Expected: {expected_response}")
            print(f"Got: {function_call_response}")
            print("-" * 50)

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Test the fine-tuned model
test_model(testing_data, model_name)
