import openai
import json
import sys

# Get the API key from the command line arguments
if len(sys.argv) != 2:
    print("Usage: python script_name.py <your_openai_api_key>")
    sys.exit(1)

api_key = sys.argv[1]

# Set your OpenAI API key
openai.api_key = api_key

# Configuration variables
type_of_data = 'expression data'
model_name = 'your_fine_tuned_model'  # Replace with your fine-tuned model name
assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

# File paths
testing_output_file_path = 'fine_tuned_testing_data_expression.jsonl'

# Load the JSONL file
testing_data = []
with open(testing_output_file_path, 'r') as file:
    for line in file:
        testing_data.append(json.loads(line))

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
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages
        )
        
        assistant_response = response.choices[0].message["content"].strip()
        
        if assistant_response == expected_response:
            correct += 1
        else:
            print(f"Sentence: {messages[1]['content']}")
            print(f"Expected: {expected_response}")
            print(f"Got: {assistant_response}")
            print("-" * 50)
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Test the fine-tuned model
test_model(testing_data, model_name)
