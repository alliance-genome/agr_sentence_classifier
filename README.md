# agr_sentence_classifier

Overview
--------

This script performs sentence-level classification using OpenAI's GPT models and APIs. It is designed to evaluate the performance of a fine-tuned model on a dataset related to biological data types. The script reads testing data, sends classification requests to the model, and computes performance metrics such as accuracy, precision, recall, and F1 score. Additional scripts used to prepare the data can be found in the `additional_scripts` folder.

Prerequisites
-------------

-   **Python 3.7 or higher**
-   **OpenAI Python Library**: Install via `pip install openai`
-   **Additional Python Packages**:
    -   `pandas`
    -   `tqdm`
    -   `tenacity`

Ensure you have an active OpenAI API key with the necessary permissions to access the required models.

Installation
------------

1.  **Clone the Repository**

    `git clone <repository_url>
    cd <repository_directory>`

2.  **Install Required Packages**

    `pip install openai pandas tqdm tenacity`

Configuration
-------------

Before running the script, you may need to modify certain variables to suit your specific needs.

### Model Name

-   **Variable**: `model_name`

-   **Description**: Specifies the OpenAI model to be used for classification.

-   **Default Value**:

    `model_name = 'ft:gpt-4o-2024-08-06:alliance-of-genome-resources:expression-first:A4SuCivz'`

-   **Action**: Replace the default value with your own fine-tuned model name or another appropriate model.

### Data Type

-   **Variable**: `type_of_data`

-   **Description**: Defines the type of data being classified.

-   **Default Value**:

    `type_of_data = 'gene expression'`

-   **Action**: Modify this variable to reflect the data type relevant to your analysis.

### File Paths

-   **Testing Data File**

    -   **Variable**: `testing_output_file_path`

    -   **Default Value**:

        `testing_output_file_path = 'fine_tuned_testing_data_expression_gene_expression.jsonl'`

    -   **Action**: Ensure this path points to your testing data file in JSON Lines format.

-   **Output Files**

    -   **Classification Results**

        -   **Variable**: `output_file_path`

        -   **Default Value**:

            `output_file_path = f'classification_results_{type_of_data_filename}_gpt4o-5.tsv'`

    -   **Unexpected Responses**

        -   **Variable**: `unexpected_responses_file_path`

        -   **Default Value**:

            `unexpected_responses_file_path = f'unexpected_responses_{type_of_data_filename}_gpt4o-5.txt'`

Usage
-----

Run the script using the following command-line syntax:

`python script_name.py [-v] [-s] <your_openai_api_key>`

### Command-Line Arguments

-   `-v`: Enables verbose mode for detailed output.
-   `-s`: Runs a random subset of 10 entries from the testing data.
-   `<your_openai_api_key>`: Your OpenAI API key for authentication.

### Examples

-   **Standard Execution**

    `python script_name.py your_api_key_here`

-   **Verbose Mode**

    `python script_name.py -v your_api_key_here`

-   **Subset Execution**

    `python script_name.py -s your_api_key_here`

-   **Verbose Mode with Subset**

    `python script_name.py -v -s your_api_key_here`

Functionality
-------------

### Overview

The script evaluates the performance of a fine-tuned OpenAI model on sentence classification tasks. It processes each sentence in the testing data, requests a classification from the model, and compares the model's output to the expected result.

### Steps

1.  **Initialization**

    -   Parses command-line arguments to determine verbosity and subset options.
    -   Initializes the OpenAI client with the provided API key.
2.  **Data Loading**

    -   Loads testing data from the specified JSON Lines file.
    -   If the subset option is enabled, selects 10 random entries from the testing data.
3.  **Classification Function**

    -   Defines a classification function `classify_sentence` that interprets the model's output.
4.  **Model Interaction**

    -   Sends a chat completion request to the model for each sentence.
    -   Uses OpenAI's function calling to receive structured responses.
    -   Implements retry logic with exponential backoff for robustness.
5.  **Result Evaluation**

    -   Compares the model's classification with the expected result.
    -   Records metrics such as true positives, true negatives, false positives, and false negatives.
6.  **Output Generation**

    -   Saves detailed classification results to a TSV file.
    -   Logs any unexpected responses to a separate text file.

### Error Handling

-   The script exits upon encountering critical errors, such as failing to receive a valid response after multiple retries.
-   Verbose mode provides detailed logging for troubleshooting.

Output Files
------------

-   **Classification Results**

    -   Saved as a TSV file specified by `output_file_path`.

    -   Contains:

        -   The original sentence.
        -   Expected classification.
        -   Model's classification.
        -   Result category (correct or incorrect).
        -   Classification type (true_positive, true_negative, false_positive, false_negative).
-   **Unexpected Responses**

    -   Saved as a text file specified by `unexpected_responses_file_path`.
    -   Contains any model responses that did not match expected formats.

Performance Metrics
-------------------

The script calculates and prints the following metrics (currently commented out):

-   **Accuracy**
-   **Precision**
-   **Recall**
-   **F1 Score**
-   **True Positives**
-   **True Negatives**
-   **False Positives**
-   **False Negatives**

To enable metric calculation, uncomment the relevant section in the script:
```
accuracy = correct / total * 100
precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"True Positives: {true_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
```
Notes
-----

-   **API Rate Limits**: The script uses exponential backoff (`tenacity` library) to handle rate limits and transient errors.
-   **Data Privacy**: Ensure compliance with data privacy regulations when using real-world data.
-   **Verbose Mode**: Useful for debugging but may generate extensive output.
-   **Function Calling**: Relies on OpenAI's function calling feature for structured responses.

Contact
-------

For questions or issues, please open an issue in this repository.
