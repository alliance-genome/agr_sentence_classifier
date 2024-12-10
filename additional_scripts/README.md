README
======

Data Preparation Scripts
------------------------

This directory contains scripts used for data preparation and validation in the context of classifying sentences. Below is a brief summary of each script and guidance on how they can be adjusted.

### 1\. `data_extraction.py`

#### Purpose

-   **Extracts and filters sentences** from a TSV file (`expression_sentence_datasets.tsv`) based on specified criteria.
-   **Generates structured data** for training and testing by creating JSON Lines (`.jsonl`) files.

#### Functionality

-   **Filters data** based on:

    -   Source (`SOURCE` column): using `source_filter`.
    -   Fully curatable sentences (`FULLY_CURATABLE` column): using `fully_curable_filter`.
    -   Partially curatable sentences (`PARTIALLY_CURATABLE` column): using `partially_curable_filter`.
    -   Related language (`RELATED_LANGUAGE` column): using `related_language_filter`.
    -   Training or testing data (`TRAINING_OR_TESTING` column).
-   **Creates assistant responses** based on the curation status of each sentence.

-   **Outputs**:

    -   `fine_tuned_training_data_expression_<type_of_data>.jsonl`
    -   `fine_tuned_testing_data_expression_<type_of_data>.jsonl`

#### How to Adjust

-   **Filtering Criteria**:

    -   Modify the `source_filter`, `fully_curable_filter`, `partially_curable_filter`, and `related_language_filter` lists to change the filtering criteria.

    `source_filter = ['GOLD', '1000']  # Adjust source identifiers
    fully_curable_filter = [0, 1]     # Use [1] for only fully curatable sentences
    partially_curable_filter = [0, 1] # Use [1] for only partially curatable sentences
    related_language_filter = [0, 1]  # Use [1] for sentences with related language`

-   **Data Type**:

    -   Update the `type_of_data` variable to reflect the specific data type you are working with.

    `type_of_data = 'gene expression'  # Change to your specific data type`

-   **Input and Output Paths**:

    -   Modify `input_file_path` if your input TSV file is named differently or located elsewhere.

    -   The output file paths are generated based on `type_of_data`; adjust if necessary.

### 2\. `check_sentences.py`

#### Purpose

-   **Verifies** whether specific sentences are present in a given JSON Lines (`.jsonl`) file.

#### Functionality

-   **Loads** the `.jsonl` file specified by `file_path`.

-   **Checks** if the sentences listed in `sentences_to_check` are present in the assistant's responses within the JSON Lines file.

-   **Outputs** a DataFrame indicating which sentences are present.

#### How to Adjust

-   **File Path**:

    -   Set `file_path` to point to the JSON Lines file you want to check.

    `file_path = 'fine_tuned_testing_data_expression.jsonl'  # Update as needed`

-   **Sentences to Check**:

    -   Modify the `sentences_to_check` list to include the sentences you want to verify.

    `sentences_to_check = [
        "Sentence 1",
        "Sentence 2",
        # Add or remove sentences as needed
    ]`

### 3\. `check_missing_data.py`

#### Purpose

-   **Checks for missing or incomplete data** in the results obtained from running the classification script.

#### Functionality

-   **Loads** prompts from a JSON Lines (`.jsonl`) file containing all possible prompts.

-   **Loads** results from a TSV file containing classification results.

-   **Identifies**:

    -   Prompts that are missing in the results file.
    -   Potential early termination of the classification script.
    -   Duplicate entries in the results file.

#### How to Adjust

-   **File Paths**:

    -   Update `jsonl_file_path` to point to your prompts file.

        `jsonl_file_path = 'fine_tuned_testing_data_expression_gene_expression.jsonl'  # Update as needed`

    -   Update `tsv_file_path` to point to your results file.

        `tsv_file_path = 'classification_results_gene_expression_4.tsv'  # Update as needed`

-   **Debugging Output**:

    -   The script prints out detailed information about missing prompts, potential early termination, and duplicates.

General Notes
-------------

-   Ensure all file paths point to the correct locations of your data files.

-   Adjust filtering criteria and data type variables to suit your specific dataset and requirements.

-   These scripts assume specific column names in your data files; ensure your data files have the necessary columns.

-   The scripts use pandas for data manipulation; make sure you have it installed:

    `pip install pandas`

Contact
-------

For questions or concerns, please open an issue in this repository.
