Characterization and automated classification of sentences in the biomedical literature: a case study for biocuration of gene expression and protein kinase activity
=======================================

Overview
--------

This repository documents the data preparation and evaluation processes undertaken to create and assess biocuration-relevant datasets for gene expression and protein kinase activity domains using a fine-tuned GPT-4o model, as presented in our scientific publication, "Characterization and automated classification of sentences in the biomedical literature: a case study for biocuration of gene expression and protein kinase activity". The workflows encompassed data extraction, labeling, stratified splitting, model testing, and metric assessment.

Scripts and Workflow
--------------------

### 1\. `data_extraction.py`

**Purpose:**\
This script was utilized to extract, filter, and label sentences from raw TSV datasets pertaining to gene expression and protein kinase activity. It systematically processed the data to prepare it for model training and evaluation.

**Input Files:**

-   `expression_sentence_datasets.tsv`
-   `kinase_sentence_datasets.tsv`

**Operations Performed:**

-   **Data Loading and Filtering:**\
    Imported the raw TSV files and filtered the data based on predefined criteria, including the source (`SOURCE`), and binary indicators for `FULLY_CURATABLE`, `PARTIALLY_CURATABLE`, and `RELATED_LANGUAGE`.

-   **Label Assignment:**\
    Assigned multi-label categories to each sentence, classifying them as:

    -   **Fully Curatable:** Contains all necessary elements for direct annotation.
    -   **Partially Curatable:** Contains some relevant information but incomplete for full annotation.
    -   **Language Related:** Contains relevant terms but lacks sufficient detail for curation.
    -   **Not Curatable:** Does not contain relevant information or terms.
-   **Stratified Splitting:**\
    Employed a multi-label stratified shuffle split to divide the dataset into training (70%), validation (15%), and testing (15%) subsets, ensuring balanced representation of all labels across splits.

**Output Files:** For each data type (`gene expression` and `protein kinase activity`), the script generated the following JSON Lines (`.jsonl`) files within respective directories:

-   **Gene Expression:**
    -   `multi_label_split_data/gene_expression/final_train.jsonl`
    -   `multi_label_split_data/gene_expression/final_val.jsonl`
    -   `multi_label_split_data/gene_expression/final_test.jsonl`*
-   **Protein Kinase Activity:**
    -   `multi_label_split_data/protein_kinase_activity/final_train.jsonl`
    -   `multi_label_split_data/protein_kinase_activity/final_val.jsonl`
    -   `multi_label_split_data/protein_kinase_activity/final_test.jsonl`*

**\*NOTE:** The final test files were manually renamed to `final_expression_test.jsonl` and `final_kinase_text.jsonl` respectively and copied to the root folder of the repository for further processing.

**Subsequent Usage:** These JSONL files served as input for the `testing_model.py` script, which conducted model evaluations on the prepared test datasets. The training and validation subsets were used to train the GPT-4o model (gpt-4o-2024-08-06) via the OpenAI Dashboard interface.

* * * * *

### 2\. `testing_model.py`

**Purpose:**\
This script conducted evaluations of fine-tuned language models on the prepared test datasets. It processed the JSONL test files to generate classification results by interacting with specified language models.

**Input Files:**

-   `final_expression_test.jsonl`
-   `final_kinase_text.jsonl`

**Operations Performed:**

-   **Model Initialization:**\
    Configured and authenticated with the OpenAI API using provided API keys and model identifiers specific to each data type.

-   **Concurrent Processing:**\
    Executed five independent runs for each data type, where each run involved:

    -   Reading the test JSONL files.
    -   Sending classification requests to the designated language models.
    -   Collecting and recording the model's responses.

**Output Files:** For each data type and run, the script generated TSV files capturing the classification results:

-   **Gene Expression:**
    -   `final_classification_results_gene_expression_run1.tsv`
    -   `final_classification_results_gene_expression_run2.tsv`
    -   `final_classification_results_gene_expression_run3.tsv`
    -   `final_classification_results_gene_expression_run4.tsv`
    -   `final_classification_results_gene_expression_run5.tsv`
-   **Protein Kinase Activity:**
    -   `final_classification_results_protein_kinase_activity_run1.tsv`
    -   `final_classification_results_protein_kinase_activity_run2.tsv`
    -   `final_classification_results_protein_kinase_activity_run3.tsv`
    -   `final_classification_results_protein_kinase_activity_run4.tsv`
    -   `final_classification_results_protein_kinase_activity_run5.tsv`

**Subsequent Usage:** These classification result TSV files were subsequently analyzed by the `metrics_assessment.py` script to compute performance metrics across multiple runs.

* * * * *

### 3\. `metrics_assessment.py`

**Purpose:**\
This script evaluated the performance of the language models by calculating precision, recall, and F1-scores based on the classification results obtained from `testing_model.py`. It aggregated metrics across multiple runs to provide a comprehensive assessment of model performance.

**Input Files:**

-   **Gene Expression:**
    -   `final_classification_results_gene_expression_run1.tsv`
    -   `final_classification_results_gene_expression_run2.tsv`
    -   `final_classification_results_gene_expression_run3.tsv`
    -   `final_classification_results_gene_expression_run4.tsv`
    -   `final_classification_results_gene_expression_run5.tsv`
-   **Protein Kinase Activity:**
    -   `final_classification_results_protein_kinase_activity_run1.tsv`
    -   `final_classification_results_protein_kinase_activity_run2.tsv`
    -   `final_classification_results_protein_kinase_activity_run3.tsv`
    -   `final_classification_results_protein_kinase_activity_run4.tsv`
    -   `final_classification_results_protein_kinase_activity_run5.tsv`

**Operations Performed:**

-   **Task Definition:**\
    Categorized responses into three primary tasks:

    -   **Task1 Fully Curatable:** Involves responses indicating fully curatable data.
    -   **Task2 Fully or Partially Curatable:** Combines responses indicating either fully or partially curatable data.
    -   **Task3 Fully Partially Or Language Related:** Combines responses indicating fully curatable, partially curatable, or language-related data.
-   **Metric Calculation:**\
    For each task, the script computed precision, recall, and F1-scores across all five runs, calculating the mean and standard deviation for each metric to assess consistency and reliability.

-   **Negative Examples Reporting:**\
    Quantified entries that did not fit into any of the defined tasks.

**Output Files:** For each data type, the script generated a comprehensive metrics summary TSV file:

-   **Gene Expression:**
    -   `final_metrics_summary_Gene_Expression.tsv`
-   **Protein Kinase Activity:**
    -   `final_metrics_summary_Protein_Kinase.tsv`

**Content of Metrics Summary Files:** Each summary file contained detailed metrics for each task, including:

-   **Combined Counts:**\
    Number of entries combined from each response type (fully curated, partially curated, language related).

-   **Entries Count Total:**\
    Total number of entries per task.

-   **Performance Metrics:**

    -   Precision (mean ± standard deviation)
    -   Recall (mean ± standard deviation)
    -   F1 Score (mean ± standard deviation)
-   **Negative Examples:**\
    Mean and standard deviation of entries not fitting any task.

* * * * *

Data Flow Summary
-----------------

1.  **Data Extraction and Preparation:**

    -   `data_extraction.py` processed raw TSV datasets (`expression_sentence_datasets.tsv` and `kinase_sentence_datasets.tsv`) to generate stratified JSONL files (`final_train.jsonl`, `final_val.jsonl`, `final_test.jsonl`) for both gene expression and protein kinase activity domains. The final test files were manually renamed to `final_expression_test.jsonl` and `final_kinase_test.jsonl` respectively and copied to the root folder of the repository for further processing.
2.  **Model Testing:**

    -   `testing_model.py` utilized the generated test JSONL files to perform five independent classification runs per data type, producing TSV files (`final_classification_results_*_run*.tsv`) capturing the model's classifications.
3.  **Metric Assessment:**

    -   `metrics_assessment.py` analyzed the classification results from all runs to compute and summarize precision, recall, and F1-scores, outputting detailed metrics summary files (`final_metrics_summary_*_Expression.tsv` and `final_metrics_summary_Protein_Kinase.tsv`).

* * * * *

Dependencies
------------

The scripts relied on the following Python packages:

-   `pandas` for data manipulation.
-   `scikit-learn` (`MultiLabelBinarizer`) for label processing.
-   `iterstrat` (`MultilabelStratifiedShuffleSplit`) for stratified data splitting.
-   `openai` for interacting with the OpenAI API during model testing.
-   `tenacity` for implementing retry logic in API calls.
-   `tqdm` for progress visualization during processing.
-   `json`, `os`, `sys`, `logging`, `argparse`, `statistics`, `concurrent.futures`, `threading`, `time` for various auxiliary functionalities.

Ensure all dependencies were installed in the Python environment prior to executing the scripts.

* * * * *

Logging and Error Handling
--------------------------

Each script incorporated logging mechanisms to document processing steps, successes, and any encountered issues. Log files (`final_data_extraction.log`, `final_testing_model_parallel.log`) captured detailed information, facilitating troubleshooting and verification of the workflows.

* * * * *

Conclusion
----------

This repository serves as a detailed archival record of the data preparation and evaluation processes employed in our research on biocuration-relevant sentence classification within gene expression and protein kinase activity domains.

For further inquiries or detailed methodological explanations, please refer to the corresponding sections of our scientific paper.