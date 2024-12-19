import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import sys
import logging

# ------------------------------
# Setup Logging
# ------------------------------
logging.basicConfig(
    filename='final_data_extraction.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def create_output_directory(base_dir, type_of_data_filename):
    """
    Creates an output directory for the given data type.
    """
    output_dir = os.path.join(base_dir, type_of_data_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        logging.info(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")
        logging.info(f"Directory already exists: {output_dir}")
    return output_dir

def load_and_filter_data(input_file_path, source_filter, fully_curable_filter, partially_curable_filter, related_language_filter, verbose=False):
    """
    Loads the TSV file and filters it based on the provided criteria.
    Converts curatable columns to integers to ensure correct label assignments.
    """
    print(f"\nLoading data from {input_file_path}...")
    try:
        df = pd.read_csv(input_file_path, sep='\t')
        logging.info(f"Loaded data from {input_file_path}. Total entries: {len(df)}")
    except FileNotFoundError:
        logging.error(f"File {input_file_path} not found.")
        print(f"Error: File {input_file_path} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"File {input_file_path} is empty.")
        print(f"Error: File {input_file_path} is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        logging.error(f"File {input_file_path} is malformed.")
        print(f"Error: File {input_file_path} is malformed.")
        sys.exit(1)

    # Ensure required columns exist
    required_columns = ['SOURCE', 'FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE', 'SENTENCE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns in {input_file_path}: {missing_columns}")
        print(f"Error: Missing required columns in {input_file_path}: {missing_columns}")
        sys.exit(1)
    else:
        print("All required columns are present.")
        logging.info("All required columns are present.")

    # Display initial data statistics
    print(f"Initial data points: {len(df)}")
    logging.info(f"Initial data points: {len(df)}")
    print("Initial label value counts:")
    label_counts_initial = df[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].apply(pd.Series.value_counts).fillna(0).astype(int)
    print(label_counts_initial)
    logging.info(f"Initial label value counts:\n{label_counts_initial}")

    # Filter based on source and curatable columns
    df_filtered = df[
        (df['SOURCE'].isin(source_filter)) &
        (df['FULLY_CURATABLE'].isin(fully_curable_filter)) &
        (df['PARTIALLY_CURATABLE'].isin(partially_curable_filter)) &
        (df['RELATED_LANGUAGE'].isin(related_language_filter))
    ].copy()

    print(f"Data points after filtering by criteria: {len(df_filtered)}")
    logging.info(f"Data points after filtering by criteria: {len(df_filtered)}")

    # Convert curatable columns to integers to ensure correct comparisons
    for col in ['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']:
        if df_filtered[col].dtype != int:
            try:
                df_filtered[col] = df_filtered[col].astype(int)
                print(f"Converted column '{col}' to integer type.")
                logging.info(f"Converted column '{col}' to integer type.")
            except ValueError:
                logging.error(f"Column '{col}' contains non-integer values in {input_file_path}.")
                print(f"Error: Column '{col}' contains non-integer values in {input_file_path}.")
                sys.exit(1)

    # Verify data types
    print("\nData types after conversion:")
    print(df_filtered[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].dtypes)
    logging.info(f"Data types after conversion:\n{df_filtered[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].dtypes}")

    # Additional Debug: Check for unexpected label combinations
    label_combinations = df_filtered[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].drop_duplicates()
    print("\nUnique label combinations after filtering:")
    print(label_combinations)
    logging.info(f"Unique label combinations after filtering:\n{label_combinations}")

    return df_filtered

def assign_labels(row):
    """
    Assigns labels based on the combination of curatable columns.
    Adds 'not_curatable' for {0, 0, 0} combinations.
    """
    labels = []
    if (row['FULLY_CURATABLE'] == 1 and 
        row['PARTIALLY_CURATABLE'] == 0 and 
        row['RELATED_LANGUAGE'] == 0):
        labels.append('fully_curatable')
    elif (row['PARTIALLY_CURATABLE'] == 1 and 
          row['RELATED_LANGUAGE'] == 0 and 
          row['FULLY_CURATABLE'] == 0):
        labels.append('partially_curatable')
    elif (row['RELATED_LANGUAGE'] == 1 and 
          row['FULLY_CURATABLE'] == 0 and 
          row['PARTIALLY_CURATABLE'] == 0):
        labels.append('language_related')
    elif (row['FULLY_CURATABLE'] == 0 and 
          row['PARTIALLY_CURATABLE'] == 0 and 
          row['RELATED_LANGUAGE'] == 0):
        labels.append('not_curatable')
    else:
        # Log the unexpected combination and assign 'unknown' label
        logging.warning(f"Unexpected combination: {row.to_dict()}")
        labels.append('unknown')
    return labels

def create_assistant_content(labels):
    """
    Generates the assistant's response based on assigned labels.
    Returns exactly one of the four enumerated responses.
    """
    if set(labels) == {'fully_curatable'}:
        return "This sentence only contains fully curatable data."
    elif set(labels) == {'partially_curatable'}:
        return "This sentence only contains partially curatable data."
    elif set(labels) == {'language_related'}:
        return "This sentence is not fully or partially curatable, but it contains terms related to the datatype."
    elif set(labels) == {'not_curatable'}:
        return "This sentence does not contain fully or partially curatable data, or terms related to the datatype."
    else:
        # Handle 'unknown' and any other unexpected combinations
        return "The classification of this sentence is ambiguous or does not fit predefined categories."

def get_prompt_instructions_for_type(type_of_data):
    """
    Returns a domain-specific prompt that instructs the model on how to classify sentences.
    These instructions are more thorough and are tailored to either gene expression or
    protein kinase activity, as requested.
    """
    if type_of_data == "gene expression":
        return """
    You are an expert in biomedical natural language processing, specializing in identifying biocuration-relevant sentences from the biomedical literature. Your goal is to classify a given sentence according to whether it contains information suitable for biocuration tasks focused on gene expression. Consider that the sentences come from full-text scientific articles and may reflect various experimental contexts, including direct experimental results, summarized findings, previously published information, and related methodological details.

    Background
    Biocuration involves extracting high-quality, trustworthy information about gene function and expression patterns from published references and integrating these data into knowledgebases. Professional curators identify relevant textual evidence—often at the sentence level—from primary literature to ensure that key experimental findings are captured in standardized ontologies and associated with original sources.

    This classification aids in guiding curators and authors to statements that can be used directly or indirectly to create annotations. With the large volume of literature and the complexity of biomedical data, semi-automated tools help make curation more efficient while maintaining quality.

    Data Type of Interest: Gene Expression
    Relevant sentences may include:
    - Mention of a specific gene or gene product by name.
    - Terms or phrases indicating gene expression (e.g., “expressed,” “localized,” “detected in,” “present in”).
    - Spatial or temporal context for the expression (e.g., a particular tissue, cell type, developmental stage).

    Fully Curatable Gene Expression Data:
    A fully curatable sentence typically includes all elements needed for a direct annotation: the gene, evidence of its expression, and the anatomical/cellular location and/or the developmental stage.

    Partially Curatable Gene Expression Data:
    A partially curatable sentence may mention some relevant information but not all. Additional sentences would be needed to complete the annotation.

    Curation-Related Terms (Related Language):
    Some sentences might not provide direct or partial annotation details but still contain terms associated with gene expression experiments (e.g., reporter constructs, in situ hybridization, qPCR methods, antibodies for detection). These sentences signal relevance but are not directly curatable.

    Non-Curation-Related Content:
    If a sentence contains no gene expression-relevant information, does not name genes or mention expression, and does not include methodological details relevant to gene expression, it falls into this category.

    Additional Notes:
    - Some sentences summarize previously published results or mention mutant backgrounds. If they contain expression-related terms but are not suitable for direct annotation, classify them as related language.
    - If a single sentence lacks all critical pieces but is on-topic, it is partially curatable or related language depending on the details.
    - Consider if the sentence reports actual experimental findings or only provides background/methodological context.

    Classify the sentence into one of four categories. Please return EXACTLY one of the four following sentences as your classification with no extra text:
    This sentence only contains fully curatable data.
    This sentence only contains partially curatable data.
    This sentence is not fully or partially curatable, but it contains terms related to the datatype.
    This sentence does not contain fully or partially curatable data, or terms related to the datatype.
        """.strip()

    elif type_of_data == "protein kinase activity":
        return """
    You are an expert in biomedical natural language processing, specializing in identifying biocuration-relevant sentences from the biomedical literature. Your goal is to classify a given sentence according to whether it contains information suitable for biocuration tasks focused on protein kinase activity. Consider that the sentences come from full-text scientific articles and may reflect various experimental contexts, including direct experimental results, summarized findings, previously published information, and related methodological details.

    Background
    Biocuration involves extracting high-quality, trustworthy information about protein function and activity, including protein kinase activity, from published references and integrating these data into knowledgebases. Professional curators identify relevant textual evidence—often at the sentence level—from primary literature to ensure that key experimental findings are captured in standardized ontologies and associated with original sources.

    This classification helps guide curators and authors to statements that can be used directly or indirectly to create annotations, aiding in efficient and comprehensive integration of kinase activity data into knowledgebases.

    Data Type of Interest: Protein Kinase Activity
    Relevant sentences may include:
    - Mention of a protein kinase by name.
    - Indications of phosphorylation events or enzymatic activity (e.g., “phosphorylates,” “kinase assay,” “in vitro phosphorylation”).
    - References to substrates or experimental conditions supporting enzymatic activity.

    Fully Curatable Protein Kinase Data:
    A fully curatable sentence provides all key elements needed for annotation, such as naming the kinase, indicating enzymatic activity, and possibly referencing an experimental assay or substrate.

    Partially Curatable Protein Kinase Data:
    A partially curatable sentence provides some information (e.g., mentions the kinase or substrate) but not all. Additional sentences would be needed to fully annotate the activity.

    Curation-Related Terms (Related Language):
    Some sentences may reference tools, methods, or general concepts related to protein kinase activity (e.g., mentioning phosphorylation or kinase assays) without providing enough detail for direct or partial annotation. These guide the curator but are not directly or partially curatable.

    Non-Curation-Related Content:
    If the sentence does not mention any kinase activity, phosphorylation, substrates, or relevant methods, it is non-curatable.

    Additional Notes:
    - Sentences summarizing previously published work or mentioning mutant backgrounds that contain relevant terms but are not suitable for annotation should be considered related language.
    - If information needed for a full annotation is spread across multiple sentences and the current sentence lacks critical details, it is partially curatable or related language depending on its content.
    - Distinguish between actual experimental findings and mere methodological/hypothetical statements.

    Classify the sentence into one of four categories. Please return EXACTLY one of the four following sentences as your classification with no extra text:
    This sentence only contains fully curatable data.
    This sentence only contains partially curatable data.
    This sentence is not fully or partially curatable, but it contains terms related to the datatype.
    This sentence does not contain fully or partially curatable data, or terms related to the datatype.
        """.strip()
    else:
        # Fallback, though not expected to be used here
        return "You are an expert classifier. Classify the sentence as fully curatable, partially curatable, curation-related, or non-curatable."

def generate_jsonl_entries(df_subset, type_of_data):
    """
    Generates JSONL entries for all subsets (Training, Validation, Test) with assistant messages.
    Uses domain-specific, thorough instructions based on the data type, without mentioning function calls.
    """
    prompt_instructions = get_prompt_instructions_for_type(type_of_data)

    entries = []
    for _, row in df_subset.iterrows():
        assistant_content = create_assistant_content(row['labels'])
        entry = {
            "messages": [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": row['SENTENCE']},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        entries.append(entry)
    return entries

def save_jsonl(entries, file_path):
    """
    Saves the list of JSON entries to a JSONL file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Successfully saved JSONL file: {file_path}")
        logging.info(f"Successfully saved JSONL file: {file_path}")
    except IOError as e:
        logging.error(f"Error writing to file {file_path}: {e}")
        print(f"Error writing to file {file_path}: {e}")
        sys.exit(1)

def process_dataset(config, base_output_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Processes a single dataset: loading, filtering, assigning labels, checking for rare labels,
    stratifying into training, validation, and testing sets, and saving JSONL files.
    Includes assistant messages in all subsets.
    """
    input_file_path = config['input_file']
    type_of_data = config['type_of_data']
    type_of_data_filename = type_of_data.replace(' ', '_')

    print(f"\n{'='*60}")
    print(f"Processing dataset: {type_of_data} from {input_file_path}")
    print(f"{'='*60}")
    logging.info(f"Processing dataset: {type_of_data} from {input_file_path}")

    # Load and filter data
    df_filtered = load_and_filter_data(
        input_file_path,
        source_filter=config['source_filter'],
        fully_curable_filter=config['fully_curable_filter'],
        partially_curable_filter=config['partially_curable_filter'],
        related_language_filter=config['related_language_filter'],
        verbose=False  # Set to True if needed
    )

    if df_filtered.empty:
        print(f"Warning: No data found after filtering for {type_of_data}. Skipping this dataset.")
        logging.warning(f"No data found after filtering for {type_of_data}. Skipping this dataset.")
        return

    # Assign multi-labels
    df_filtered['labels'] = df_filtered.apply(assign_labels, axis=1)

    # Handle 'unknown' labels
    initial_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['labels'].apply(lambda x: x != ['unknown'])]
    excluded_count = initial_count - len(df_filtered)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} entries with 'unknown' label.")
        logging.info(f"Excluded {excluded_count} entries with 'unknown' label.")

    # Debug: Print sample label assignments
    print("\nSample label assignments:")
    print(df_filtered[['SENTENCE', 'FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE', 'labels']].head(5))
    logging.info(f"Sample label assignments:\n{df_filtered[['SENTENCE', 'FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE', 'labels']].head(5)}")

    # Display label distribution before checking
    print("\nLabel distribution before checking for rare labels:")
    mlb = MultiLabelBinarizer(classes=['fully_curatable', 'partially_curatable', 'language_related', 'not_curatable'])
    Y = mlb.fit_transform(df_filtered['labels'])
    label_counts = Y.sum(axis=0)
    label_distribution = pd.Series(label_counts, index=mlb.classes_)
    print(label_distribution)
    logging.info(f"Label distribution before checking for rare labels:\n{label_distribution}")

    # Identify labels with fewer than two samples
    rare_labels = [label for idx, label in enumerate(mlb.classes_) if label_counts[idx] < 2]

    if rare_labels:
        print("\nError: The following labels have fewer than two samples and cannot be used for stratification:")
        logging.error(f"The following labels have fewer than two samples and cannot be used for stratification: {rare_labels}")
        for label in rare_labels:
            count = label_distribution[label]
            print(f"\nLabel '{label}' has {count} sample(s).")
            logging.error(f"Label '{label}' has {count} sample(s).")
            problematic_entries = df_filtered[df_filtered['labels'].apply(lambda x: label in x)]
            print(problematic_entries.to_string(index=False))
            logging.error(f"Problematic entries for label '{label}':\n{problematic_entries.to_string(index=False)}")
        print("\nPlease address these rare labels before proceeding.")
        sys.exit(1)  # Halt execution
    else:
        print("\nNo rare labels detected. Proceeding with stratified split.")
        logging.info("No rare labels detected. Proceeding with stratified split.")

    # Prepare label matrix for stratification
    Y = mlb.transform(df_filtered['labels'])

    # Initialize Multilabel Stratified Shuffle Split for first split (train vs temp)
    msss_train = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state)

    # Perform the first split
    print("\nPerforming first stratified split (Training vs Temp)...")
    logging.info("Performing first stratified split (Training vs Temp)...")
    try:
        for train_index, temp_index in msss_train.split(df_filtered, Y):
            df_train = df_filtered.iloc[train_index].reset_index(drop=True)
            df_temp = df_filtered.iloc[temp_index].reset_index(drop=True)
    except ValueError as e:
        logging.error(f"Error during first splitting for {type_of_data}: {e}")
        print(f"Error during first splitting for {type_of_data}: {e}")
        print("Halting execution.")
        sys.exit(1)

    print(f"Training set size: {len(df_train)}")
    print(f"Temporary set size (Validation + Testing): {len(df_temp)}")
    logging.info(f"Training set size: {len(df_train)}")
    logging.info(f"Temporary set size (Validation + Testing): {len(df_temp)}")

    # Initialize Multilabel Stratified Shuffle Split for second split (Validation vs Test)
    msss_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(test_size / (val_size + test_size)), random_state=random_state)

    # Perform the second split
    print("\nPerforming second stratified split (Validation vs Testing)...")
    logging.info("Performing second stratified split (Validation vs Testing)...")
    Y_temp = mlb.transform(df_temp['labels'])
    try:
        for val_index, test_index in msss_val_test.split(df_temp, Y_temp):
            df_val = df_temp.iloc[val_index].reset_index(drop=True)
            df_test = df_temp.iloc[test_index].reset_index(drop=True)
    except ValueError as e:
        logging.error(f"Error during second splitting for {type_of_data}: {e}")
        print(f"Error during second splitting for {type_of_data}: {e}")
        print("Halting execution.")
        sys.exit(1)

    print(f"Validation set size: {len(df_val)}")
    print(f"Testing set size: {len(df_test)}")
    logging.info(f"Validation set size: {len(df_val)}")
    logging.info(f"Testing set size: {len(df_test)}")

    # Create output directory for this dataset
    output_dir = create_output_directory(base_output_dir, type_of_data_filename)

    # Generate JSONL entries
    print("\nGenerating JSONL entries for Training set...")
    logging.info("Generating JSONL entries for Training set...")
    training_entries = generate_jsonl_entries(df_train, type_of_data)
    print("Generating JSONL entries for Validation set...")
    logging.info("Generating JSONL entries for Validation set...")
    val_entries = generate_jsonl_entries(df_val, type_of_data)
    print("Generating JSONL entries for Testing set...")
    logging.info("Generating JSONL entries for Testing set...")
    test_entries = generate_jsonl_entries(df_test, type_of_data)

    # Define file paths
    train_file = os.path.join(output_dir, f'train.jsonl')
    val_file = os.path.join(output_dir, f'val.jsonl')
    test_file = os.path.join(output_dir, f'test.jsonl')  # Test file with assistant messages

    # Save JSONL files
    save_jsonl(training_entries, train_file)
    save_jsonl(val_entries, val_file)
    save_jsonl(test_entries, test_file)  # Save test entries with assistant messages

    # Verification of label distribution in splits
    print("\nVerification of label distributions:")
    logging.info("Verification of label distributions:")

    def print_label_distribution(df_subset, subset_name, mlb):
        Y_subset = mlb.transform(df_subset['labels'])
        distribution = Y_subset.sum(axis=0)
        distribution_series = pd.Series(distribution, index=mlb.classes_)
        print(f"\n{subset_name} set label distribution:")
        print(distribution_series)
        logging.info(f"{subset_name} set label distribution:\n{distribution_series}")

    print_label_distribution(df_train, "Training", mlb)
    print_label_distribution(df_val, "Validation", mlb)
    print_label_distribution(df_test, "Testing", mlb)  # Verify test set

    print(f"\nCompleted processing for dataset: {type_of_data}")
    logging.info(f"Completed processing for dataset: {type_of_data}")

def main():
    # Base output directory where all splits will be saved
    base_output_dir = 'multi_label_split_data'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created base output directory: {base_output_dir}")
        logging.info(f"Created base output directory: {base_output_dir}")
    else:
        print(f"Base output directory already exists: {base_output_dir}")
        logging.info(f"Base output directory already exists: {base_output_dir}")

    # List of dataset configurations
    datasets = [
        {
            'input_file': 'expression_sentence_datasets.tsv',
            'type_of_data': 'gene expression',
            'source_filter': ['GOLD', '1000', 'NEGATIVE'],
            'fully_curable_filter': [0, 1],      # Include both 0 and 1
            'partially_curable_filter': [0, 1],
            'related_language_filter': [0, 1]
        },
        {
            'input_file': 'kinase_sentence_datasets.tsv',
            'type_of_data': 'protein kinase activity',
            'source_filter': ['GOLD', '1000', 'NEGATIVE'],
            'fully_curable_filter': [0, 1],      # Include both 0 and 1
            'partially_curable_filter': [0, 1],
            'related_language_filter': [0, 1]
        }
    ]

    # Split parameters
    train_size = 0.7  # 70% for training
    val_size = 0.15   # 15% for validation
    test_size = 0.15  # 15% for testing
    random_state = 42  # For reproducibility

    # Validate that split sizes sum to 1
    total_size = train_size + val_size + test_size
    if not abs(total_size - 1.0) < 1e-6:
        print("Error: The sum of train_size, val_size, and test_size must be 1.")
        logging.error("The sum of train_size, val_size, and test_size must be 1.")
        sys.exit(1)

    # Process each dataset
    for config in datasets:
        process_dataset(
            config,
            base_output_dir,
            train_size,
            val_size,
            test_size,
            random_state
        )

    print("\nAll datasets have been processed successfully.")
    logging.info("All datasets have been processed successfully.")

if __name__ == "__main__":
    main()
