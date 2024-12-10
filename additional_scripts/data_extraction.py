import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import sys

def create_output_directory(base_dir, type_of_data_filename):
    """
    Creates an output directory for the given data type.
    """
    output_dir = os.path.join(base_dir, type_of_data_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")
    return output_dir

def load_and_filter_data(input_file_path, source_filter, fully_curable_filter, partially_curable_filter, related_language_filter):
    """
    Loads the TSV file and filters it based on the provided criteria.
    Converts curatable columns to integers to ensure correct label assignments.
    """
    print(f"\nLoading data from {input_file_path}...")
    try:
        df = pd.read_csv(input_file_path, sep='\t')
    except FileNotFoundError:
        print(f"Error: File {input_file_path} not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File {input_file_path} is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: File {input_file_path} is malformed.")
        sys.exit(1)

    # Ensure required columns exist
    required_columns = ['SOURCE', 'FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE', 'SENTENCE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in {input_file_path}: {missing_columns}")
        sys.exit(1)
    else:
        print("All required columns are present.")

    # Display initial data statistics
    print(f"Initial data points: {len(df)}")
    print("Initial label value counts:")
    print(df[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].apply(pd.Series.value_counts).fillna(0).astype(int))

    # Filter based on source and curatable columns
    df_filtered = df[
        (df['SOURCE'].isin(source_filter)) &
        (df['FULLY_CURATABLE'].isin(fully_curable_filter)) &
        (df['PARTIALLY_CURATABLE'].isin(partially_curable_filter)) &
        (df['RELATED_LANGUAGE'].isin(related_language_filter))
    ].copy()

    print(f"Data points after filtering by criteria: {len(df_filtered)}")

    # Convert curatable columns to integers to ensure correct comparisons
    for col in ['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']:
        if df_filtered[col].dtype != int:
            try:
                df_filtered[col] = df_filtered[col].astype(int)
                print(f"Converted column '{col}' to integer type.")
            except ValueError:
                print(f"Error: Column '{col}' contains non-integer values in {input_file_path}.")
                sys.exit(1)

    # Verify data types
    print("\nData types after conversion:")
    print(df_filtered[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].dtypes)

    # Additional Debug: Check for unexpected label combinations
    label_combinations = df_filtered[['FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE']].drop_duplicates()
    print("\nUnique label combinations after filtering:")
    print(label_combinations)

    return df_filtered

def assign_labels(row):
    """
    Assigns labels based on the combination of curatable columns.
    Adds 'not_curatable' for {0, 0, 0} combinations.
    """
    labels = []
    if (row['FULLY_CURATABLE'] == 1 and 
        row['PARTIALLY_CURATABLE'] == 1 and 
        row['RELATED_LANGUAGE'] == 1):
        labels.extend(['fully_curatable', 'partially_curatable', 'language_related'])
    elif (row['PARTIALLY_CURATABLE'] == 1 and 
          row['RELATED_LANGUAGE'] == 1 and 
          row['FULLY_CURATABLE'] == 0):
        labels.extend(['partially_curatable', 'language_related'])
    elif (row['RELATED_LANGUAGE'] == 1 and 
          row['FULLY_CURATABLE'] == 0 and 
          row['PARTIALLY_CURATABLE'] == 0):
        labels.append('language_related')
    elif (row['FULLY_CURATABLE'] == 0 and 
          row['PARTIALLY_CURATABLE'] == 0 and 
          row['RELATED_LANGUAGE'] == 0):
        labels.append('not_curatable')
    else:
        raise ValueError(f"Unexpected combination: Row: {row}")
    return labels

def create_assistant_content(labels):
    """
    Generates the assistant's response based on assigned labels.
    Returns exactly one of the four enumerated responses.
    """
    if set(labels) == {'fully_curatable', 'partially_curatable', 'language_related'}:
        return "This sentence contains both fully and partially curatable data as well as terms related to curation."
    elif set(labels) == {'partially_curatable', 'language_related'}:
        return "This sentence does not contain fully curatable data but it does contain partially curatable data and terms related to curation."
    elif set(labels) == {'language_related'}:
        return "This sentence does not contain fully or partially curatable data but does contain terms related to curation."
    elif set(labels) == {'not_curatable'}:
        return "This sentence does not contain fully or partially curatable data or terms related to curation."
    else:
        raise ValueError(f"Unexpected label combination: {labels}")

def generate_jsonl_entries_train_val(df_subset, type_of_data):
    """
    Generates JSONL entries for training and validation data with assistant messages.
    """
    entries = []
    for _, row in df_subset.iterrows():
        try:
            assistant_content = create_assistant_content(row['labels'])
        except ValueError as e:
            print(f"Error generating assistant content: {e}")
            continue  # Skip rows with unexpected label combinations

        entry = {
            "messages": [
                {"role": "system", "content": f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."},
                {"role": "user", "content": row['SENTENCE']},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        entries.append(entry)
    return entries

def generate_jsonl_entries_test(df_subset, type_of_data):
    """
    Generates JSONL entries for testing data without assistant messages.
    """
    entries = []
    for _, row in df_subset.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."},
                {"role": "user", "content": row['SENTENCE']}
                # No assistant message
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
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
        sys.exit(1)

def process_dataset(config, base_output_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Processes a single dataset: loading, filtering, assigning labels, checking for rare labels,
    stratifying into training, validation, and testing sets, and saving JSONL files.
    """
    input_file_path = config['input_file']
    type_of_data = config['type_of_data']
    type_of_data_filename = type_of_data.replace(' ', '_')

    print(f"\n{'='*60}")
    print(f"Processing dataset: {type_of_data} from {input_file_path}")
    print(f"{'='*60}")

    # Load and filter data
    df_filtered = load_and_filter_data(
        input_file_path,
        source_filter=config['source_filter'],
        fully_curable_filter=config['fully_curable_filter'],
        partially_curable_filter=config['partially_curable_filter'],
        related_language_filter=config['related_language_filter']
    )

    if df_filtered.empty:
        print(f"Warning: No data found after filtering for {type_of_data}. Skipping this dataset.")
        return

    # Assign multi-labels
    df_filtered['labels'] = df_filtered.apply(assign_labels, axis=1)

    # Debug: Print sample label assignments
    print("\nSample label assignments:")
    print(df_filtered[['SENTENCE', 'FULLY_CURATABLE', 'PARTIALLY_CURATABLE', 'RELATED_LANGUAGE', 'labels']].head(5))

    # Display label distribution before checking
    print("\nLabel distribution before checking for rare labels:")
    mlb = MultiLabelBinarizer(classes=['fully_curatable', 'partially_curatable', 'language_related', 'not_curatable'])
    Y = mlb.fit_transform(df_filtered['labels'])
    label_counts = Y.sum(axis=0)
    for idx, label in enumerate(mlb.classes_):
        print(f"{label}: {label_counts[idx]}")

    # Identify labels with fewer than two samples
    rare_labels = [label for idx, label in enumerate(mlb.classes_) if label_counts[idx] < 2]

    if rare_labels:
        print("\nError: The following labels have fewer than two samples and cannot be used for stratification:")
        for label in rare_labels:
            count = label_counts[mlb.classes_.tolist().index(label)]
            print(f"\nLabel '{label}' has {count} sample(s).")
            problematic_entries = df_filtered[df_filtered['labels'].apply(lambda x: label in x)]
            print(problematic_entries.to_string(index=False))
        print("\nPlease address these rare labels before proceeding.")
        sys.exit(1)  # Halt execution
    else:
        print("\nNo rare labels detected. Proceeding with stratified split.")

    # Prepare label matrix for stratification
    Y = mlb.transform(df_filtered['labels'])

    # Initialize Multilabel Stratified Shuffle Split for first split (train vs temp)
    msss_train = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state)

    # Perform the first split
    print("\nPerforming first stratified split (Training vs Temp)...")
    try:
        for train_index, temp_index in msss_train.split(df_filtered, Y):
            df_train = df_filtered.iloc[train_index].reset_index(drop=True)
            df_temp = df_filtered.iloc[temp_index].reset_index(drop=True)
    except ValueError as e:
        print(f"Error during first splitting for {type_of_data}: {e}")
        print("Halting execution.")
        sys.exit(1)

    print(f"Training set size: {len(df_train)}")
    print(f"Temporary set size (Validation + Testing): {len(df_temp)}")

    # Initialize Multilabel Stratified Shuffle Split for second split (Validation vs Test)
    msss_val_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(test_size / (val_size + test_size)), random_state=random_state)

    # Perform the second split
    print("\nPerforming second stratified split (Validation vs Testing)...")
    Y_temp = mlb.transform(df_temp['labels'])
    try:
        for val_index, test_index in msss_val_test.split(df_temp, Y_temp):
            df_val = df_temp.iloc[val_index].reset_index(drop=True)
            df_test = df_temp.iloc[test_index].reset_index(drop=True)
    except ValueError as e:
        print(f"Error during second splitting for {type_of_data}: {e}")
        print("Halting execution.")
        sys.exit(1)

    print(f"Validation set size: {len(df_val)}")
    print(f"Testing set size: {len(df_test)}")

    # Create output directory for this dataset
    output_dir = create_output_directory(base_output_dir, type_of_data_filename)

    # Generate JSONL entries
    print("\nGenerating JSONL entries for Training set...")
    training_entries = generate_jsonl_entries_train_val(df_train, type_of_data)
    print("Generating JSONL entries for Validation set...")
    val_entries = generate_jsonl_entries_train_val(df_val, type_of_data)
    print("Generating JSONL entries for Testing set (No assistant content)...")
    test_entries = generate_jsonl_entries_test(df_test, type_of_data)

    # Define file paths
    train_file = os.path.join(output_dir, f'train.jsonl')
    val_file = os.path.join(output_dir, f'val.jsonl')
    test_file = os.path.join(output_dir, f'test.jsonl')  # New test file

    # Save JSONL files
    save_jsonl(training_entries, train_file)
    save_jsonl(val_entries, val_file)
    save_jsonl(test_entries, test_file)  # Save test entries

    # Verification of label distribution in splits
    print("\nVerification of label distributions:")

    def print_label_distribution(df_subset, subset_name, mlb):
        Y_subset = mlb.transform(df_subset['labels'])
        distribution = Y_subset.sum(axis=0)
        print(f"\n{subset_name} set label distribution:")
        for idx, label in enumerate(mlb.classes_):
            print(f"{label}: {distribution[idx]}")

    print_label_distribution(df_train, "Training", mlb)
    print_label_distribution(df_val, "Validation", mlb)
    print_label_distribution(df_test, "Testing", mlb)  # Verify test set

    print(f"\nCompleted processing for dataset: {type_of_data}")

def main():
    # Base output directory where all splits will be saved
    base_output_dir = 'multi_label_split_data'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created base output directory: {base_output_dir}")
    else:
        print(f"Base output directory already exists: {base_output_dir}")

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
        sys.exit(1)

    # Process each dataset
    for config in datasets:
        process_dataset(config, base_output_dir, train_size, val_size, test_size, random_state)

    print("\nAll datasets have been processed successfully.")

if __name__ == "__main__":
    main()
