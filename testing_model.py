#!/usr/bin/env python

import json
import random
import pandas as pd
import openai  # Using the latest OpenAI library
from tenacity import retry, wait_random_exponential, stop_after_attempt
import sys
from tqdm import tqdm
import os
import concurrent.futures
import threading
import logging
import time

# ------------------------------
# Configuration for Multiple Data Types
# ------------------------------
DATA_TYPES = [
    {
        'type_of_data': 'gene expression',
        'input_file': 'final_expression_test.jsonl',
        'model_name': 'ft:gpt-4o-2024-08-06:alliance-of-genome-resources:expression-12:AfaszfkK'
    },
    {
        'type_of_data': 'protein kinase activity',
        'input_file': 'final_kinase_test.jsonl',
        'model_name': 'ft:gpt-4o-2024-08-06:alliance-of-genome-resources:kinase-12:Afar4SrV'
    }
]

# Maximum number of classification attempts per sentence
MAX_CLASSIFICATION_ATTEMPTS = 5

def parse_arguments():
    if len(sys.argv) not in [2, 3, 4, 5, 6]:
        print("Usage: python testing_model.py [-v] [-s] <your_openai_api_key>")
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
    
    if api_key_index >= len(sys.argv):
        print("Error: Missing OpenAI API key.")
        sys.exit(1)
    
    api_key = sys.argv[api_key_index]
    return api_key, verbose, subset

def initialize_openai_client(api_key):
    openai.api_key = api_key
    return openai

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_sentence",
                "description": "Classify the sentence based on curatable data types.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "enum": [
                                "This sentence only contains fully curatable data.",
                                "This sentence only contains partially curatable data.",
                                "This sentence is not fully or partially curatable, but it contains terms related to the datatype.",
                                "This sentence does not contain fully or partially curatable data, or terms related to the datatype."
                            ],
                            "description": "The classification result of the sentence."
                        }
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            }
        }
    ]

def load_jsonl(file_path):
    testing_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file, start=1):
                entry = json.loads(line)
                if 'id' not in entry:
                    entry['id'] = idx
                testing_data.append(entry)
        print(f"Loaded {len(testing_data)} entries from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as jde:
        print(f"JSON decode error in file {file_path}: {jde}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        sys.exit(1)
    return testing_data

def get_prompt_instructions_for_type(type_of_data):
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

    Please use the tool function to return an appropriate answer.
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
    Some sentences may reference tools, methods, or general concepts related to protein kinase activity (e.g., mentioning phosphorylation or kinase assays) without providing enough detail for direct or partial annotation. These sentences signal relevance but are not directly or partially curatable.

    Non-Curation-Related Content:
    If the sentence does not mention any kinase activity, phosphorylation, substrates, or relevant methods, it is non-curatable.

    Additional Notes:
    - Sentences summarizing previously published work or mentioning mutant backgrounds that contain relevant terms but are not suitable for annotation should be considered related language.
    - If information needed for a full annotation is spread across multiple sentences and the current sentence lacks critical details, it is partially curatable or related language depending on its content.
    - Distinguish between actual experimental findings and mere methodological/hypothetical statements.

    Please use the tool function to return an appropriate answer.
        """.strip()

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools, model):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0,
        tool_choice={"type": "function", "function": {"name": "classify_sentence"}}
    )
    return response

def test_model_concurrent(tools, testing_data, model_name, assistant_description, verbose, run_number, data_type_filename, type_of_data, max_workers=1):
    total = len(testing_data)
    unexpected_responses = []
    
    results = []
    lock = threading.Lock()

    def process_entry(entry, type_of_data):
        nonlocal unexpected_responses
        attempts = 0
        while attempts < MAX_CLASSIFICATION_ATTEMPTS:
            attempts += 1
            # Use only system and user messages, no assistant message.
            prompt_instructions = get_prompt_instructions_for_type(type_of_data)
            # The user message is just the sentence from entry['messages'][1]['content'].
            # We do NOT include the assistant message from the training set.
            # The expected response is still entry['messages'][-1]['content'] for comparison.
            expected_response = entry["messages"][-1]["content"]
            user_sentence = entry["messages"][1]["content"]

            messages = [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": user_sentence}
            ]

            if verbose:
                print(f"\nRun {run_number}, Entry {entry['id']}:")
                print("Request Messages:")
                print(json.dumps(messages, indent=2))
            
            try:
                completion = chat_completion_request(messages, tools, model_name)
                if verbose:
                    print("Response Object:")
                    print(json.dumps(completion.model_dump(), indent=2))
                
                tool_calls = completion.choices[0].message.tool_calls
                if not tool_calls or len(tool_calls) == 0:
                    if verbose:
                        print(f"Run {run_number}, Entry {entry['id']}: No function call made.")
                    continue  # Retry

                tool_call = tool_calls[0]
                function_args_str = tool_call.function.arguments
                function_call_response = json.loads(function_args_str)
                content = function_call_response.get("content", "").strip()

                if verbose:
                    print(f"Assistant Response Content: {content}")

                valid_responses = [
                    "This sentence only contains fully curatable data.",
                    "This sentence only contains partially curatable data.",
                    "This sentence is not fully or partially curatable, but it contains terms related to the datatype.",
                    "This sentence does not contain fully or partially curatable data, or terms related to the datatype."
                ]
                if content not in valid_responses:
                    if verbose:
                        print(f"Run {run_number}, Entry {entry['id']}: Invalid response received: {content}")
                    if attempts >= MAX_CLASSIFICATION_ATTEMPTS:
                        print(f"Run {run_number}, Entry {entry['id']}: Failed to get a valid response after {MAX_CLASSIFICATION_ATTEMPTS} attempts.")
                    continue
                else:
                    with lock:
                        result = {
                            "sentence": user_sentence,
                            "expected_response": expected_response,
                            "assistant_response": content
                        }
                        results.append(result)
                    break

            except Exception as e:
                if verbose:
                    print(f"Exception: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda entry: process_entry(entry, type_of_data), testing_data),
            total=total,
            desc=f"Processing Run {run_number}",
            unit="sentence"
        ))

    output_file_path = f'final_classification_results_{data_type_filename}_run{run_number}.tsv'
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, sep='\t', index=False)
    print(f"Successfully saved TSV file: {output_file_path}")

    unexpected_responses_file_path = f'unexpected_responses_{data_type_filename}_run{run_number}.txt'
    with open(unexpected_responses_file_path, 'w', encoding='utf-8') as file:
        for response in unexpected_responses:
            file.write(response + '\n')
    print(f"Successfully saved unexpected responses file: {unexpected_responses_file_path}")

    return {
        # Metrics are removed as per requirements
    }

def main():
    api_key, verbose, subset = parse_arguments()

    initialize_openai_client(api_key)

    tools = get_tools()

    NUM_RUNS = 5
    MAX_WORKERS = 1  # Ensured to be 1 as per earlier user requirement for sequential processing

    logging.basicConfig(
        filename='final_testing_model_parallel.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG if verbose else logging.INFO
    )

    for data_type_config in DATA_TYPES:
        type_of_data = data_type_config['type_of_data']
        type_of_data_filename = type_of_data.replace(' ', '_')
        input_file = data_type_config['input_file']
        model_name = data_type_config['model_name']
        assistant_description = f"This GPT assistant is an expert biocurator and sentence-level classifier for {type_of_data}."

        print(f"\n{'='*60}")
        print(f"Preparing to process data type: {type_of_data}")
        print(f"Input JSONL file: {input_file}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        testing_data = load_jsonl(input_file)

        if subset:
            if len(testing_data) < 10:
                print(f"Warning: Requested subset size of 10, but only {len(testing_data)} entries are available.")
                subset_size = len(testing_data)
            else:
                subset_size = 10  # Corrected to select 10 as per the option
            testing_data = random.sample(testing_data, subset_size)
            print(f"Selected a random subset of {len(testing_data)} entries for testing.")

        for run in range(1, NUM_RUNS + 1):
            print(f"\n--- Starting Test Run {run} for {type_of_data} ---")
            logging.info(f"Starting Test Run {run} for {type_of_data}")

            _ = test_model_concurrent(
                tools=tools,
                testing_data=testing_data,
                model_name=model_name,
                assistant_description=assistant_description,
                verbose=verbose,
                run_number=run,
                data_type_filename=type_of_data_filename,
                type_of_data=type_of_data,
                max_workers=MAX_WORKERS
            )

            print(f"Completed Test Run {run} for {type_of_data}.\n")
            logging.info(f"Completed Test Run {run} for {type_of_data}.")

    print("\nAll test runs have been completed successfully.")
    logging.info("All test runs have been completed successfully.")

if __name__ == "__main__":
    main()
