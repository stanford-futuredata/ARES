import argparse
import ast
import copy
import csv
import json
import pdb
import re
import requests
import sys
import time
import warnings
import together

import numpy as np
import openai
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, BitsAndBytesConfig)

from ares.LLM_as_a_Judge_Adaptation.Filter_Synthetic_Queries import (filter_synthetic_queries,
                                                                      generate_additional_negatives,
                                                                      generate_additional_positives,
                                                                      generate_index, get_embedding)
from ares.LLM_as_a_Judge_Adaptation.LLM_Generation_Functions import (check_generated_answer,
                                                                      generate_answer_from_context,
                                                                      generate_answer_llm_approach,
                                                                      generate_contradictory_answer_examples,
                                                                      generate_contradictory_answer_from_context,
                                                                      generate_synthetic_query_llm_approach,
                                                                      generate_synthetic_query_openai_approach)

from ares.LLM_as_a_Judge_Adaptation.LLM_Synthetic_Generation import (generate_synthetic_query_api_approach,
                                                                    generate_synthetic_answer_api_approach,
                                                                    generate_synthetic_contradictory_answers_api_approach)

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_colwidth', None) 

def clean_document(document: str) -> str:
    """
    Cleans the input document by removing unnecessary whitespace characters and replacing certain punctuation.

    Args:
        document (str): The original document text that needs to be cleaned.

    Returns:
        str: The cleaned document text.
    """
    # Replace carriage returns and tabs with a space, and reduce multiple newlines to a single newline
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r", " ").replace("\t", " ")).strip()
    # Replace equals signs and hyphens with spaces
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    # Reduce multiple spaces to a single space
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    # Join words with a single space (this line seems redundant and could be removed if confirmed)
    cleaned_document = (" ").join(cleaned_document.split(" "))  # [:512] - this part is commented out and can be ignored or removed
    return cleaned_document

def validate_input_file(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """
    Validates that the DataFrame contains all required columns. Exits the program if any are missing.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): A list of strings representing the column names that are required in the DataFrame.

    Returns:
        bool: True if the DataFrame contains all required columns, otherwise the program will exit with an error.
    """
    # Identify any missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    # Exit the program with an error message if there are missing columns
    if missing_columns:
        sys.exit(f"Error: The DataFrame is missing the following required column(s): {', '.join(missing_columns)}.")
    return True

def load_model(model_choice: str, api_model: bool) -> tuple:
    """
    Loads the specified model and tokenizer, and sets the model to evaluation mode on the appropriate device.

    Args:
        model_choice (str): The model identifier to load from the Hugging Face model hub.

    Returns:
        tuple: A tuple containing the model, tokenizer, and device.
    """

    if api_model: 
        return model_choice, None, None

    # Load the tokenizer and model from the specified model choice
    tokenizer = AutoTokenizer.from_pretrained(model_choice)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

    # Disable gradient calculations and set the model to evaluation mode
    torch.no_grad()
    model.eval()

    # Set the device to CUDA if available
    device = torch.device("cuda:0")
    model.to(device)
    
    return model, tokenizer, device

def load_documents(document_filepath: str, clean_documents: bool, documents_sampled: int) -> pd.DataFrame:
    """
    Loads and processes documents for synthetic query and answer generation.

    Args:
        document_filepath (str): The path to the document file.
        clean_documents (bool): Flag indicating whether to clean the documents.
        documents_sampled (int): The number of documents to sample.

    Returns:
        pd.DataFrame: A DataFrame containing the processed documents.
    """
    documents = []
    required_columns = ['Query', 'Document', 'Answer']

    if "docs_aws" in document_filepath:
        with open(document_filepath, "r") as json_file:
            json_data = json.load(json_file)
            documents = [x['text'] for x in json_data]

            # Clean document
            if clean_documents:
                documents = [clean_document(text) for text in documents]

        documents = pd.DataFrame(documents, columns=["document"])
    else:
        if not document_filepath.endswith('.tsv'):
            sys.exit(f"Error: The file {document_filepath} is not a TSV file.")
        try:
            documents = pd.read_csv(document_filepath, sep="\t")
            validate_input_file(documents, required_columns)
            documents.rename(columns={"Document": "document"}, inplace=True)
            documents['document'] = documents['document'].str.strip()
        except Exception as e:
            sys.exit(f"Error reading the file {document_filepath}: {e}")

    initial_count = len(documents)
    documents = documents[documents['document'].str.split().apply(len) >= 50]  # Filter documents with less than 50 words.
    after_filter_count = len(documents)

    count = initial_count - after_filter_count

    if after_filter_count == 0:
        sys.exit("All documents were less than 50 words, please provide a dataset with documents containing more than 50 words.")

    if documents_sampled > initial_count:
        print(f"\nThe `documents_sampled` parameter ({documents_sampled}) exceeds the available number of documents ({initial_count}). Sampling will be adjusted to the maximum available documents ({initial_count}).\n")
        documents_sampled = initial_count

    if count > 0:
        print(f"Filtered out {count} documents because they had less than 50 words.")
        if documents_sampled > after_filter_count: 
            print(f"Document sample is greater than document count. Sampling will be adjusted to {after_filter_count} documents\n")
            documents_sampled = after_filter_count

    documents = documents.sample(n=documents_sampled)

    return documents

def load_few_shot_prompt(few_shot_prompt_filename: str, for_fever_dataset: bool, for_wow_dataset: bool) -> tuple[str, int]:
    """
    Loads and processes a few-shot prompt from a TSV file.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.
        for_fever_dataset (bool): Flag indicating if the prompts are for the FEVER dataset.
        for_wow_dataset (bool): Flag indicating if the prompts are for the WoW dataset.

    Returns:
        tuple[str, int]: A tuple containing the few-shot examples as a string and the length of the few-shot prompt.
    """
    few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt = few_shot_prompt[few_shot_prompt['Context_Relevance_Label'] == "[[Yes]]"]
    
    if "Query" not in few_shot_prompt:
        few_shot_prompt['Query'] = few_shot_prompt['Question']

    length_of_fewshot_prompt = len(few_shot_prompt)
    few_shot_examples = ""

    for row in range(len(few_shot_prompt)):
        few_shot_examples += f"Example {row + 1}:\n"
        few_shot_examples += f"Document: {clean_document(few_shot_prompt.iloc[row]['Document'])}\n"
        
        if for_fever_dataset:
            few_shot_examples += f"Statement: {few_shot_prompt.iloc[row]['Query']}\n\n"
        elif for_wow_dataset:
            few_shot_examples += f"Dialogue: {few_shot_prompt.iloc[row]['Query']}\n\n"
        else:
            few_shot_examples += f"Question: {few_shot_prompt.iloc[row]['Query']}\n\n"

    return few_shot_examples, length_of_fewshot_prompt

def generate_contradictory_answers(few_shot_prompt_filename: str, for_fever_dataset: bool, for_wow_dataset: bool) -> str:
    """
    Generates few-shot examples for contradictory answers based on the provided dataset.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.
        for_fever_dataset (bool): Flag indicating if the prompts are for the FEVER dataset.
        for_wow_dataset (bool): Flag indicating if the prompts are for the WoW dataset.

    Returns:
        str: A string containing the few-shot examples for contradictory answers.
    """
    # Load the few-shot prompt data
    few_shot_prompt_for_contradictory_answers = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt_for_contradictory_answers = few_shot_prompt_for_contradictory_answers[
        few_shot_prompt_for_contradictory_answers['Contradictory_Answer'].str.len() > 4
    ]

    # Initialize the few-shot examples string
    few_shot_examples_for_contradictory_answers = ""

    for row in range(len(few_shot_prompt_for_contradictory_answers)):
        few_shot_examples_for_contradictory_answers += f"Example {row + 1}:\n"
        few_shot_examples_for_contradictory_answers += f"Document: {few_shot_prompt_for_contradictory_answers.iloc[row]['Document']}\n"
        
        if for_fever_dataset:
            few_shot_examples_for_contradictory_answers += f"Statement: {few_shot_prompt_for_contradictory_answers.iloc[row]['Query']}\n"
            few_shot_examples_for_contradictory_answers += f"Incorrect Answer: {few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer']}\n\n"
        elif for_wow_dataset:
            few_shot_examples_for_contradictory_answers += f"Dialogue: {few_shot_prompt_for_contradictory_answers.iloc[row]['Query']}\n"
            few_shot_examples_for_contradictory_answers += f"Incorrect Response: {few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer']}\n\n"
        else:
            few_shot_examples_for_contradictory_answers += f"Question: {few_shot_prompt_for_contradictory_answers.iloc[row]['Query']}\n"
            few_shot_examples_for_contradictory_answers += f"Incorrect Answer: {few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer']}\n\n"

    return few_shot_examples_for_contradictory_answers

def generate_few_shot_prompts(few_shot_prompt_filename: str, for_fever_dataset: bool, for_wow_dataset: bool) -> tuple[str, int]:
    """
    Generates few-shot prompts for answer generation based on the provided dataset.

    Args:
        few_shot_prompt_filename (str): The filename of the TSV file containing the few-shot prompts.
        for_fever_dataset (bool): Flag indicating if the prompts are for the FEVER dataset.
        for_wow_dataset (bool): Flag indicating if the prompts are for the WoW dataset.

    Returns:
        tuple: A tuple containing the few-shot examples string and the length of the few-shot prompt.
    """
    # Load the few-shot prompt data
    answer_gen_few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    
    # Filter the prompts based on relevance and faithfulness labels
    answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[
        (answer_gen_few_shot_prompt['Answer_Relevance_Label'] == "[[Yes]]") & 
        (answer_gen_few_shot_prompt['Answer_Faithfulness_Label'] == "[[Yes]]")
    ]
    
    # Get the length of the few-shot prompt
    length_of_fewshot_prompt_answer_gen = len(answer_gen_few_shot_prompt)
    
    # Rename 'Query' column to 'Question' if it exists
    if "Query" in answer_gen_few_shot_prompt.columns:
        answer_gen_few_shot_prompt['Question'] = answer_gen_few_shot_prompt['Query']
    
    # Initialize the few-shot examples string
    answer_gen_few_shot_examples = ""
    
    # Construct the few-shot examples
    for row in range(len(answer_gen_few_shot_prompt)):
        answer_gen_few_shot_examples += f"Example {row + 1}:\n"
        answer_gen_few_shot_examples += f"Document: {answer_gen_few_shot_prompt.iloc[row]['Document']}\n"
        
        if for_fever_dataset:
            answer_gen_few_shot_examples += f"Statement: {answer_gen_few_shot_prompt.iloc[row]['Query']}\n"
            answer_gen_few_shot_examples += f"Answer: {answer_gen_few_shot_prompt.iloc[row]['Answer']}\n\n"
        elif for_wow_dataset:
            answer_gen_few_shot_examples += f"Dialogue: {answer_gen_few_shot_prompt.iloc[row]['Query']}\n"
            answer_gen_few_shot_examples += f"Response: {answer_gen_few_shot_prompt.iloc[row]['Answer']}\n\n"
        else:
            answer_gen_few_shot_examples += f"Question: {answer_gen_few_shot_prompt.iloc[row]['Query']}\n"
            answer_gen_few_shot_examples += f"Answer: {answer_gen_few_shot_prompt.iloc[row]['Answer']}\n\n"
    
    return answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen

def generate_query(document: str, settings: dict) -> list:
    """
    Generates synthetic queries for a given document.

    Args:
        document (str): The document text.
        settings (dict): Dictionary containing various settings and parameters required for generating synthetic queries.

    Returns:
        list: List of generated synthetic queries.
    """

    if settings['api_model']:
        return generate_synthetic_query_api_approach(
        document, 
        settings["synthetic_query_prompt"], 
        settings['few_shot_examples'], 
        settings['length_of_fewshot_prompt'], 
        settings['model'], 
        settings['percentiles'], 
        settings['for_fever_dataset'], 
        settings['for_wow_dataset'])
    else: 
        return generate_synthetic_query_llm_approach(
            document, 
            settings['few_shot_examples'], 
            settings['length_of_fewshot_prompt'], 
            settings['device'], 
            settings['tokenizer'], 
            settings['model'], 
            settings['percentiles'], 
            settings['for_fever_dataset'], 
            settings['for_wow_dataset']
        )

def process_documents(documents: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    Processes the documents to generate synthetic queries and remove duplicates.

    Args:
        documents (pd.DataFrame): DataFrame containing the documents.
        settings (dict): Dictionary containing various settings and parameters required for generating synthetic queries.

    Returns:
        pd.DataFrame: DataFrame containing the documents with the generated synthetic queries.
    """
    if settings['api_model']:
        tqdm.pandas(desc=f"Generating synthetic queries ({settings['model_choice']})...", total=documents.shape[0])
        # If the API model is used, we query API model.
        documents["synthetic_query"] = documents.progress_apply(
            lambda x: generate_query(x["document"], settings), axis=1
        )
    else:
        tqdm.pandas(desc="Generating synthetic queries (FLAN)...", total=documents.shape[0])
        # If the API model is not used, we query the local model.
        documents["synthetic_query"] = documents.progress_apply(
            lambda x: generate_query(x["document"], settings), axis=1
        )
    
    documents = documents.explode("synthetic_query", ignore_index=True)
    documents = documents.drop_duplicates(subset=['synthetic_query'])
    return documents

def save_synthetic_queries(documents: pd.DataFrame, filename: str) -> None:
    """
    Saves the generated synthetic queries to a TSV file.

    Args:
        documents (pd.DataFrame): DataFrame containing the documents with the generated synthetic queries.
        filename (str): Filename to save the generated synthetic queries.
    """
    documents.to_csv(filename, index=False, sep="\t")
    print("Saved synthetic queries to: " + filename)

def generate_synthetic_queries(documents: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    Generate synthetic queries using the FLAN approach.

    Args:
        documents (pd.DataFrame): DataFrame containing the documents for which synthetic queries are to be generated.
        settings (dict): Dictionary containing various settings and parameters required for generating synthetic queries.

    Returns:
        pd.DataFrame: DataFrame containing the documents with the generated synthetic queries.
    """
    print("Beginning synthetic query generation!")
    documents = process_documents(documents, settings)
    save_synthetic_queries(documents, settings['synthetic_queries_filename'])
    return documents

def generate_answers(synthetic_queries: pd.DataFrame, answer_generation_settings: dict) -> pd.DataFrame:
    """
    Generate synthetic answers using the FLAN approach.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries.
        answer_generation_settings (dict): Dictionary containing settings and parameters for answer generation.

    Returns:
        pd.DataFrame: DataFrame containing the synthetic queries with generated answers.
    """
    if answer_generation_settings['api_model']:
        tqdm.pandas(desc=f"Generating answers... ({answer_generation_settings['model']})", total=synthetic_queries.shape[0])
        synthetic_queries["generated_answer"] = synthetic_queries.progress_apply(
            lambda x: generate_synthetic_answer_api_approach(
                x["document"], 
                x["synthetic_query"], 
                answer_generation_settings['synthetic_valid_answer_prompt'], 
                answer_generation_settings['answer_gen_few_shot_examples'], 
                answer_generation_settings['length_of_fewshot_prompt_answer_gen'], 
                answer_generation_settings['model'],  
                answer_generation_settings['for_fever_dataset'], 
                answer_generation_settings['for_wow_dataset']
            ), 
            axis=1
        )
    else: 
        tqdm.pandas(desc="Generating answers... (FLAN)", total=synthetic_queries.shape[0])
        synthetic_queries["generated_answer"] = synthetic_queries.progress_apply(
            lambda x: generate_answer_llm_approach(
            x["document"], 
            x["synthetic_query"], 
            answer_generation_settings['answer_gen_few_shot_examples'], 
            answer_generation_settings['length_of_fewshot_prompt_answer_gen'], 
            answer_generation_settings['device'], 
            answer_generation_settings['tokenizer'], 
            answer_generation_settings['model'], 
            answer_generation_settings['for_fever_dataset'], 
            answer_generation_settings['for_wow_dataset']
        ), 
        axis=1
    )
    return synthetic_queries

def label_answers(synthetic_queries: pd.DataFrame) -> pd.DataFrame:
    """
    Label the generated answers for faithfulness and relevance.

    This function takes a DataFrame containing synthetic queries and their generated answers,
    and labels each answer for faithfulness and relevance. The labels are added as new columns
    in the DataFrame.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries and their generated answers.

    Returns:
        pd.DataFrame: DataFrame with additional columns for answer faithfulness and relevance labels.
    """
    # Label each generated answer for faithfulness
    synthetic_queries["Answer_Faithfulness_Label"] = [
        check_generated_answer(synthetic_queries.iloc[i]['generated_answer']) for i in range(len(synthetic_queries))
    ]
    
    # Label each generated answer for relevance
    synthetic_queries["Answer_Relevance_Label"] = [
        check_generated_answer(synthetic_queries.iloc[i]['generated_answer']) for i in range(len(synthetic_queries))
    ]
    
    return synthetic_queries

def generate_contradictory_answers_wrapper(synthetic_queries: pd.DataFrame, answer_generation_settings: dict) -> pd.DataFrame:
    """
    Generate contradictory answers using the specified approach.

    This function generates contradictory answers for the given synthetic queries based on the provided settings.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries.
        answer_generation_settings (dict): Dictionary containing settings for answer generation, including:
            - 'number_of_contradictory_answers_added_ratio' (float): Ratio to determine the number of contradictory answers to add.
            - 'few_shot_examples_for_contradictory_answers' (list): Few-shot examples for generating contradictory answers (if applicable).
            - 'device' (str): Device to use for model inference.
            - 'tokenizer' (transformers.PreTrainedTokenizer): Tokenizer for the model.
            - 'model' (transformers.PreTrainedModel): Model to use for generating answers.
            - 'for_fever_dataset' (bool): Flag indicating if the dataset is for FEVER.
            - 'for_wow_dataset' (bool): Flag indicating if the dataset is for WoW.

    Returns:
        pd.DataFrame: DataFrame with added contradictory answers.
    """

    synthetic_contradictory_answers = generate_contradictory_answer_examples(
        synthetic_queries, 
        int(len(synthetic_queries) * answer_generation_settings['number_of_contradictory_answers_added_ratio']), 
        few_shot_examples_for_contradictory_answers=answer_generation_settings['few_shot_examples_for_contradictory_answers'], 
        api_model=answer_generation_settings['api_model'],
        synthetic_contradictory_answer_prompt=answer_generation_settings['synthetic_contradictory_answer_prompt'],
        device=answer_generation_settings['device'], 
        tokenizer=answer_generation_settings['tokenizer'], 
        model=answer_generation_settings['model'], 
        for_fever_dataset=answer_generation_settings['for_fever_dataset'], 
        for_wow_dataset=answer_generation_settings['for_wow_dataset']
    )
    return synthetic_contradictory_answers

def process_embeddings(synthetic_queries: pd.DataFrame, answer_generation_settings: dict) -> pd.DataFrame:
    """
    Handle embedding generation and additional negatives/positives.

    This function processes the embeddings for the synthetic queries based on the provided settings.
    It generates an index, filters the synthetic queries, and adds additional negatives and positives.

    Args:
        synthetic_queries (pd.DataFrame): DataFrame containing the synthetic queries.
        answer_generation_settings (dict): Dictionary containing settings for answer generation, including:
            - 'regenerate_embeddings' (bool): Flag to determine if embeddings should be regenerated.
            - 'number_of_negatives_added_ratio' (float): Ratio to determine the number of additional negatives to add.
            - 'lower_bound_for_negatives' (int): Lower bound for the number of negatives.
            - 'number_of_positives_added_ratio' (float): Ratio to determine the number of additional positives to add.

    Returns:
        pd.DataFrame: DataFrame with processed embeddings and additional negatives/positives.
    """
    if answer_generation_settings['regenerate_embeddings']:
        print("Generating index and negatives!")
        documentation_index = generate_index(synthetic_queries)
        synthetic_queries = filter_synthetic_queries(synthetic_queries, documentation_index)
        synthetic_queries = generate_additional_negatives(
            synthetic_queries, 
            documentation_index, 
            answer_generation_settings['number_of_negatives_added_ratio'], 
            answer_generation_settings['lower_bound_for_negatives']
        )
        synthetic_queries = generate_additional_positives(
            synthetic_queries, 
            documentation_index, 
            answer_generation_settings['number_of_positives_added_ratio']
        )
    return synthetic_queries

def shuffle_and_save(synthetic_queries: pd.DataFrame, synthetic_queries_filename: str) -> None:
    """
    Shuffle and save the synthetic queries to a specified file.

    This function shuffles the rows of the synthetic queries DataFrame and saves the result to a file in TSV format.

    Args:
        synthetic_queries (pd.DataFrame): The DataFrame containing synthetic queries to be shuffled and saved.
        synthetic_queries_filename (str): The filename where the shuffled synthetic queries will be saved.

    Returns:
        None
    """
    # Shuffle the synthetic queries DataFrame with a fixed random state for reproducibility
    synthetic_queries = synthetic_queries.sample(n=len(synthetic_queries), random_state=42)
    
    # Save the shuffled DataFrame to a TSV file without the index
    synthetic_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
    
    # Print completion messages
    print("Completed synthetic generation!")
    print(f"Saved synthetic queries file to: {synthetic_queries_filename}")

def Generate_Synthetic_Answers(synthetic_queries_filename: str, answer_generation_settings: dict) -> None:
    """
    Main function to generate and save synthetic answers.

    This function reads synthetic queries from a file, processes them to generate answers,
    labels, and contradictory answers, and then saves the results back to the file. It also
    processes embeddings and shuffles the synthetic queries before saving.

    Args:
        synthetic_queries_filename (str): The filename where the synthetic queries are stored.
        answer_generation_settings (dict): Dictionary containing settings for answer generation, including:
            - 'regenerate_answers' (bool): Flag to determine if answers should be regenerated.
            - 'regenerate_embeddings' (bool): Flag to determine if embeddings should be regenerated.
            - 'number_of_negatives_added_ratio' (float): Ratio to determine the number of additional negatives to add.
            - 'lower_bound_for_negatives' (int): Lower bound for the number of negatives.
            - 'number_of_positives_added_ratio' (float): Ratio to determine the number of additional positives to add.

    Returns:
        None
    """
    # Read the synthetic queries from the specified file
    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    # Filter out queries with a length of 10 or less
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]

    # Check if answers need to be regenerated
    if answer_generation_settings['regenerate_answers']:
        print("Beginning answer generation!")
        # Generate answers for the synthetic queries
        synth_queries = generate_answers(synth_queries, answer_generation_settings)
        # Label the generated answers
        synth_queries = label_answers(synth_queries)
        print("Generating contradictory answers!")
        # Generate contradictory answers
        synth_queries = generate_contradictory_answers_wrapper(synth_queries, answer_generation_settings)
        # Save the synthetic queries with answers back to the file
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print(f"Saved answers to: {synthetic_queries_filename}")

    # Re-read the synthetic queries from the file
    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    # Filter out queries with a length of 10 or less
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]

    # Process embeddings for the synthetic queries
    synth_queries = process_embeddings(synth_queries, answer_generation_settings)
    # Shuffle and save the synthetic queries
    shuffle_and_save(synth_queries, synthetic_queries_filename)