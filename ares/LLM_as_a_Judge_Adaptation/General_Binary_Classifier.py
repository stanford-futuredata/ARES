import os
import re
import ast
import json
import time
import random
import argparse
import statistics
import warnings
import subprocess as sp
import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from tqdm import tqdm, tqdm_pandas
from tqdm.auto import tqdm

from transformers import (
    PreTrainedTokenizer, T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration,
    BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer,
    TrainingArguments, Trainer, get_scheduler,
    AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
)
import datasets
from datasets import load_metric

warnings.filterwarnings(
    "ignore", 
    message=(
        "The sentencepiece tokenizer that you are converting to a fast tokenizer "
        "uses the byte fallback option which is not implemented in the fast tokenizers."
    )
)
def combine_query_document(query: str, document: str, answer: str = None) -> str:
    """
    Combines a query and a document into a single string, optionally including an answer.

    Parameters:
    query (str): The query string.
    document (str): The document string.
    answer (str, optional): The answer string. Defaults to None.

    Returns:
    str: A combined string of the query, cleaned document, and optionally the answer.
    """
    # Clean the document by removing extra newlines, carriage returns, and tabs
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r", " ").replace("\t", " ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = " ".join(cleaned_document.split(" ")[:512])

    # Truncate the query if it is too long
    if len(query.split(" ")) > 100:
        query = " ".join(query.split(" ")[:30])

    # Combine query and cleaned document, optionally including the answer
    if answer is None:
        return query + " | " + cleaned_document
    else:
        try:
            return query + " | " + cleaned_document + " | " + answer
        except Exception as e:
            breakpoint()
            print("Error with combine_query_document")
            print("Query: " + str(query))
            print("Cleaned Document: " + str(cleaned_document))
            print("Answer: " + str(answer))
            return str(query) + " | " + str(cleaned_document) + " | " + str(answer)

def format_text_for_fine_tuning_content_relevance_sequence_classification(question: str, document: str) -> str:
    """
    Formats text for fine-tuning content relevance sequence classification.

    Parameters:
    question (str): The question string.
    document (str): The document string.

    Returns:
    str: A formatted string containing the instruction, question, and cleaned document.
    """
    instruction = (
        "You are an expert judge for evaluating question answering systems. "
        "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. \n\n"
    )

    # Clean the document by removing extra newlines, carriage returns, and tabs
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r", " ").replace("\t", " ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = " ".join(cleaned_document.split(" ")[:512])

    # Construct the instruction string
    instruction += "### Instruction:\n"
    instruction += f"Question: {question}\n"
    instruction += f"Document: {cleaned_document}\n"
    instruction += "### Response:\n"

    return instruction

def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_gpu_memory() -> list:
    """
    Retrieves the free GPU memory for each GPU available on the system.

    This function uses the `nvidia-smi` command to query the free memory of each GPU and returns a list of free memory values.

    Returns:
    list: A list of integers representing the free memory (in MiB) for each GPU.
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    return memory_free_values

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels: int, model_choice: str):
        """
        Initializes the CustomBERTModel with the specified number of labels and model choice.

        Args:
            number_of_labels (int): The number of labels for the classification task.
            model_choice (str): The choice of the pre-trained model to use.
        """
        self.model_choice = model_choice
        super(CustomBERTModel, self).__init__()

        if model_choice in ["mosaicml/mpt-7b-instruct", "mosaicml/mpt-7b"]:
            config = AutoConfig.from_pretrained(model_choice, trust_remote_code=True)
            config.attn_config['attn_impl'] = 'triton'  # Use triton-based FlashAttention
            config.max_seq_len = max_token_length

            model_encoding = AutoModelForCausalLM.from_pretrained(
                model_choice,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_auth_token=True
            )
            embedding_size = 4096
            self.encoderModel = model_encoding.transformer

        elif model_choice in ['mosaicml/mpt-1b-redpajama-200b']:
            model_encoding = MptForSequenceClassification.from_pretrained(
                "mosaicml/mpt-1b-redpajama-200b", 
                trust_remote_code=True
            )
            embedding_size = 2048
            self.encoderModel = model_encoding

        elif model_choice in ["google/t5-large-lm-adapt", "google/t5-xl-lm-adapt"]:
            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

        elif model_choice in ["roberta-large", "microsoft/deberta-v3-large"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

        elif model_choice in ["microsoft/deberta-v2-xlarge", "microsoft/deberta-v2-xxlarge"]:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1536
            self.encoderModel = model_encoding

        else:
            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.Linear(256, number_of_labels)
        )
        self.embedding_size = embedding_size

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor = None, 
    decoder_input_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Parameters:
        ids (torch.Tensor): Input IDs tensor.
        mask (torch.Tensor): Attention mask tensor.
        labels (torch.Tensor, optional): Labels tensor. Defaults to None.
        decoder_input_ids (torch.Tensor, optional): Decoder input IDs tensor. Defaults to None.

        Returns:
        torch.Tensor: The output logits or classifier output.
        """
        # Retrieve the model choice
        model_choice = self.model_choice

        # Check if the model choice is one of the specified models
        if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt", "mosaicml/mpt-1b-redpajama-200b"]:
            # Perform a forward pass for the specified models
            total_output = self.encoderModel(input_ids=ids, attention_mask=mask)
            return total_output['logits']
        else:
            # Perform a forward pass for other models
            total_output = self.encoderModel(ids, attention_mask=mask)
            sequence_output = total_output['last_hidden_state']

            # Format the last hidden state and pass it through the classifier
            last_hidden_state_formatted = sequence_output[:, 0, :].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)

            return linear2_output

def tokenize_function(tokenizer: AutoTokenizer, examples: dict) -> dict:
    """
    Tokenizes the input examples using the provided tokenizer.

    Parameters:
    tokenizer (AutoTokenizer): The tokenizer to use for tokenizing the examples.
    examples (dict): A dictionary containing the text to be tokenized. 
                     It should have a key "text" with the corresponding text data.

    Returns:
    dict: A dictionary containing the tokenized input with padding and truncation applied.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def checkpoints(classification_datasets: list, model_choice: str) -> None:
    """
    Creates necessary checkpoint directories for the given classification datasets and model choice.

    Parameters:
    classification_datasets (list): A list of dataset names for which checkpoints need to be created.
    model_choice (str): The model choice string used to determine the folder structure.

    Returns:
    None
    """
    checkpoints_folder_path = "checkpoints/"
    
    # Create the main checkpoints directory if it doesn't exist
    if not os.path.isdir(checkpoints_folder_path):
        os.mkdir(checkpoints_folder_path)

    # Create a directory for the specific model choice
    dataset_folder_path = os.path.join(checkpoints_folder_path, model_choice.replace("/", "-"))
    if not os.path.isdir(dataset_folder_path):
        print(f"Creating folder: {dataset_folder_path}")
        os.mkdir(dataset_folder_path)

    # Create directories for each dataset within the model choice directory
    for dataset in classification_datasets:
        dataset_path = os.path.join(dataset_folder_path, dataset.replace("../", "").replace("/", "-"))
        try:
            os.mkdir(dataset_path)
        except FileExistsError:
            print("Already exists")
            print(dataset_path)

def load_model(model_choice: str) -> tuple[AutoTokenizer, int]:
    """
    Loads the tokenizer for the specified model choice and sets the maximum token length.

    Parameters:
    model_choice (str): The model identifier to load the tokenizer from.

    Returns:
    tuple: A tuple containing the tokenizer and the maximum token length.
           - tokenizer (AutoTokenizer): The tokenizer loaded from the specified model.
           - max_token_length (int): The maximum token length set for the tokenizer.
    """
    max_token_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length)
    
    return tokenizer, max_token_length

def prepare_and_clean_data(params: dict) -> tuple[str, int]:
    """
    Prepares and cleans data based on the provided parameters.

    Parameters:
    params (dict): A dictionary containing the following keys:
        - "training_dataset_path" (str): Path to the training dataset.
        - "learning_rate_choices" (list): List of possible learning rates.
        - "chosen_learning_rate" (float): The chosen learning rate for the model.
        - "model_choice" (str): The model identifier.
        - "number_of_runs" (int): Number of runs for the training.
        - "validation_set_scoring" (str): Scoring method for the validation set.
        - "label" (str): The label column name in the dataset.
        - "validation_dataset_path" (str): Path to the validation dataset.
        - "patience_value" (int): Patience value for early stopping.
        - "num_epochs" (int): Number of epochs for training.
        - "num_warmup_steps" (int): Number of warmup steps for the learning rate scheduler.
        - "gradient_accumulation_multiplier" (int): Multiplier for gradient accumulation.
        - "assigned_batch_size" (int): Batch size assigned for training.
        - "tokenizer" (AutoTokenizer): Tokenizer to be used.

    Returns:
    tuple: A tuple containing:
        - checkpoint_path (str): Path to the checkpoint file.
        - patience_value (int): The patience value for early stopping.
    """
    # Extract parameters from the dictionary
    dataset = params["training_dataset_path"]
    learning_rate_choices = params["learning_rate_choices"]
    chosen_learning_rate = params["chosen_learning_rate"]
    model_choice = params["model_choice"]
    number_of_runs = params["number_of_runs"]
    validation_set_scoring = params["validation_set_scoring"]
    label_column = params["label"]
    validation_set = params["validation_dataset_path"]
    patience_value = params["patience_value"]
    num_epochs = params["num_epochs"]
    num_warmup_steps = params["num_warmup_steps"]
    gradient_accumulation_multiplier = params["gradient_accumulation_multiplier"]
    assigned_batch_size = params["assigned_batch_size"]
    tokenizer = params["tokenizer"]

    # Log the start of a new learning rate
    print("--------------------------------------------------------------------------")
    print("Starting new learning rate: " + str(chosen_learning_rate))
    print("--------------------------------------------------------------------------")

    # Generate current datetime for checkpoint naming
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Define parent directory for checkpoints
    parent_dir = "checkpoints/" + model_choice.replace("/", "-")

    # Create parent directory if it doesn't exist
    if not os.path.exists(parent_dir):
        print(f"Creating parent checkpoint directory: {parent_dir}")
        print("--------------------------------------------------------------------------")
        os.makedirs(parent_dir)

    # Define the checkpoint path
    checkpoint_path = os.path.join(
        "checkpoints",
        model_choice.replace("/", "-"),
        f"{label_column}_{os.path.basename(validation_set).replace('.tsv', '')}_{current_datetime}.pt"
    )

    # Record the start time of execution
    execution_start = time.time()

    # Log various parameters
    print("Dataset: " + dataset)
    print("Model: " + model_choice)
    print("Test Set Selection: " + validation_set)
    print("Number of Runs: " + str(number_of_runs))
    print('Learning Rate: ' + str(chosen_learning_rate))
    print("Checkpoint Path: " + checkpoint_path)
    print("Patience: " + str(patience_value))
    print("Validation Set Choice: " + str(validation_set_scoring))
    print("Number of Epochs: " + str(num_epochs))
    print("Number of warmup steps: " + str(num_warmup_steps))
    print("--------------------------------------------------------------------------")

    return checkpoint_path, patience_value

def analyze_and_report_data(dataset: str, label_column: str, tokenizer: AutoTokenizer, max_token_length: int) -> pd.DataFrame:
    """
    Analyzes and reports data from a given dataset.

    Parameters:
    - dataset (str): Path to the dataset file.
    - label_column (str): The column name in the dataset that contains the labels.
    - tokenizer: Tokenizer object used for encoding text.
    - max_token_length (int): Maximum token length allowed for the text.

    Returns:
    - pd.DataFrame: A DataFrame containing the processed and filtered data.
    """
    # Read the dataset
    synth_queries = pd.read_csv(dataset, sep="\t")

    # If the dataset is reformatted, rename columns accordingly
    if "nq_reformatted" in dataset:
        synth_queries['synthetic_query'] = synth_queries['Query']
        synth_queries['generated_answer'] = synth_queries['Answer']
        synth_queries['document'] = synth_queries['Document']

    # Print initial count
    print(f"Initial count: {len(synth_queries)}")

    # Filter out rows with NaN values in the specified columns
    synth_queries = synth_queries[synth_queries[label_column] != "NaN"]
    synth_queries = synth_queries[synth_queries["synthetic_query"].notna()]
    synth_queries = synth_queries[synth_queries["document"].notna()]
    synth_queries = synth_queries[synth_queries['generated_answer'].notna()]
    synth_queries = synth_queries[synth_queries[label_column].notna()]

    # Print count after initial filtering
    print(f"Count after initial filtering: {len(synth_queries)}")

    # Shuffle the dataset
    synth_queries = synth_queries.sample(n=len(synth_queries), random_state=42)

    # Print counts of Answer_Relevance_Label before any further filtering
    print(f"Answer_Relevance_Label counts before filtering: Yes - {synth_queries[synth_queries['Answer_Relevance_Label'] == 'Yes'].shape[0]}, No - {synth_queries[synth_queries['Answer_Relevance_Label'] == 'No'].shape[0]}")

    # Combine query and document (and generated answer if applicable) into a single text field
    if "Context" in label_column:
        synth_queries["concat_text"] = [
            combine_query_document(synth_queries.iloc[i]['synthetic_query'], synth_queries.iloc[i]['document'])
            for i in range(len(synth_queries))
        ]

        # Print the count before filtering duplicates
        print(f"Count before filtering duplicates for context relevance: {len(synth_queries)}")

        # Temporarily remove rows with duplicate query/document pairs for context relevance
        synth_queries = synth_queries.drop_duplicates(subset=["synthetic_query", "document"])

        # Print the count after filtering
        print(f"Count after filtering duplicates for context relevance: {len(synth_queries)}")

    else:
        synth_queries["concat_text"] = [
            combine_query_document(synth_queries.iloc[i]['synthetic_query'], synth_queries.iloc[i]['document'], synth_queries.iloc[i]['generated_answer'])
            for i in range(len(synth_queries))
        ]

        # Print the count before filtering
        print(f"Count before filtering for context relevance: {len(synth_queries)}")

        # Print counts of Context_Relevance_Label before filtering
        print(f"Context_Relevance_Label counts before filtering: Yes - {synth_queries[synth_queries['Context_Relevance_Label'] == 'Yes'].shape[0]}, No - {synth_queries[synth_queries['Context_Relevance_Label'] == 'No'].shape[0]}")

        # Temporarily remove rows where context relevance is "No" for answer relevance/faithfulness
        synth_queries = synth_queries[synth_queries["Context_Relevance_Label"] != "No"]

        # Print the count after filtering
        print(f"Count after filtering for context relevance: {len(synth_queries)}")

    # Print counts of Answer_Relevance_Label after filtering for context relevance
    print(f"Answer_Relevance_Label counts after filtering: Yes - {synth_queries[synth_queries['Answer_Relevance_Label'] == 'Yes'].shape[0]}, No - {synth_queries[synth_queries['Answer_Relevance_Label'] == 'No'].shape[0]}")

    # Tokenize the concatenated text and calculate token lengths
    synth_queries['token_length'] = [
        len(tokenizer.encode(text, return_tensors='pt')[0])
        for text in tqdm(synth_queries['concat_text'], desc="Tokenizing")
    ]

    # Print count before token length filtering
    print(f"Count before token length filtering: {len(synth_queries)}")

    # Remove duplicate rows based on the concatenated text
    synth_queries = synth_queries.drop_duplicates(["concat_text"])

    # Filter out rows where token length exceeds the maximum allowed token length
    synth_queries = synth_queries[synth_queries['token_length'] <= 4096]

    # Print final count
    print(f"Final count after token length filtering: {len(synth_queries)}")

    return synth_queries

def transform_data(synth_queries: pd.DataFrame, validation_set: str, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transforms the synthetic queries and validation set for training and testing.

    Parameters:
    - synth_queries (pd.DataFrame): DataFrame containing synthetic queries.
    - validation_set (str): Path to the validation set file.
    - label_column (str): The column name used for labeling.

    Returns:
    - tuple: A tuple containing the transformed training and test DataFrames.
    """
    # Initialize the training DataFrame
    train_df = synth_queries

    # Read and preprocess the validation set
    test_set = pd.read_csv(validation_set, sep="\t")
    test_set['Question'] = test_set['Query']
    test_set['Document'] = test_set['Document'].str.strip()
    test_set = test_set[test_set["Document"].str.len() > 100]
    test_set = test_set[test_set[label_column].notna()]

    # Preprocess the training DataFrame
    train_df['document'] = train_df['document'].astype(str).str.strip()
    train_df = train_df[train_df["document"].str.len() > 100]
    train_df = train_df[train_df[label_column].notna()]

    # Print counts of Answer_Relevance_Label before any further filtering
    print(f"Answer_Relevance_Label counts before filtering: Yes - {train_df[train_df['Answer_Relevance_Label'] == 'Yes'].shape[0]}, No - {train_df[train_df['Answer_Relevance_Label'] == 'No'].shape[0]}")

    # Combine query and document (and generated answer if applicable) into a single text field
    if "Context" in label_column:
        test_set['concat_text'] = [
            combine_query_document(test_set.iloc[i]['Question'], test_set.iloc[i]['Document'])
            for i in range(len(test_set))
        ]

        # Print the count before filtering duplicates
        print(f"Count before filtering duplicates for context relevance: {len(train_df)}")

        # Temporarily remove rows with duplicate query/document pairs for context relevance
        train_df = train_df.drop_duplicates(subset=["synthetic_query", "document"])

        # Print the count after filtering
        print(f"Count after filtering duplicates for context relevance: {len(train_df)}")

    else:
        test_set['concat_text'] = [
            combine_query_document(test_set.iloc[i]['Question'], test_set.iloc[i]['Document'], test_set.iloc[i]['Answer'])
            for i in range(len(test_set))
        ]

        # Print the count before filtering
        print(f"Count before filtering for context relevance: {len(train_df)}")

        # Temporarily remove rows where context relevance is 0 for answer relevance/faithfulness
        train_df = train_df[train_df["Context_Relevance_Label"] != "No"]

        # Print the count after filtering
        print(f"Count after filtering for context relevance: {len(train_df)}")

    # Print counts of Answer_Relevance_Label after filtering for context relevance
    print(f"Answer_Relevance_Label counts after filtering: Yes - {train_df[train_df['Answer_Relevance_Label'] == 'Yes'].shape[0]}, No - {train_df[train_df['Answer_Relevance_Label'] == 'No'].shape[0]}")

    # Remove duplicate rows based on the concatenated text
    train_df = train_df.drop_duplicates(["concat_text"])
    test_set = test_set.drop_duplicates(["concat_text"])

    # Additional filtering for Answer_Faithfulness classification
    if "Faith" in label_column:
        print("Refining data for Answer_Faithfulness classification!")
        train_df = train_df[train_df["Context_Relevance_Label"].notna()]
        train_df = train_df[train_df["Answer_Faithfulness_Label"].notna()]
        error_strings = ['answer', 'contrad', 'false', 'information', 'unanswer', 'Answer', 'Contrad', 'False', 'Information', 'Unanswer']
        train_df['generated_answer'] = train_df['generated_answer'].astype(str)
        train_df = train_df[~train_df['generated_answer'].str.contains('|'.join(error_strings))]

    return train_df, test_set
def split_dataset(train_df: pd.DataFrame, dataset: str, 
test_set: pd.DataFrame, label_column: str) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int], list[int]]:
    """
    Splits the dataset into training, development, and test sets, and extracts the corresponding labels.

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - dataset (str): Name of the dataset.
    - test_set (pd.DataFrame): DataFrame containing the test data.
    - label_column (str): The column name used for labeling.

    Returns:
    - tuple: A tuple containing:
        - train_set_text (list): List of concatenated text fields for the training set.
        - train_set_label (list): List of labels for the training set.
        - dev_set_text (list): List of concatenated text fields for the development set.
        - dev_set_label (list): List of labels for the development set.
        - test_set_text (list): List of concatenated text fields for the test set.
        - test_set_label (list): List of labels for the test set.
        - labels_list (list): Sorted list of unique labels.
    """
    
    # Conversion dictionary for binary classification
    conversion_dict = {"Yes": 1, "No": 0}
    
    # Extract concatenated text fields for the training set
    train_set_text = [train_df.iloc[i]['concat_text'] for i in range(len(train_df))]
    
    # Extract labels for the training set
    if "nq_reformatted" not in dataset:
        train_set_label = [conversion_dict[train_df.iloc[i][label_column]] for i in range(len(train_df))]
    else:
        train_set_label = [int(train_df.iloc[i][label_column]) for i in range(len(train_df))]
    
    # Extract concatenated text fields and labels for the development set
    dev_set_text = [test_set.iloc[i]['concat_text'] for i in range(len(test_set))]
    dev_set_label = [int(test_set.iloc[i][label_column]) for i in range(len(test_set))]
    
    # Extract concatenated text fields and labels for the test set
    test_set_text = [test_set.iloc[i]['concat_text'] for i in range(len(test_set))]
    test_set_label = [int(test_set.iloc[i][label_column]) for i in range(len(test_set))]

    # Create a sorted list of unique labels
    labels_list = sorted(list(set(train_set_label + dev_set_label + test_set_label)))

    return train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, test_set_label, labels_list

    ############################################################

def prepare_dataset(validation_set_scoring: bool, 
                    train_set_label: list[int], 
                    train_set_text: list[str], 
                    dev_set_label: list[int], 
                    dev_set_text: list[str], 
                    test_set_label: list[int] = None, 
                    test_set_text: list[str] = None) -> tuple[pd.DataFrame, datasets.Dataset, datasets.Dataset, datasets.Dataset, pd.DataFrame]:
    """
    Prepares the dataset for training, validation, and testing by converting them into pandas DataFrames and Arrow tables.

    Parameters:
    - validation_set_scoring (bool): Flag to determine if validation set scoring is enabled.
    - train_set_label (list[int]): List of labels for the training set.
    - train_set_text (list[str]): List of text data for the training set.
    - dev_set_label (list[int]): List of labels for the development set.
    - dev_set_text (list[str]): List of text data for the development set.
    - test_set_label (list[int], optional): List of labels for the test set. Required if validation_set_scoring is False.
    - test_set_text (list[str], optional): List of text data for the test set. Required if validation_set_scoring is False.

    Returns:
    - tuple: A tuple containing:
        - training_dataset_pandas (pd.DataFrame): DataFrame for the training set.
        - training_dataset_arrow (datasets.Dataset): Arrow table for the training set.
        - validation_dataset_arrow (datasets.Dataset): Arrow table for the validation set.
        - test_dataset_arrow (datasets.Dataset): Arrow table for the test set.
        - test_dataset_pandas (pd.DataFrame): DataFrame for the test set.
    """
    
    if validation_set_scoring:
        # Prepare training dataset
        training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})
        training_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(training_dataset_pandas))

        # Prepare validation dataset
        validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
        validation_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(validation_dataset_pandas))

        # Prepare test dataset
        test_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
        test_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(test_dataset_pandas))
    else:
        # Prepare training dataset
        training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})
        training_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(training_dataset_pandas))

        # Prepare validation dataset
        validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
        validation_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(validation_dataset_pandas))

        # Prepare test dataset
        test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
        test_dataset_arrow = datasets.Dataset(pa.Table.from_pandas(test_dataset_pandas))

    return training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas

def initalize_dataset_for_tokenization(tokenizer: PreTrainedTokenizer, 
                                      training_dataset_arrow: datasets.Dataset, 
                                      validation_dataset_arrow: datasets.Dataset, 
                                      test_dataset_arrow: datasets.Dataset) -> datasets.DatasetDict:
    """
    Initializes and tokenizes the dataset for training, validation, and testing.

    Parameters:
    - tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the text data.
    - training_dataset_arrow (datasets.Dataset): Arrow table for the training set.
    - validation_dataset_arrow (datasets.Dataset): Arrow table for the validation set.
    - test_dataset_arrow (datasets.Dataset): Arrow table for the test set.

    Returns:
    - datasets.DatasetDict: A dictionary containing the tokenized datasets for training, validation, and testing.
    """
    
    # Create a DatasetDict with the provided datasets
    classification_dataset = datasets.DatasetDict({
        'train': training_dataset_arrow, 
        'validation': validation_dataset_arrow, 
        'test': test_dataset_arrow
    })
    
    # Tokenize the datasets
    tokenized_datasets = classification_dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)

    # Remove the 'text' column as it is no longer needed after tokenization
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
    # Rename the 'label' column to 'labels' to match the expected input format for the model
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Set the format of the datasets to PyTorch tensors
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

    ############################################################

def train_and_evaluate_model(params: dict) -> tuple[list[torch.nn.Module], list[float], list[float], list[float]]:
    """
    Trains and evaluates a model based on the provided parameters.

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - number_of_runs (int): Number of training runs.
        - tokenized_datasets (datasets.DatasetDict): Tokenized datasets for training, validation, and testing.
        - assigned_batch_size (int): Batch size for training and evaluation.
        - train_set_label (list): Labels for the training set.
        - model_choice (str): Model choice identifier.
        - chosen_learning_rate (float): Learning rate for the optimizer.
        - device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        - checkpoint_path (str): Path to save the model checkpoints.
        - patience_value (int): Patience value for early stopping.
        - num_epochs (int): Number of epochs for training.
        - num_warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        - gradient_accumulation_multiplier (int): Gradient accumulation steps.

    Returns:
    - tuple: A tuple containing the trained model, average training losses, average validation losses, evaluation dataloader, and inference times.
    """
    
    # Extract parameters from the dictionary
    number_of_runs = params["number_of_runs"]
    tokenized_datasets = params["tokenized_datasets"]
    assigned_batch_size = params["assigned_batch_size"]
    train_set_label = params["train_set_label"]
    model_choice = params["model_choice"]
    chosen_learning_rate = params["chosen_learning_rate"]
    device = params["device"]
    checkpoint_path = params["checkpoint_path"]
    patience_value = params["patience_value"]
    num_epochs = params["num_epochs"]
    num_warmup_steps = params["num_warmup_steps"]
    gradient_accumulation_multiplier = params["gradient_accumulation_multiplier"]

    # Initialize lists to store metrics
    micro_averages = []
    macro_averages = []
    inference_times = []

    for i in range(number_of_runs):
        run_start = time.time()
        print("Loading Model")

        # Create dataloaders for training, validation, and evaluation
        train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
        validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
        eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

        # Initialize the model and move it to the specified device
        model = CustomBERTModel(len(set(train_set_label)), model_choice)
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=chosen_learning_rate)

        # Calculate the total number of training steps and initialize the learning rate scheduler
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        # Initialize lists to store losses
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        # Import EarlyStopping utility
        from ares.LLM_as_a_Judge_Adaptation.pytorchtools import EarlyStopping

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience_value, verbose=True, path=checkpoint_path)
        print("Checkpoint Path: " + checkpoint_path)
        print("Beginning Training")

        total_epochs_performed = 0

        for epoch in range(num_epochs):
            total_epochs_performed += 1
            print("Current Epoch: " + str(epoch))

            progress_bar = tqdm(range(len(train_dataloader)))
            gradient_accumulation_count = 0
            model.train()
            for batch in train_dataloader:
                # Prepare the batch for the model
                if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
                else:
                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                # Forward pass
                outputs = model(**new_batch)
                loss = criterion(outputs, batch['labels'].to(device))
                loss.backward()

                # Gradient accumulation
                gradient_accumulation_count += 1
                if gradient_accumulation_count % gradient_accumulation_multiplier == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
                train_losses.append(loss.item())

            progress_bar = tqdm(range(len(validation_dataloader)))
            model.eval()
            for batch in validation_dataloader:
                with torch.no_grad():
                    # Prepare the batch for the model
                    if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
                    else:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)

                    # Forward pass
                    outputs = model(**new_batch)
                    loss = criterion(outputs, batch['labels'].to(device))
                    progress_bar.update(1)
                    valid_losses.append(loss.item())

            # Calculate average losses
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            # Print epoch summary
            epoch_len = len(str(num_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

            # Reset losses for the next epoch
            train_losses = []
            valid_losses = []
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return model, avg_train_losses, avg_valid_losses, eval_dataloader, inference_times

        ############################################################

def evaluate_model(model: torch.nn.Module, model_choice: str, checkpoint_path: str, 
device: torch.device, eval_dataloader: DataLoader, inference_times: list) -> tuple:
    """
    Evaluates the given model on the evaluation dataset.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - model_choice (str): The choice of model architecture.
    - checkpoint_path (str): Path to the model checkpoint to load.
    - device (torch.device): The device to run the evaluation on (e.g., 'cuda:0').
    - eval_dataloader (DataLoader): DataLoader for the evaluation dataset.
    - inference_times (list): List to store inference times.

    Returns:
    - tuple: A tuple containing total predictions, total references, and the evaluation metric.
    """

    print("Loading the Best Model")
    model.load_state_dict(torch.load(checkpoint_path))

    print("Beginning Evaluation")
    metric = load_metric("accuracy")

    total_predictions = torch.FloatTensor([]).to(device)
    total_references = torch.FloatTensor([]).to(device)

    inference_start = time.time()

    progress_bar = tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:
        with torch.no_grad():
            # Prepare the batch for the model
            if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
            else:
                new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

            if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)

            # Forward pass
            outputs = model(**new_batch)

            logits = outputs
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

            total_predictions = torch.cat((total_predictions, predictions), 0)
            total_references = torch.cat((total_references, batch['labels'].to(device)), 0)

            progress_bar.update(1)

    inference_end = time.time()
    total_inference_time = inference_end - inference_start
    inference_times.append(total_inference_time)

    return total_predictions, total_references, metric

        ############################################################

def print_and_save_model(total_predictions: torch.Tensor, total_references: torch.Tensor, checkpoint_path: str, metric) -> None:
    """
    Prints the shapes of the predictions and references, computes and prints the accuracy and F1 scores,
    and saves the classification checkpoint.

    Args:
    - total_predictions (torch.Tensor): Tensor containing the model's predictions.
    - total_references (torch.Tensor): Tensor containing the true labels.
    - checkpoint_path (str): Path to save the classification checkpoint.
    - metric: Metric object used to compute the accuracy.

    Returns:
    - None
    """
    print("--------------------------")
    print("Predictions and Reference Shapes")
    print(total_predictions.shape)
    print(total_references.shape)

    # Compute and print accuracy
    results = metric.compute(references=total_references, predictions=total_predictions)
    print("Accuracy for Test Set: " + str(results['accuracy']))

    # Compute and print F1 scores
    f_1_metric = load_metric("f1", trust_remote_code=True)
    macro_f_1_results = f_1_metric.compute(average='macro', references=total_references, predictions=total_predictions)
    print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
    micro_f_1_results = f_1_metric.compute(average='micro', references=total_references, predictions=total_predictions)
    print("Micro F1 for Test Set: " + str(micro_f_1_results['f1'] * 100))

    # Compute and print positive/negative reference ratio
    positive_ratio = round(total_references.tolist().count(1) / len(total_references.tolist()), 3)
    print("Positive / Negative Reference Ratio: " + str(positive_ratio))

    # Print checkpoint save path
    print("Saved classification checkpoint to: " + str(checkpoint_path))
