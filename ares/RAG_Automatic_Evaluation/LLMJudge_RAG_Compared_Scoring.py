import torch.nn as nn
from transformers import (
    T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration, 
    BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer, 
    TrainingArguments, Trainer, get_scheduler, 
    AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, 
    MptForSequenceClassification
)
import sys
import pandas as pd
import numpy as np
import csv
import ast
import datasets
import evaluate
import pyarrow as pa
import pyarrow.dataset as ds
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import statistics
import time
import subprocess as sp
import os
from sklearn.model_selection import train_test_split
import json
import random
import re
import scipy.stats as stats
import argparse
import openai
from tqdm import tqdm
import subprocess
tqdm.pandas()

# Set random seed for reproducibility
random_state = 44

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)
os.environ["HUGGINGFACE_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ares.RAG_Automatic_Evaluation.ppi import clt_iid, binomial_iid, pp_mean_iid_asymptotic
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import (
    calculate_accuracy, few_shot_context_relevance_scoring, 
    few_shot_answer_faithfulness_scoring, few_shot_answer_relevance_scoring, 
    few_shot_context_relevance_scoring_togetherai, few_shot_answer_faithfulness_scoring_togetherai, 
    few_shot_answer_relevance_scoring_togetherai, few_shot_context_relevance_scoring_claude, 
    few_shot_answer_faithfulness_scoring_claude, few_shot_answer_relevance_scoring_claude, 
    few_shot_context_relevance_scoring_vllm, few_shot_answer_relevance_scoring_vllm, 
    few_shot_answer_faithfulness_scoring_vllm
)

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels: int, model_choice: str):
        """
        Initializes the CustomBERTModel with the specified number of labels and model choice.

        Args:
            number_of_labels (int): The number of labels for the classification task.
            model_choice (str): The model choice for the encoder.
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
                "mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True
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

        self.encoderModel.eval()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.Linear(256, number_of_labels)
        )
        self.embedding_size = embedding_size

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor = None, 
                decoder_input_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters:
        ids (torch.Tensor): Input tensor containing token ids.
        mask (torch.Tensor): Attention mask tensor.
        labels (torch.Tensor, optional): Labels tensor for supervised training. Defaults to None.
        decoder_input_ids (torch.Tensor, optional): Decoder input ids for models that require them. Defaults to None.

        Returns:
        torch.Tensor: The output logits or classifier output depending on the model choice.
        """
        model_choice = self.model_choice

        # For specific models, use the encoder model to get the logits directly
        if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt", "mosaicml/mpt-1b-redpajama-200b"]:
            total_output = self.encoderModel(input_ids=ids, attention_mask=mask)
            return total_output['logits']
        else:
            # For other models, process the output through the classifier
            total_output = self.encoderModel(ids, attention_mask=mask)
            sequence_output = total_output['last_hidden_state']

            # Format the last hidden state and pass it through the classifier
            last_hidden_state_formatted = sequence_output[:, 0, :].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)

            return linear2_output
        
def combine_query_document(query: str, document: str, answer=None):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r", " ").replace("\t", " ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = " ".join(cleaned_document.split(" ")[:512])

    if len(query.split(" ")) > 100:
        query = " ".join(query.split(" ")[:30])

    if answer is None:
        return f"{query} | {cleaned_document}"
    else:
        try:
            return f"{query} | {cleaned_document} | {answer}"
        except Exception as e:
            print(f"Error with combine_query_document: {e}")
            print(f"Query: {query}")
            print(f"Cleaned Document: {cleaned_document}")
            print(f"Answer: {answer}")
            return "Error"

def tokenize_function(tokenizer, examples: dict) -> dict:
    """
    Tokenizes the input examples using the provided tokenizer.

    Parameters:
    tokenizer (Tokenizer): The tokenizer to be used for tokenizing the text.
    examples (dict): A dictionary containing the text to be tokenized. 
                     It should have a key "text" with the text data as its value.

    Returns:
    dict: A dictionary containing the tokenized text with padding and truncation applied.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def prepare_dataset_for_evaluation(dataframe: pd.DataFrame, label_column: str, 
                                   text_column: str, assigned_batch_size: int, tokenizer) -> DataLoader:
    """
    Prepares a dataset for evaluation by tokenizing the text and creating a DataLoader.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the data.
    label_column (str): The name of the column containing the labels.
    text_column (str): The name of the column containing the text data.
    assigned_batch_size (int): The batch size to be used for the DataLoader.
    tokenizer: The tokenizer to be used for tokenizing the text.

    Returns:
    DataLoader: A DataLoader object for the tokenized dataset.
    """
    from datasets.utils.logging import disable_progress_bar
    disable_progress_bar()

    # Extract text and labels from the dataframe
    test_set_text = [dataframe.iloc[i][text_column] for i in range(len(dataframe))]
    
    if label_column in dataframe.columns:
        test_set_label = dataframe[label_column].tolist()
        # Create a pandas DataFrame with the extracted text and labels
        test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    else: 
        # Create a pandas DataFrame with only the text data
        test_dataset_pandas = pd.DataFrame({'text': test_set_text})

    # Convert the pandas DataFrame to an Arrow Table and then to a Hugging Face Dataset
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    # Create a DatasetDict with the test dataset
    classification_dataset = datasets.DatasetDict({'test': test_dataset_arrow})

    # Tokenize the dataset
    tokenized_datasets = classification_dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)

    # Remove the original text column and rename the label column to "labels"
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    if 'label' in tokenized_datasets['test'].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set the format of the dataset to PyTorch tensors
    tokenized_datasets.set_format("torch")

    # Create a DataLoader for the tokenized dataset
    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

    return eval_dataloader

def calculate_ppi(Y_labeled: np.ndarray, Yhat_labeled: np.ndarray, 
                  Yhat_unlabeled: np.ndarray, alpha: float, num_trials: int) -> tuple:
    """
    Calculate prediction-powered inference (PPI) and classical inference intervals.

    Parameters:
    Y_labeled (np.ndarray): Labeled ground truth values.
    Yhat_labeled (np.ndarray): Predictions for the labeled data.
    Yhat_unlabeled (np.ndarray): Predictions for the unlabeled data.
    alpha (float): Significance level for the confidence intervals.
    num_trials (int): Number of trials to run for the inference.

    Returns:
    tuple: A tuple containing the average PPI confidence interval, the average classical confidence interval, and the imputed-only confidence interval.
    """
    
    n_max = Y_labeled.shape[0]
    ns = np.linspace(0, n_max, 20).astype(int)

    # Imputed-only estimate
    imputed_estimate = (Yhat_labeled.sum() + Yhat_unlabeled.sum()) / (Yhat_labeled.shape[0] + Yhat_unlabeled.shape[0])

    # Initialize arrays to store confidence intervals
    ci = np.zeros((num_trials, ns.shape[0], 2))
    ci_classical = np.zeros((num_trials, ns.shape[0], 2))

    # Run prediction-powered inference and classical inference for many values of n
    for j in tqdm(range(num_trials), desc="Trials"):  # Wrap the outer loop with tqdm for the progress bar
        for i, n in enumerate(ns):  # Iterate over ns with an index
            rand_idx = np.random.permutation(Y_labeled.shape[0])
            f = Yhat_labeled.astype(float)[rand_idx[:n]]
            y = Y_labeled.astype(float)[rand_idx[:n]]
            output = pp_mean_iid_asymptotic(y, f, Yhat_unlabeled, alpha)
            ci[j, i, :] = output
            # Classical interval
            try:
                if n == 0:
                    ci_classical[j, i, :] = [0, 0]
                else:
                    ci_classical[j, i, :] = binomial_iid(n, alpha, y.mean())
            except:
                avg_ci_classical = None

    avg_ci = ci.mean(axis=0)[-1]

    try:
        ci_imputed = binomial_iid(Yhat_unlabeled.shape[0], alpha, imputed_estimate)
    except:
        ci_imputed = None
    try:
        avg_ci_classical = ci_classical.mean(axis=0)[-1]
    except:
        avg_ci_classical = None

    return avg_ci, avg_ci_classical, ci_imputed

def begin(evaluation_datasets: list, checkpoints: list, labels: list, 
        few_shot_examples_filepath: str) -> pd.DataFrame:
    """
    Begin the evaluation process by printing the evaluation datasets, checkpoints, and labels.
    If a few-shot examples file path is provided, read the file and return the few-shot examples.

    Parameters:
    evaluation_datasets (list): List of evaluation datasets.
    checkpoints (list): List of checkpoints.
    labels (list): List of labels.
    GPT_scoring (bool): Flag to indicate if GPT scoring is used.
    few_shot_examples_filepath (str): File path to the few-shot examples.

    Returns:
    pd.DataFrame: DataFrame containing the few-shot examples if the file path is provided, otherwise None.
    """
    print("--------------------------------------------------------")
    print("Evaluation Sets: " + str(evaluation_datasets))
    print("Checkpoints: " + str(checkpoints))
    print("Labels: " + str(labels))
    print("--------------------------------------------------------")

    few_shot_examples = None
    if few_shot_examples_filepath != "None":
        few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
    
    return few_shot_examples

def clean_document(document: str) -> str:
    """
    Cleans the input document by removing extra newlines, carriage returns, tabs, 
    and replacing certain characters with spaces. It also ensures that the document 
    has consistent spacing.

    Parameters:
    document (str): The document to be cleaned.

    Returns:
    str: The cleaned document.
    """
    # Replace carriage returns and tabs with spaces, and remove extra newlines
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r", " ").replace("\t", " ")).strip()
    
    # Replace '=' and '-' with spaces
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    
    # Ensure consistent spacing
    cleaned_document = (" ").join(cleaned_document.split(" ")) 
    
    # Join the words with a single space
    cleaned_document = " ".join(cleaned_document.split(" "))
    
    return cleaned_document

def clean_query(query: str) -> str:
    """
    Cleans the input query by removing newlines, carriage returns, and tabs, 
    and ensuring that the query has consistent spacing.

    Parameters:
    query (str): The query to be cleaned.

    Returns:
    str: The cleaned query.
    """
    # Replace newlines, carriage returns, and tabs with spaces, and strip leading/trailing spaces
    cleaned_query = query.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    
    return cleaned_query

def filter_dataset(rag_type: str = "question_answering") -> tuple[str, str, str]:
    """
    Generates system prompts for different RAG (Retrieval-Augmented Generation) types.

    Parameters:
    rag_type (str): The type of RAG task. Options are "question_answering", "fact_checking", or "dialogue_agent".

    Returns:
    tuple[str, str, str]: A tuple containing the context relevance system prompt, 
                          answer faithfulness system prompt, and answer relevance system prompt.
    """
    if rag_type == "question_answering":
        context_relevance_system_prompt = (
            "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
            "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

        answer_faithfulness_system_prompt = (
            "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
            "The answer must not offer new information beyond the context provided in the document. "
            "The answer also must not contradict information provided in the document. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

        answer_relevance_system_prompt = (
            "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
            "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

    elif rag_type == "fact_checking":
        context_relevance_system_prompt = (
            "You are an expert fact-checking agent. "
            "Given the following statement and document, you must analyze the provided document and determine whether it is sufficient for determining the statement's factuality. "
            "In your evaluation, you should consider the content of the document and how it relates to the provided statement's factuality. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document is not sufficient. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

        answer_faithfulness_system_prompt = (
            "Given the following statement, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        )

        answer_relevance_system_prompt = (
            "Given the following statement, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided statement. "
            "In your evaluation, you should consider whether the answer addresses all aspects of the statement and provides only correct information from the document for answering the statement. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given statement and "[[No]]" if the answer is not relevant for the given statement. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

    elif rag_type == "dialogue_agent":
        context_relevance_system_prompt = (
            "You are an expert dialogue agent. "
            "Given the following dialogue and document, you must analyze the provided document and determine whether it is relevant for responding to the dialogue. "
            "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

        answer_faithfulness_system_prompt = (
            "Given the following dialogue, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        )

        answer_relevance_system_prompt = (
            "Given the following dialogue, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided dialogue. "
            "In your evaluation, you should consider whether the answer addresses all aspects of the dialogue and provides only correct information from the document for responding to the dialogue. "
            'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given dialogue and "[[No]]" if the answer is not relevant for the given dialogue. '
            "Do not provide any additional explanation for your decision.\n\n"
        )

    else:
        sys.exit("Error: rag_type not specified, please specify in configuration with 'rag_type'")

    return context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt

def preprocess_data(test_set_selection: str, label_column: str, labels: list):
    """
    Preprocesses the data for evaluation.

    Parameters:
    - test_set_selection (str): The file path to the test set selection in CSV format.
    - label_column (str): The column name in the test set that contains the labels.
    - labels (list): A list of labels to be used for filtering the test set.

    Returns:
    - Tuple[pd.DataFrame, str]: A tuple containing the preprocessed test set DataFrame and the name of the text column.
    
    Raises:
    - ValueError: If the dataset has fewer than 10 rows after filtering.
    """
    
    # Read the test set from a CSV file
    test_set = pd.read_csv(test_set_selection, sep="\t")
    
    # Define the text column name
    text_column = 'concat_text'
    
    if label_column in test_set.columns:
        test_set = test_set[test_set[label_column].notna()]
    
    # Combine query and document (and answer if applicable) into the text column
    if "Context" in label_column:
        test_set[text_column] = [
            combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document']) 
            for i in range(len(test_set))
        ]
    else:
        test_set[text_column] = [
            combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document'], test_set.iloc[i]['Answer']) 
            for i in range(len(test_set))
        ]

    # Filter out rows where the text column has the value "Error"
    test_set = test_set[test_set[text_column] != "Error"]
    
    # Check if the dataset has fewer than 10 rows after filtering
    if len(test_set) < 10:
        raise ValueError("Insufficient Data: Dataset has fewer than 10 rows after filtering!")
    
    return test_set, text_column

        ############################################################

def togetherai_list_models(api_key: str) -> list:
    """
    Lists available models from the Together API.

    Parameters:
    - api_key (str): The API key for authentication.

    Returns:
    - list: A list of model names available from the Together API. If an error occurs, returns ["N/A"].
    """
    if not api_key:
        return []

    try:
        # Running the command to list models
        result = subprocess.run(['together', 'models', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        models = []

        # Parse the output to extract model names
        for line in lines[3:]:
            if '|' in line:
                parts = line.split('|')
                if len(parts) > 1:
                    model_name = parts[1].strip()
                    models.append(model_name)

        return models

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return ["N/A"]

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ["N/A"]

def load_api_model(model_identifier: str, vllm: bool) -> str:
    """
    Loads an API model based on the provided model identifier.

    Parameters:
    - model_identifier (str): The identifier of the model to load.
    - vllm (bool): A flag indicating whether to use vllm.

    Returns:
    - str: The model identifier if it corresponds to an API model.

    Raises:
    - ValueError: If the model identifier does not correspond to an API model.
    """
    api_key = os.getenv("TOGETHER_API_KEY")
    together_models = togetherai_list_models(api_key)
    
    if model_identifier in together_models or "gpt" in model_identifier or "claude" in model_identifier or vllm:
        model = model_identifier
        print("Loaded API model based on model identifier:", model_identifier)
        return model
    else:
        raise ValueError("Model identifier does not correspond to an API model.")

def load_tokenizer_and_model(model_identifier: str, number_of_labels: int, checkpoint: str = None) -> tuple:
    """
    Loads a tokenizer and model based on the provided model identifier and number of labels.

    Parameters:
    - model_identifier (str): The identifier of the model to load.
    - number_of_labels (int): The number of labels for the model.
    - checkpoint (str, optional): The path to a checkpoint file to load the model state from.

    Returns:
    - tuple: A tuple containing the model, tokenizer, and device.

    Raises:
    - FileNotFoundError: If the checkpoint file is not found.
    """
    max_token_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_token_length)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")

    model = CustomBERTModel(number_of_labels, model_identifier)
    model.to(device)

    if checkpoint:
        checkpoint_dict = torch.load(checkpoint)
        if "encoderModel.embeddings.position_ids" in checkpoint_dict:
            del checkpoint_dict["encoderModel.embeddings.position_ids"]
        model.load_state_dict(checkpoint_dict)
        print("Loaded model from checkpoint:", checkpoint)
    else:
        print("Loaded model based on model identifier:", model_identifier)

    return model, tokenizer, device

    ############################################################

def evaluate_model(params: dict) -> tuple:
    """
    Evaluates a model based on the provided parameters.

    Parameters:
    - params (dict): A dictionary containing the following keys:
        - test_set (pd.DataFrame): The test dataset.
        - label_column (str): The column name for labels in the test set.
        - text_column (str): The column name for text in the test set.
        - device (str): The device to run the model on (e.g., 'cuda:0').
        - checkpoint (str): The path to a checkpoint file to load the model state from.
        - tokenizer (AutoTokenizer): The tokenizer to use.
        - model (CustomBERTModel): The model to evaluate.
        - assigned_batch_size (int): The batch size for evaluation.
        - model_choice (str): The choice of model.
        - context_relevance_system_prompt (str): The system prompt for context relevance.
        - answer_faithfulness_system_prompt (str): The system prompt for answer faithfulness.
        - answer_relevance_system_prompt (str): The system prompt for answer relevance.
        - few_shot_examples_filepath (str): The file path to few-shot examples.
        - llm_judge (str): The LLM judge to use.
        - vllm (bool): Flag indicating if vllm is used.
        - host_url (str): The host URL for the LLM judge.
        - request_delay (int): The delay between requests.
        - debug_mode (bool): Flag indicating if debug mode is enabled.

    Returns:
    - tuple: A tuple containing total_predictions, total_references, results, and metric.
    """
    test_set = params["test_set"]
    label_column = params["label_column"]
    text_column = params["text_column"]
    device = params["device"]
    checkpoint = params["checkpoint"]
    tokenizer = params["tokenizer"]
    model = params["model"]
    assigned_batch_size = params["assigned_batch_size"]
    model_choice = params["model_choice"]
    context_relevance_system_prompt = params["context_relevance_system_prompt"]
    answer_faithfulness_system_prompt = params["answer_faithfulness_system_prompt"]
    answer_relevance_system_prompt = params["answer_relevance_system_prompt"]
    few_shot_examples_filepath = params["few_shot_examples_filepath"]
    llm_judge = params["llm_judge"]
    vllm = params["vllm"]
    host_url = params["host_url"]
    request_delay = params["request_delay"]
    debug_mode = params["debug_mode"]

    metric = evaluate.load("accuracy")

    if checkpoint:
        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)
        total_logits = torch.FloatTensor([]).to(device)
        eval_dataloader = prepare_dataset_for_evaluation(test_set, label_column, text_column, assigned_batch_size, tokenizer)
        model.eval()
        with tqdm(eval_dataloader, desc="Evaluating", leave=False) as progress_bar:
            for batch in progress_bar:
                with torch.no_grad():
                    if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
                    else:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)

                    outputs = model(**new_batch)

                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)
                    
                    if 'labels' in batch:
                        # Add the batch to the metric
                        metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

                        # Concatenate the references for later use
                        total_references = torch.cat((total_references, batch['labels'].to(device)), 0)

                    total_predictions = torch.cat((total_predictions, predictions), 0)
                    total_logits = torch.cat((total_logits, logits), 0)

                    progress_bar.update(1)
    else:
        print("Performing Model scoring!")
        few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
        try:
            _ = few_shot_examples.iloc[0]['Query']
            query_id = "Query"
        except KeyError:
            try:
                _ = few_shot_examples.iloc[0]['Question']
                query_id = "Question"
            except KeyError:
                sys.exit("Both 'Query' and 'Question' keys are missing for the given row in few shot dataset.")

        failed_extraction_count = 0
        
        if "Context_Relevance_Label" == label_column:
            if vllm:
                test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring_vllm(
                    context_relevance_system_prompt, clean_query(x[query_id]), x["Document"], llm_judge, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "gpt" in llm_judge:
                test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring(
                    context_relevance_system_prompt, clean_query(x[query_id]), x["Document"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "claude" in llm_judge:
                test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring_claude(
                    context_relevance_system_prompt, clean_query(x[query_id]), x["Document"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            else:
                test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring_togetherai(
                    context_relevance_system_prompt, clean_query(x[query_id]), x["Document"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
        elif "Answer_Faithfulness_Label" == label_column:
            if vllm:
                test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring_vllm(
                    answer_faithfulness_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "gpt" in llm_judge:
                test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(
                    answer_faithfulness_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "claude" in llm_judge:
                test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring_claude(
                    answer_faithfulness_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            else:
                test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring_togetherai(
                    answer_faithfulness_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
        elif "Answer_Relevance_Label" == label_column:
            if vllm:
                test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring_vllm(
                    answer_relevance_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "gpt" in llm_judge:
                test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring(
                    answer_relevance_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            elif "claude" in llm_judge:
                test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring_claude(
                    answer_relevance_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)
            else:
                test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring_togetherai(
                    answer_relevance_system_prompt, clean_query(x[query_id]), x["Document"], x["Answer"], llm_judge, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples), axis=1)

        total_predictions = test_set[label_column.replace("_Label", "_Prediction")].to_numpy()
        total_references = test_set[label_column].to_numpy()
    
    
    if total_references.nelement() > 0: 
        results = metric.compute(references=total_references, predictions=total_predictions)
    else:
        results = None

    return total_predictions, total_references, results, metric

def validate_input(machine_label_path: str, machine_label_llm_model: str) -> None:
    """
    Validates the input parameters for machine labeling.

    Parameters:
    - machine_label_path (str): The path to the machine label file.
    - machine_label_llm_model (str): The LLM model used for machine labeling.

    Raises:
    - ValueError: If either machine_label_path or machine_label_llm_model is not provided.
    """
    if machine_label_path == "None":
        raise ValueError("Error: machine_label_path is not provided.")
    
    if machine_label_llm_model == "None":
        raise ValueError("Error: machine_label_llm_model is not provided.")
    
def preprocess_text(s: str) -> str:
    """
    Preprocesses the input text by replacing newlines and tabs with spaces.

    Parameters:
    - s (str): The input string to preprocess.

    Returns:
    - str: The preprocessed string.
    """
    return s.replace('\n', ' ').replace('\t', ' ')

def create_machine_label_file(machine_label_path: str, unlabeled_eval_set: pd.DataFrame, label_column: str) -> None:
    """
    Creates a machine label file from the unlabeled evaluation set.

    Parameters:
    - machine_label_path (str): The path to save the machine label file.
    - unlabeled_eval_set (pd.DataFrame): The DataFrame containing the unlabeled evaluation set.
    - label_column (str): The column name for the label to be created.

    Returns:
    - None
    """
    # Slice the first 500 rows from the DataFrame
    machine_labels = unlabeled_eval_set.head(1000)

    # Apply cleanup function to all string columns to replace problematic characters
    def clean_text(text: str) -> str:
        if isinstance(text, str):
            return text.replace('\t', ' ').replace('\n', ' ').replace('\\', '\\\\')
        return text

    # Clean up all columns
    for col in machine_labels.columns:
        machine_labels[col] = machine_labels[col].apply(clean_text)

    # Initialize the specified label column to be empty
    machine_labels[label_column] = ''

    # Establish the desired column order and ensure all necessary columns are present
    column_order = ["Query", "Document", "Answer", label_column]

    # Ensure the DataFrame only contains the specified columns, in order
    machine_labels = machine_labels[column_order]

    # Write the DataFrame to a TSV file with the specified quoting and escaping rules
    machine_labels.to_csv(machine_label_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

def determine_query_column(machine_labels: pd.DataFrame, few_shot_examples: pd.DataFrame):
    """
    Determines the query column and query ID from the machine labels and few shot examples.

    Parameters:
    - machine_labels (pd.DataFrame): The DataFrame containing the machine labels.
    - few_shot_examples (pd.DataFrame): The DataFrame containing the few shot examples.

    Returns:
    - Tuple[pd.Series, str]: A tuple containing the query series and the query ID.

    Raises:
    - SystemExit: If both 'Query' and 'Question' keys are missing for the given row.
    """
    # Check if the query column is 'Query' or 'Question'
    try:
        query = machine_labels['Query']
    except KeyError:
        query = machine_labels['Question']

    # Check if the query_id is 'Query' or 'Question'
    try:
        _ = few_shot_examples.iloc[0]['Query']
        query_id = "Query"
    except KeyError:
        try:
            _ = few_shot_examples.iloc[0]['Question']
            query_id = "Question"
        except KeyError:
            sys.exit("Both 'Query' and 'Question' keys are missing for the given row.")
    return query, query_id

def apply_labeling_functions(
    machine_labels: pd.DataFrame, 
    query: pd.Series, 
    machine_label_llm_model: str, 
    query_id: str, 
    vllm: bool, 
    host_url: str, 
    debug_mode: bool, 
    request_delay: int, 
    failed_extraction_count: int, 
    few_shot_examples: pd.DataFrame, 
    machine_label_prompt: str, 
    label_column: str
) -> None:
    """
    Applies labeling functions to each row in the machine_labels DataFrame.

    Parameters:
    - machine_labels (pd.DataFrame): The DataFrame containing the machine labels.
    - query (pd.Series): The series containing the queries.
    - machine_label_llm_model (str): The model used for labeling.
    - query_id (str): The identifier for the query.
    - vllm (bool): Flag indicating whether to use vllm.
    - host_url (str): The host URL for the model.
    - debug_mode (bool): Flag indicating whether to run in debug mode.
    - request_delay (int): The delay between requests.
    - failed_extraction_count (int): The count of failed extractions.
    - few_shot_examples (pd.DataFrame): The DataFrame containing few-shot examples.
    - machine_label_prompt (str): The prompt used for machine labeling.
    - label_column (str): The column to store the labels.

    Returns:
    - None
    """
    # Loop through each row in the DataFrame and apply labeling functions
    for index, row in tqdm(machine_labels.iterrows(), total=machine_labels.shape[0], desc="Generating machine labels!"):
        current_query = query.iloc[index]
        
        if "gpt" in machine_label_llm_model:
            if vllm:
                context_relevance_score = few_shot_context_relevance_scoring_vllm(
                    machine_label_prompt, current_query, row['Document'], machine_label_llm_model, 
                    query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples
                )
            else:
                context_relevance_score = few_shot_context_relevance_scoring(
                    machine_label_prompt, current_query, row['Document'], machine_label_llm_model, 
                    query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples
                )
        elif "claude" in machine_label_llm_model:
            context_relevance_score = few_shot_context_relevance_scoring_claude(
                machine_label_prompt, current_query, row['Document'], machine_label_llm_model, 
                query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples
            )
        else:
            if vllm:
                context_relevance_score = few_shot_context_relevance_scoring_vllm(
                    machine_label_prompt, current_query, row['Document'], machine_label_llm_model, 
                    query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples
                )
            else:
                context_relevance_score = few_shot_context_relevance_scoring_togetherai(
                    machine_label_prompt, current_query, row['Document'], machine_label_llm_model, 
                    query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples
                )

        if context_relevance_score == 0 and label_column in ["Answer_Relevance_Label", "Answer_Faithfulness_Label"]:
            answer_relevance_score = 0
            answer_faithfulness_score = 0
        else:
            if label_column == "Answer_Relevance_Label":
                if "gpt" in machine_label_llm_model:
                    if vllm:
                        answer_relevance_score = few_shot_answer_relevance_scoring_vllm(
                            machine_label_prompt, current_query, row['Document'], row['Answer'], 
                            machine_label_llm_model, query_id, debug_mode, host_url, request_delay, 
                            failed_extraction_count, few_shot_examples
                        )
                    else:
                        answer_relevance_score = few_shot_answer_relevance_scoring(
                            machine_label_prompt, current_query, row['Document'], row['Answer'], 
                            machine_label_llm_model, query_id, debug_mode, request_delay, 
                            failed_extraction_count, few_shot_examples
                        )
                elif "claude" in machine_label_llm_model:
                    answer_relevance_score = few_shot_answer_relevance_scoring_claude(
                        machine_label_prompt, current_query, row['Document'], row['Answer'], 
                        machine_label_llm_model, query_id, debug_mode, request_delay, 
                        failed_extraction_count, few_shot_examples
                    )
                else:
                    answer_relevance_score = few_shot_answer_relevance_scoring_togetherai(
                        machine_label_prompt, current_query, row['Document'], row['Answer'], 
                        machine_label_llm_model, query_id, debug_mode, request_delay, 
                        failed_extraction_count, few_shot_examples
                    )
                machine_labels.at[index, label_column] = answer_relevance_score

            elif label_column == "Answer_Faithfulness_Label":
                if "gpt" in machine_label_llm_model:
                    if vllm:
                        answer_faithfulness_score = few_shot_answer_faithfulness_scoring_vllm(
                            machine_label_prompt, current_query, row['Document'], row['Answer'], 
                            machine_label_llm_model, query_id, debug_mode, host_url, request_delay, 
                            failed_extraction_count, few_shot_examples
                        )
                    else:
                        answer_faithfulness_score = few_shot_answer_faithfulness_scoring(
                            machine_label_prompt, current_query, row['Document'], row['Answer'], 
                            machine_label_llm_model, query_id, debug_mode, request_delay, 
                            failed_extraction_count, few_shot_examples
                        )
                elif "claude" in machine_label_llm_model:
                    answer_faithfulness_score = few_shot_answer_faithfulness_scoring_claude(
                        machine_label_prompt, current_query, row['Document'], row['Answer'], 
                        machine_label_llm_model, query_id, debug_mode, request_delay, 
                        failed_extraction_count, few_shot_examples
                    )
                else:
                    answer_faithfulness_score = few_shot_answer_faithfulness_scoring_togetherai(
                        machine_label_prompt, current_query, row['Document'], row['Answer'], 
                        machine_label_llm_model, query_id, debug_mode, request_delay, 
                        failed_extraction_count, few_shot_examples
                    )
                machine_labels.at[index, label_column] = answer_faithfulness_score

        if label_column == "Context_Relevance_Label":
            machine_labels.at[index, label_column] = context_relevance_score
        
def generate_machine_labels(
    machine_label_path: str, 
    unlabeled_eval_set: pd.DataFrame, 
    machine_label_prompt: str, 
    machine_label_llm_model: str, 
    vllm: bool, 
    host_url: str, 
    debug_mode: bool, 
    request_delay: int, 
    label_column: str, 
    few_shot_examples: list
) -> None:
    """
    Generates machine labels for the given evaluation set and saves them to a file.

    Parameters:
    - machine_label_path (str): Path to the file where machine labels will be saved.
    - unlabeled_eval_set (pd.DataFrame): The evaluation set without labels.
    - machine_label_prompt (str): The prompt to be used for generating machine labels.
    - machine_label_llm_model (str): The language model to be used for generating labels.
    - vllm (bool): Flag indicating whether to use vLLM for scoring.
    - host_url (str): The host URL for the language model service.
    - debug_mode (bool): Flag indicating whether to run in debug mode.
    - request_delay (int): Delay between requests to the language model service.
    - label_column (str): The column name for the label to be generated.
    - few_shot_examples (list): List of few-shot examples to be used for generating labels.

    Returns:
    - None
    """
    
    failed_extraction_count = {'failed': 0}
    
    # Validate the input parameters
    validate_input(machine_label_path, machine_label_llm_model)
    
    # Create the machine label file
    create_machine_label_file(machine_label_path, unlabeled_eval_set, label_column)
    
    # Read the machine labels from the file
    machine_labels = pd.read_csv(machine_label_path, sep='\t')
    
    # Determine the query column and query ID
    query, query_id = determine_query_column(machine_labels, few_shot_examples)
    
    # Apply the labeling functions to generate the labels
    apply_labeling_functions(
        machine_labels, query, machine_label_llm_model, query_id, vllm, host_url, 
        debug_mode, request_delay, failed_extraction_count, few_shot_examples, 
        machine_label_prompt, label_column
    )
    
    # Save the updated DataFrame back to the TSV file
    machine_labels.to_csv(machine_label_path, sep='\t', index=False)

def post_process_predictions(params: dict):
    checkpoint = params["checkpoint"]
    test_set = params["test_set"]
    label_column = params["label_column"]
    total_predictions = params["total_predictions"]
    labels = params["labels"]
    gold_label_path = params["gold_label_path"]
    tokenizer = params["tokenizer"]
    assigned_batch_size = params["assigned_batch_size"]
    device = params["device"]
    machine_label_path = params["gold_machine_label_path"]
    unlabeled_eval_set = params["test_set"]
    machine_label_prompt = params["machine_label_system_prompt"]
    machine_label_llm_model = params["machine_label_llm_model"]
    vllm = params["vllm"]
    host_url = params["host_url"]
    debug_mode = params["debug_mode"]
    request_delay = params["request_delay"]
    few_shot_examples = params["few_shot_examples"]

    prediction_column = label_column + "_Model_Predictions"
    test_set[prediction_column] = total_predictions.tolist()
    
    if label_column in test_set.columns:
        test_set = test_set[test_set[label_column].notna()]
        
    for label in labels:
        if label != label_column:
            test_set = test_set[test_set[label] != 0]

    # Generate machine labels if parameters are provided
    if machine_label_path != "None" and machine_label_llm_model != "None":
        generate_machine_labels(machine_label_path, unlabeled_eval_set, machine_label_prompt, machine_label_llm_model, vllm, host_url, debug_mode, request_delay, label_column, few_shot_examples)
        Y_labeled_dataset = pd.read_csv(machine_label_path, sep='\t')
        Y_labeled_dataset = Y_labeled_dataset.head(500)
    else:
        Y_labeled_dataset = pd.read_csv(gold_label_path, sep="\t")
        Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[label_column].notna()].head(300)

    text_column = 'concat_text'
    if "Context" in label_column:
        Y_labeled_dataset[text_column] = [combine_query_document(Y_labeled_dataset.iloc[i]['Query'], Y_labeled_dataset.iloc[i]['Document']) for i in range(len(Y_labeled_dataset))]
    else:
        Y_labeled_dataset[text_column] = [combine_query_document(Y_labeled_dataset.iloc[i]['Query'], Y_labeled_dataset.iloc[i]['Document'], Y_labeled_dataset.iloc[i]['Answer']) for i in range(len(Y_labeled_dataset))]

    Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[text_column] != "Error"]

    if checkpoint:
        Y_labeled_dataloader = prepare_dataset_for_evaluation(Y_labeled_dataset, label_column, text_column, assigned_batch_size, tokenizer)
    else:
        Y_labeled_dataloader = None

    Y_labeled_predictions = torch.FloatTensor([]).to(device)
    Yhat_unlabeled_dataset = test_set

    return test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column

def evaluate_and_scoring_data(params: dict):
    test_set = params["test_set"]
    Y_labeled_predictions = params["Y_labeled_predictions"]
    Y_labeled_dataset = params["Y_labeled_dataset"]
    Y_labeled_dataloader = params["Y_labeled_dataloader"]
    Yhat_unlabeled_dataset = params["Yhat_unlabeled_dataset"]
    alpha = params["alpha"]
    num_trials = params["num_trials"]
    model = params["model"]
    device = params["device"]
    model_choice = params["model_choice"]
    context_relevance_system_prompt = params["context_relevance_system_prompt"]
    answer_faithfulness_system_prompt = params["answer_faithfulness_system_prompt"]
    answer_relevance_system_prompt = params["answer_relevance_system_prompt"]
    few_shot_examples = params["few_shot_examples"]
    metric = params["metric"]
    prediction_column = params["prediction_column"]
    label_column = params["label_column"]
    test_set_selection = params["test_set_selection"]
    LLM_judge_ratio_predictions = params["LLM_judge_ratio_predictions"]
    validation_set_lengths = params["validation_set_lengths"]
    validation_set_ratios = params["validation_set_ratios"]
    ppi_confidence_intervals = params["ppi_confidence_intervals"]
    accuracy_scores = params["accuracy_scores"]
    results = params["results"]
    checkpoint = params["checkpoint"]
    llm_judge = params["llm_judge"]
    vllm = params["vllm"]
    host_url = params["host_url"]
    request_delay = params["request_delay"]
    debug_mode = params["debug_mode"]
    prediction_filepath = params["prediction_filepath"]
    
    failed_extraction_count = {'failed': 0}  # Reset failed extraction count

    if checkpoint:
        model.eval()
        with tqdm(Y_labeled_dataloader, desc="Scoring", leave=False) as progress_bar:
            for batch in progress_bar:
                with torch.no_grad():
                    new_batch = {
                        'ids': batch['input_ids'].to(device),
                        'mask': batch['attention_mask'].bool().to(device) if model_choice in ["mosaicml/mpt-1b-redpajama-200b"] else batch['attention_mask'].to(device)
                    }
                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)
                    
                    outputs = model(**new_batch)
                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)

                    if 'labels' in batch:
                        # Get the labels from the batch
                        labels = batch['labels'].to(device)

                        # Add the batch to the metric
                        metric.add_batch(predictions=predictions, references=labels)

                    # Concatenate all predictions for later use
                    Y_labeled_predictions = torch.cat((Y_labeled_predictions, predictions), 0)
                    progress_bar.update(1)
                    
        Y_labeled_dataset[prediction_column] = Y_labeled_predictions.detach().cpu().numpy().tolist()
        Yhat_unlabeled_dataset = test_set
    else:
        if llm_judge == "None":
            sys.exit("Error: No llm_judge provided")
        
        elif "gpt" in llm_judge or "claude" in llm_judge:
            Y_labeled_predictions = []

            query_id = "Query" if 'Query' in few_shot_examples.columns else "Question"
            query_labeled_id = "Query" if 'Query' in Y_labeled_dataset.columns else "Question"

            with tqdm(total=len(Y_labeled_dataset), desc="Evaluating", leave=False) as progress_bar:
                for _, row in Y_labeled_dataset.iterrows():
                    query = row[query_labeled_id]
                    document = row["Document"]
                    answer = row["Answer"]
                    
                    if "Context_Relevance_Label" == label_column:
                        if vllm:
                            score = few_shot_context_relevance_scoring_vllm(context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        elif "gpt" in llm_judge:
                            score = few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(query), document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        elif "claude" in llm_judge:
                            score = few_shot_context_relevance_scoring_claude(context_relevance_system_prompt, clean_query(query), document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_context_relevance_scoring_togetherai(context_relevance_system_prompt, clean_query(query), document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    elif "Answer_Faithfulness_Label" == label_column:
                        if vllm:
                            score = few_shot_answer_faithfulness_scoring_vllm(answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        elif "gpt" in llm_judge:
                            score = few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        elif "claude" in llm_judge:
                            score = few_shot_answer_faithfulness_scoring_claude(answer_faithfulness_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_answer_faithfulness_scoring_togetherai(answer_faithfulness_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    elif "Answer_Relevance_Label" == label_column:
                        if vllm:
                            score = few_shot_answer_relevance_scoring_vllm(answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        elif "gpt" in llm_judge:
                            score = few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        elif "claude" in llm_judge:
                            score = few_shot_answer_relevance_scoring_claude(answer_relevance_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_answer_relevance_scoring_togetherai(answer_relevance_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    Y_labeled_predictions.append(score)
                    progress_bar.update(1)
            
            Y_labeled_predictions_np = np.array(Y_labeled_predictions)
            Y_labeled_dataset[prediction_column] = Y_labeled_predictions_np.tolist()
        else:
            Y_labeled_predictions = []

            query_id = "Query" if 'Query' in few_shot_examples.columns else "Question"
            query_labeled_id = "Query" if 'Query' in Y_labeled_dataset.columns else "Question"

            with tqdm(total=len(Y_labeled_dataset), desc="Evaluating", leave=False) as progress_bar:
                for _, row in Y_labeled_dataset.iterrows():
                    query = row[query_labeled_id]
                    document = row["Document"]
                    answer = row["Answer"]
                    
                    if "Context_Relevance_Label" == label_column:
                        if vllm:
                            score = few_shot_context_relevance_scoring_vllm(context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_context_relevance_scoring_togetherai(context_relevance_system_prompt, clean_query(query), document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    elif "Answer_Faithfulness_Label" == label_column:
                        if vllm:
                            score = few_shot_answer_faithfulness_scoring_vllm(answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_answer_faithfulness_scoring_togetherai(answer_faithfulness_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    elif "Answer_Relevance_Label" == label_column:
                        if vllm:
                            score = few_shot_answer_relevance_scoring_vllm(answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, few_shot_examples)
                        else:
                            score = few_shot_answer_relevance_scoring_togetherai(answer_relevance_system_prompt, clean_query(query), document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, few_shot_examples)
                    
                    Y_labeled_predictions.append(score)
                    progress_bar.update(1)
            
            Y_labeled_predictions_np = np.array(Y_labeled_predictions)
            Y_labeled_dataset[prediction_column] = Y_labeled_predictions_np.tolist()

    Y_labeled = Y_labeled_dataset[label_column].values.astype(int)
    Yhat_labeled = Y_labeled_dataset[prediction_column].values.astype(int)
    Yhat_unlabeled = Yhat_unlabeled_dataset[prediction_column].values.astype(int)
    
    avg_ci, avg_ci_classical, ci_imputed = calculate_ppi(Y_labeled, Yhat_labeled, Yhat_unlabeled, alpha, num_trials)
    LLM_judge_prediction = sum(avg_ci) / len(avg_ci)
    LLM_judge_ratio_predictions.append(LLM_judge_prediction)
    validation_set_lengths.append(len(test_set))
    ppi_confidence_intervals.append([round(value, 3) for value in avg_ci])

    # Check if the ground truth label column exists and has any non-null values
    if label_column in Yhat_unlabeled_dataset.columns and not Yhat_unlabeled_dataset[label_column].isnull().all():
        validation_set_ratios.append(round(Yhat_unlabeled_dataset[label_column].tolist().count(1) / len(Yhat_unlabeled_dataset), 3))
        accuracy_scores.append(round(results['accuracy'], 3))
        ground_truth_available = True
    else:
        ground_truth_available = False

    results = {
        "Label_Column": label_column,
        "Evaluation_Set": test_set_selection,
        "ARES_Prediction": LLM_judge_ratio_predictions[-1] if LLM_judge_ratio_predictions else None,
        "ARES_Confidence_Interval": ppi_confidence_intervals[-1] if ppi_confidence_intervals else None,
        "Number_of_Examples_in_Evaluation_Set": validation_set_lengths[-1] if validation_set_lengths else None,
        "Ground_Truth_Performance": validation_set_ratios[-1] if ground_truth_available and validation_set_ratios else None,
        "ARES_LLM_Judge_Accuracy_on_Ground_Truth_Labels": accuracy_scores[-1] if ground_truth_available and accuracy_scores else None,
        "Annotated_Examples_used_for_PPI": len(Y_labeled)
    }
    # Save the labeled dataset with predictions to a new TSV file
    if prediction_filepath != "None": 
        # Update the prediction column name based on the label column
        if label_column == "Context_Relevance_Label":
            prediction_column_name = "ARES_Context_Relevance_Prediction"
        elif label_column == "Answer_Relevance_Label":
            prediction_column_name = "ARES_Answer_Relevance_Prediction"
        elif label_column == "Answer_Faithfulness_Label":
            prediction_column_name = "ARES_Answer_Faithfulness_Prediction"

        Yhat_unlabeled_dataset.rename(columns={prediction_column: prediction_column_name}, inplace=True)
        Yhat_unlabeled_dataset.to_csv(prediction_filepath, sep='\t', index=False)
        print("--------------------------------------------------")
        print(f"Labeled dataset with predictions saved to {prediction_filepath}")
        print("--------------------------------------------------")

    print("--------------------------------------------------")
    print(label_column + " Scoring")
    print("ARES Ranking")
    print("Evaluation_Set:" +str(test_set_selection))
    print("Checkpoint:" +str(checkpoint))
    print("ARES Prediction: " + str(LLM_judge_ratio_predictions))
    print("ARES Confidence Interval: " + str(ppi_confidence_intervals))
    print("Number of Examples in Evaluation Set: " + str(validation_set_lengths))
    if ground_truth_available:
        print("Ground Truth Performance: " + str(validation_set_ratios))
    if ground_truth_available:
        print("ARES LLM Judge Accuracy on Ground Truth Labels: " + str(accuracy_scores))
    print("Annotated Examples used for PPI: " + str(len(Y_labeled)))
    print("--------------------------------------------------\n")

    return results