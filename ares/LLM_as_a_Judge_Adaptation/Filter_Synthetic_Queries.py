
import ast
import copy
import csv
import json
import pdb
import random
import time
import warnings

import numpy as np
import openai
import pandas as pd
import requests
import torch
from datasets import Dataset
from openai import OpenAI
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generates an embedding for the given text using the specified model.

    Parameters:
    text (str): The input text to generate the embedding for.
    model (str): The model to use for generating the embedding. Default is "text-embedding-ada-002".

    Returns:
    list: A list representing the embedding of the input text.
    """
    client = OpenAI()
    
    # Replace newline characters with spaces
    text = text.replace("\n", " ")
    
    # Truncate text to the first 50 words if it exceeds 50 words
    if len(text) > 50:
        text = " ".join(text.split(" ")[:50])
    
    # Attempt to generate the embedding up to 5 times in case of failure
    for _ in range(5):
        try:
            return client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}. Attempting again...")
            time.sleep(30)

def generate_index(dataframe: pd.DataFrame) -> Dataset:
    """
    Generates an index for the given dataframe by creating embeddings for each document and adding a FAISS index.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing documents to be indexed.

    Returns:
    Dataset: A Hugging Face Dataset object with FAISS index added.
    """
    # Ignore SettingWithCopyWarning warnings
    warnings.simplefilter("ignore", SettingWithCopyWarning)
    
    # Drop duplicate documents
    dataframe = dataframe.drop_duplicates(subset="document")
    
    # Initialize tqdm progress bar for generating embeddings
    tqdm.pandas(desc="Generating embeddings...", total=dataframe.shape[0])
    
    # Generate embeddings for each document
    dataframe['embeddings'] = dataframe["document"].progress_apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    
    # Filter out rows where the embedding length is not 1536
    dataframe = dataframe[dataframe['embeddings'].apply(lambda x: len(x)) == 1536]
    
    # Convert dataframe to Hugging Face Dataset
    dataset = Dataset.from_pandas(dataframe)
    
    # Add FAISS index to the dataset
    dataset.add_faiss_index(column="embeddings")
    
    return dataset

def filter_synthetic_queries(queries_dataset: pd.DataFrame, document_index) -> pd.DataFrame:
    """
    Filters synthetic queries based on their relevance to the documents in the document index.

    Parameters:
    queries_dataset (pd.DataFrame): The original dataset containing synthetic queries and their corresponding documents.
    document_index: An index object used to retrieve document samples based on embeddings.

    Returns:
    pd.DataFrame: The updated dataset with a new column 'Context_Relevance_Label' indicating the relevance of each query.
    """
    
    total_filtered_questions = []
    total_labels = []
    
    # Convert the pandas DataFrame to a Hugging Face Dataset
    queries_dataset = Dataset.from_pandas(queries_dataset)
    
    # Iterate over each query in the dataset
    for i in tqdm(range(len(queries_dataset))):
        question = queries_dataset[i]["synthetic_query"]
        embedding = get_embedding(question)
        
        # Check if embedding is valid
        if embedding is None or len(embedding) == 0:
            print(f"Warning: No embedding generated for query '{question}'. Skipping.")
            continue
        question_embedding = np.array(embedding).astype(np.float32)
        
        # Ensure question_embedding is a 2D array with shape (1, 1536)
        if question_embedding.shape != (1536,):
            print(f"Warning: Invalid embedding shape {question_embedding.shape} for query '{question}'. Skipping.")
            continue
        question_embedding = question_embedding.reshape(1, -1)
        
        # Retrieve the nearest examples from the document index
        scores, samples = document_index.get_nearest_examples("embeddings", question_embedding, k=20)
        
        # Check if the top result matches the document in the query dataset
        if samples["document"][0] == queries_dataset[i]["document"]:
            total_labels.append("Yes")
        else:
            # Create negatives by filtering out top-k results 
            found_gold_in_top_k = False
            for j in range(20):
                found_gold_in_top_k = found_gold_in_top_k or (samples["document"][j] == queries_dataset[i]["document"])
            if not found_gold_in_top_k:
                total_labels.append("No")
            else:
                total_labels.append("N/A")
    
    # Convert the Hugging Face Dataset back to a pandas DataFrame
    queries_dataset = queries_dataset.to_pandas()
    
    # Add the 'Context_Relevance_Label' column to the DataFrame
    queries_dataset['Context_Relevance_Label'] = total_labels

    return queries_dataset

def generate_additional_negatives(queries_dataset: pd.DataFrame, document_index, 
number_of_negatives_added_ratio: float, lower_bound_for_negatives: int) -> pd.DataFrame:
    """
    Generates additional negative examples for the queries dataset.

    This function creates additional negative examples by sampling from the nearest documents that do not match the original document.
    The number of negatives added is determined by the specified ratio.

    Parameters:
    queries_dataset (pd.DataFrame): The original dataset containing synthetic queries and their corresponding documents.
    document_index: An index object used to retrieve document samples based on embeddings.
    number_of_negatives_added_ratio (float): The ratio of the current dataset size to determine the number of negatives to add.
    lower_bound_for_negatives (int): The lower bound index for selecting negative samples from the nearest documents.

    Returns:
    pd.DataFrame: The updated dataset with additional negative examples.
    """
    
    negative_documents = []
    negative_labels = []
    
    # Create a copy of the dataset and remove duplicates
    queries_dataset_copy = queries_dataset.copy().drop_duplicates(["synthetic_query", "document"])
    
    # Sample a subset of the dataset based on the specified ratio
    queries_dataset_copy = queries_dataset_copy.sample(n=int(len(queries_dataset_copy) * number_of_negatives_added_ratio), random_state=42)
    
    negative_sample_retrieved = []

    # Iterate over each query in the sampled dataset
    for i in tqdm(range(len(queries_dataset_copy))):
        question = queries_dataset_copy.iloc[i]["synthetic_query"]
        question_embedding = np.array(get_embedding(question)).astype(np.float32)
        
        # Ensure question_embedding is a 2D array with shape (1, 1536)
        if question_embedding.shape != (1536,):
            raise ValueError(f"Expected embedding of shape (1536,), but got {question_embedding.shape}")

        question_embedding = question_embedding.reshape(1, -1)
        
        # Retrieve the nearest examples from the document index
        scores, samples = document_index.get_nearest_examples("embeddings", question_embedding, k=100)

        # Select a random negative sample from the nearest documents
        random_negative_sample = random.randint(lower_bound_for_negatives, len(samples["document"]) - 1)
        negative_sample_retrieved.append(random_negative_sample)
        
        try:
            negative_documents.append(samples["document"][random_negative_sample])
        except IndexError:
            negative_documents.append(samples["document"][0])
        
        negative_labels.append("No")

    # Update the dataset copy with negative documents and labels
    queries_dataset_copy["document"] = negative_documents
    queries_dataset_copy['Context_Relevance_Label'] = negative_labels
    queries_dataset_copy['negative_sample_retrieved'] = negative_sample_retrieved
    
    # Shuffle the updated dataset copy
    queries_dataset_copy = queries_dataset_copy.sample(n=len(queries_dataset_copy), random_state=42)

    # Concatenate the original dataset with the updated dataset copy
    queries_dataset = pd.concat([queries_dataset, queries_dataset_copy], axis=0, ignore_index=True)
    
    return queries_dataset

def generate_additional_positives(queries_dataset, document_index, number_of_positives_added_ratio: float) -> pd.DataFrame:
    """
    Enhances the queries dataset by adding additional positive examples.
    
    This function selects positive examples from the provided dataset, duplicates them based on the specified ratio,
    and then appends these duplicates back to the original dataset to create a larger set with more positive examples.
    
    Args:
    queries_dataset (DataFrame): The original dataset containing queries.
    document_index (DocumentIndex): An index object used to retrieve document samples based on embeddings.
    number_of_positives_added_ratio (float): The ratio of the current positive examples to generate as additional data.
    
    Returns:
    DataFrame: The updated dataset containing the original data and the added positive examples.
    """
    
    # Copy the dataset and filter for unique positive examples
    queries_dataset_copy = queries_dataset.copy()
    queries_dataset_copy = queries_dataset_copy.drop_duplicates(["synthetic_query", "document"])
    queries_dataset_copy = queries_dataset_copy[queries_dataset_copy['Context_Relevance_Label'] == "Yes"]
    
    # Sample additional positives based on the specified ratio
    queries_dataset_copy = queries_dataset_copy.sample(n=int(len(queries_dataset_copy) * number_of_positives_added_ratio), random_state=42)

    # Initialize lists to store the new positive examples
    positive_queries = []
    positive_documents = []
    positive_labels = []

    # Retrieve embeddings and nearest examples for each query
    for i in tqdm(range(len(queries_dataset_copy))):
        question = queries_dataset_copy.iloc[i]["synthetic_query"]
        question_embedding = np.array(get_embedding(question)).astype(np.float32)
        scores, samples = document_index.get_nearest_examples("embeddings", question_embedding, k=100)
        # Assuming the nearest examples are the positive documents
        positive_documents.extend(samples["document"])
        positive_queries.append(question)
        positive_labels.extend(["Yes"] * len(samples["document"]))

    # Update the dataset with the new positive examples
    queries_dataset_copy = queries_dataset_copy.iloc[:len(positive_queries)]
    queries_dataset_copy["synthetic_query"] = positive_queries
    queries_dataset_copy["document"] = positive_documents
    queries_dataset_copy['Context_Relevance_Label'] = positive_labels
    queries_dataset_copy['generated_answer'] = ["" for _ in range(len(queries_dataset_copy))]
    queries_dataset_copy['additional_positive_added'] = [True for _ in range(len(queries_dataset_copy))]

    # Shuffle the updated dataset
    queries_dataset_copy = queries_dataset_copy.sample(n=len(queries_dataset_copy), random_state=42)

    # Concatenate the original dataset with the new positive examples
    queries_dataset = pd.concat([queries_dataset, queries_dataset_copy], axis=0, ignore_index=True)
    
    return queries_dataset