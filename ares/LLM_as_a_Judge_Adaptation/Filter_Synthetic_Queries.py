
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from openai import OpenAI
from tqdm import tqdm
import csv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
import random
import pdb

#################################################
client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if len(text) > 50:
        text = (" ").join(text.split(" ")[:50])
    for _ in range(5):
        try:
            return client.embeddings.create(input = [text], model=model).data[0].embedding
        except:
            print("Error generating embedding! Attempting again...")
            time.sleep(30)

def generate_index(dataframe):
   dataframe = dataframe.drop_duplicates(subset="document")
   tqdm.pandas(desc="Generating embeddings...", total=dataframe.shape[0])
   dataframe['embeddings'] = dataframe["document"].progress_apply(lambda x: get_embedding(x, model='text-embedding-ada-002')) # model='text-embedding-ada-002'
   dataframe =  dataframe[dataframe['embeddings'].apply(lambda x: len(x)) == 1536]
   
   dataframe = Dataset.from_pandas(dataframe)
   dataframe.add_faiss_index(column="embeddings")
   return dataframe

def filter_synthetic_queries(queries_dataset, document_index):
    
    total_filtered_questions = []
    total_labels = []
    
    queries_dataset = Dataset.from_pandas(queries_dataset)
    for i in tqdm(range(len(queries_dataset))):
        question = queries_dataset[i]["synthetic_query"]
        question_embedding = np.array(get_embedding(question)).astype(np.float32)
        scores, samples = document_index.get_nearest_examples("embeddings", question_embedding, k=20)
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

    #################################################################

    queries_dataset = queries_dataset.to_pandas()
    queries_dataset['Context_Relevance_Label'] = total_labels
    
    print("Before filter")
    print(len(queries_dataset))
    print("After filter")
    print(len(queries_dataset[queries_dataset['Context_Relevance_Label'] == "Yes"]))
    print(len(queries_dataset[queries_dataset['Context_Relevance_Label'] == "No"]))

    return queries_dataset

############################################################

def generate_additional_negatives(queries_dataset, document_index, number_of_negatives_added_ratio: float, lower_bound_for_negatives: int):
    
    negative_documents = []
    negative_labels = []
    queries_dataset_copy = queries_dataset.copy().drop_duplicates(["synthetic_query", "document"])
    queries_dataset_copy = queries_dataset_copy.sample(n=int(len(queries_dataset_copy) * number_of_negatives_added_ratio), random_state=42)
    negative_sample_retrieved = []

    for i in tqdm(range(len(queries_dataset_copy))):
        question = queries_dataset_copy.iloc[i]["synthetic_query"]
        question_embedding = np.array(get_embedding(question)).astype(np.float32)
        scores, samples = document_index.get_nearest_examples(
            "embeddings", question_embedding, k=100
        )
        # if len(samples["document"]) <= 98:
        print(f"Number of samples: {len(samples['document'])}")
        #    raise ValueError('Less than 100 documents in dataset! Please add more documents for retrieval.')
        random_negative_sample = random.randint(lower_bound_for_negatives, len(samples["document"]) - 1)
        negative_sample_retrieved.append(random_negative_sample)
        try:
            negative_documents.append(samples["document"][random_negative_sample])
        except:
            breakpoint()
            negative_documents.append(samples["document"][0])
        negative_labels.append("No")

    # Swap documents with negative documents + negative labels
    queries_dataset_copy["document"] = negative_documents
    queries_dataset_copy['Context_Relevance_Label'] = negative_labels
    queries_dataset_copy['negative_sample_retrieved'] = negative_sample_retrieved

    print("Negatives Added")
    print(len(queries_dataset_copy))
    
    # Shuffle dataframe
    queries_dataset_copy = queries_dataset_copy.sample(n=len(queries_dataset_copy), random_state=42)

    queries_dataset = pd.concat([queries_dataset, queries_dataset_copy], axis=0, ignore_index=True)
    return queries_dataset

############################################################

def generate_additional_positives(queries_dataset, document_index, number_of_positives_added_ratio: float):
    
    positive_queries = []
    positive_documents = []
    positive_labels = []
    
    queries_dataset_copy = queries_dataset.copy()
    queries_dataset_copy = queries_dataset_copy.drop_duplicates(["synthetic_query", "document"])
    queries_dataset_copy = queries_dataset_copy[queries_dataset_copy['Context_Relevance_Label'] == "Yes"]
    queries_dataset_copy = queries_dataset_copy.sample(n=int(len(queries_dataset_copy) * number_of_positives_added_ratio), random_state=42)

    for i in tqdm(range(len(queries_dataset_copy))):
        question = queries_dataset_copy.iloc[i]["synthetic_query"]
        question_embedding = np.array(get_embedding(question)).astype(np.float32)
        scores, samples = document_index.get_nearest_examples(
            "embeddings", question_embedding, k=100
        )

    # Swap documents with positive docs + labels
    if len(positive_queries) < len(queries_dataset_copy):
        queries_dataset_copy = queries_dataset_copy[:len(positive_queries)]
    queries_dataset_copy["synthetic_query"] = positive_queries[:len(queries_dataset_copy)]
    queries_dataset_copy["document"] = positive_documents[:len(queries_dataset_copy)]
    queries_dataset_copy['Context_Relevance_Label'] = positive_labels[:len(queries_dataset_copy)]
    queries_dataset_copy['generated_answer'] = ["" for j in range(len(queries_dataset_copy))]
    queries_dataset_copy['additional_positive_added'] = [True for j in range(len(queries_dataset_copy))]

    print("Positives Added")
    print(len(queries_dataset_copy))

    # Shuffle dataframe
    queries_dataset_copy = queries_dataset_copy.sample(n=len(queries_dataset_copy), random_state=42)

    queries_dataset = pd.concat([queries_dataset, queries_dataset_copy], axis=0, ignore_index=True)
    return queries_dataset


