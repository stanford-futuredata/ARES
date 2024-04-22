
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import re
import random
import os

####################################################################

def superGlue(dataset):

    dataset_choices = [dataset] #"record", "rte", "boolq", "multirc"

    positive_negative_ratios = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]

    ####################################################################

    for dataset_chosen in dataset_choices:

        for split in ['train', 'validation']:

            folder_path = "../datasets_v2/" + dataset_chosen + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            dataset = load_dataset("super_glue", dataset_chosen)[split]
            print("Preparing SuperGlue dataset...")
            dataset = dataset.to_pandas()
            #dataset['index_column'] = dataset['idx']
            #dataset = dataset.set_index("index_column")

            if dataset_chosen == "multirc":
                dataset = dataset.rename(columns={"paragraph": "Document",
                                                "question": "Query",
                                                "answer": "Answer"})
            elif dataset_chosen == "record":
                dataset = dataset.rename(columns={"passage": "Document",
                                                "query": "Query",
                                                "answers": "Answer"})
                dataset['Original_Answer_Lists'] = dataset['Answer']
                dataset['Answer'] = [item.tolist() for item in dataset['Answer'].tolist()]
                dataset['entities'] = [item.tolist() for item in dataset['entities'].tolist()]
                dataset = dataset.explode("Answer", ignore_index=True)
                dataset = dataset.drop(columns=['idx', 'entity_spans'])
                
            ####################################################################

            dataset['Context_Relevance_Label'] = [1 for _ in range(len(dataset))] 
            dataset['Answer_Faithfulness_Label'] = [1 for _ in range(len(dataset))] 
            dataset['Answer_Relevance_Label'] = [1 for _ in range(len(dataset))] 

            file_path = folder_path + dataset_chosen + "_" + split + ".tsv"
            dataset.to_csv(file_path, sep="\t", index=False)
            print("Saved file to: " + file_path)

            ####################################################################
            
            total_documents = dataset['Document'].tolist()
            total_documents = list(set(total_documents))
            total_answers = dataset['Answer'].tolist()
            total_answers = list(set(total_answers))

            print("Number of Documents and Answers")
            print(len(total_documents))
            print(len(total_answers))

            ####################################################################

            if split in ['validation']:
            
                # Create negatives for Context Relevance and Answer Faithfulness/Relevance
                dataset_1 = dataset.copy()
                dataset_2 = dataset.copy()
                negative_documents = []
                negative_answers = []
                for row in range(len(dataset_1)):

                    random_int = random.randint(0, len(total_documents) - 1)
                    while total_documents[random_int] == dataset_1.iloc[row]['Document']:
                        random_int = random.randint(0, len(total_documents) - 1)
                    negative_documents.append(total_documents[random_int])

                    if dataset_chosen == "record":
                        alternate_entities = dataset_1.iloc[row]['entities']
                        assert type(alternate_entities) == list 
                        current_answers = dataset_1.iloc[row]['Original_Answer_Lists']
                        alternate_entities = [entity for entity in alternate_entities if entity not in current_answers]
                        negative_answers.append(random.choice(alternate_entities))
                    elif dataset_chosen == "multirc":
                        random_int = random.randint(0, len(total_answers) - 1)
                        while total_answers[random_int] == dataset_1.iloc[row]['Answer']:
                            random_int = random.randint(0, len(total_answers))
                        negative_answers.append(total_answers[random_int] )

                ####################################################################

                dataset_1['Document'] = negative_documents
                dataset_1['Context_Relevance_Label'] = [0 for _ in range(len(dataset_1))] 

                dataset_2['Answer'] = negative_answers
                dataset_2['Answer_Faithfulness_Label'] = [0 for _ in range(len(dataset_2))] 
                dataset_2['Answer_Relevance_Label'] = [0 for _ in range(len(dataset_2))] 

                total_filepaths = []
                for ratio in positive_negative_ratios:

                    negatives_to_add = (1 - ratio) / ratio
                    negatives_to_add = int(negatives_to_add * len(dataset_1))

                    kilt_dataset_combined = pd.concat([dataset, dataset_1[:negatives_to_add], dataset_2[:negatives_to_add]], axis=0, ignore_index=True)
                    kilt_dataset_combined = kilt_dataset_combined.sample(n=len(kilt_dataset_combined), random_state=42) #Shuffled

                    print("Positive - Negative Ratio")
                    print(str(ratio))
                    print(len(dataset) / (len(dataset) + len(dataset_1[:negatives_to_add])))

                    file_path = folder_path + "multirc_" + "ratio_" + str(ratio) + ".tsv"
                    kilt_dataset_combined.to_csv(file_path, sep="\t", index=False)
                    print("Saved file to: " + file_path)
                    print("-------------------------------------------------------")
                    total_filepaths.append(file_path)

                ####################################################################

                print("Total Filepaths for Evaluation:")
                print(str(total_filepaths))

                kilt_dataset_combined = pd.concat([dataset, dataset_1, dataset_2], axis=0, ignore_index=True)
                kilt_dataset_combined = kilt_dataset_combined.sample(n=len(kilt_dataset_combined), random_state=42) #Shuffled

                file_path = folder_path + "multirc_" + "ratio_" + str(ratio) + ".tsv"
                kilt_dataset_combined.to_csv(file_path, sep="\t", index=False)
                print("Saved file to: " + file_path)
                print("-------------------------------------------------------")






