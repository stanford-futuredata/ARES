
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import re

####################################################################

def calculate_accuracy(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        print(len(predictions))
        print(len(ground_truth))
        raise ValueError("Input lists must have the same length")

    correct_count = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
    total_count = len(predictions)

    accuracy = round(correct_count * 100 / total_count, 2)
    return accuracy

####################################################################

def few_shot_context_relevance_scoring(system_prompt: str, query: str, document: str, gpt_model: str, few_shot_examples=None):
    
    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row]['Query'] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Label: "
            
            messages = [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
            response = openai.ChatCompletion.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response["choices"][0]["message"]["content"]


            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                print("Didn't extract Yes or No!")
                return 1
        #except:
        #    print("Error with querying OpenAI! Trying again...")
        #    time.sleep(60)
    
####################################################################

def few_shot_answer_faithfulness_scoring(system_prompt, query: str, document: str, answer: str, gpt_model: str, few_shot_examples=None):

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row]['Query'] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + answer + "\n"
            user_prompt += "Label: "
            
            messages = [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
            response = openai.ChatCompletion.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response["choices"][0]["message"]["content"]

            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                print("Didn't extract Yes or No!")
                return 1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)
    
####################################################################

def few_shot_answer_relevance_scoring(system_prompt: str, query: str, document: str, answer: str, gpt_model: str, few_shot_examples=None):

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row]['Query'] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    user_prompt += "Label: " + str(current_label) + "\n\n"

            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + answer + "\n"
            user_prompt += "Label: "
            
            messages = [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
            response = openai.ChatCompletion.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response["choices"][0]["message"]["content"]


            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                print("Didn't extract Yes or No!")
                return 1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)


            