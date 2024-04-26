
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM
from vllm import LLM
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import requests 
import os
import re
import time
import anthropic
import sys
import re
import warnings

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

def few_shot_context_relevance_scoring(system_prompt: str, query: str, document: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):
    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
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
            response = openai.chat.completions.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)

            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
    
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
        #    print("Error with querying OpenAI! Trying again...")
        #    time.sleep(60)
    
####################################################################

def few_shot_answer_faithfulness_scoring(system_prompt, query: str, document: str, answer: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + str(answer) + "\n"
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
            response = openai.chat.completions.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)

            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
    
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)
    
####################################################################

def few_shot_answer_relevance_scoring(system_prompt: str, query: str, document: str, answer: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + str(answer) + "\n"
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
            response = openai.chat.completions.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)


            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
    
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)


##############################TOGETHERAI CUSTOM MODELS (NOT OPENAI)######################################
            
def few_shot_context_relevance_scoring_togetherai(system_prompt: str, query: str, document: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Label: "

    time.sleep(4)

    client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1',
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "system",
        "content": system_prompt,
        },
        {
        "role": "user",
        "content": user_prompt,
        }
    ],
    model=model_choice
    )
    final_response = chat_completion.choices[0].message.content

    if debug_mode is True: 
        print(final_response)

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"
    
    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1

def few_shot_answer_faithfulness_scoring_togetherai(system_prompt: str, query: str, document: str, answer: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
                user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                if current_label == 1 or current_label == 1.0:
                    warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                    current_label = "[[Yes]]"
                elif current_label == 0 or current_label == 0.0:
                    warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                    current_label = "[[No]]"
                user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Answer: " + answer + "\n"
    user_prompt += "Label: "

    time.sleep(4)

    client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1',
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "system",
        "content": system_prompt,
        },
        {
        "role": "user",
        "content": user_prompt,
        }
    ],
    model=model_choice
    )
    final_response = chat_completion.choices[0].message.content

    if debug_mode is True: 
        print(final_response)

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"
    
    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1

def few_shot_answer_relevance_scoring_togetherai(system_prompt: str, query: str, document: str, answer: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
            user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
            user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
            user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
            current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
            if current_label == 1 or current_label == 1.0:
                warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                current_label = "[[Yes]]"
            elif current_label == 0 or current_label == 0.0:
                warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                current_label = "[[No]]"
            user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Answer: " + answer + "\n"
    user_prompt += "Label: "

    time.sleep(4)

    client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1',
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
        "role": "system",
        "content": system_prompt,
        },
        {
        "role": "user",
        "content": user_prompt,
        }
    ],
    model=model_choice
    )
    final_response = chat_completion.choices[0].message.content
    
    if debug_mode is True: 
        print(final_response)

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"
    
    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1


############################## ANTHROPIC CUSTOM MODELS (NOT OPENAI) ######################################

            
def few_shot_context_relevance_scoring_claude(system_prompt: str, query: str, document: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Label: "

    time.sleep(2)

    client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    )

    chat_completion = client.messages.create(
        model=model_choice,
        max_tokens=1024,
        messages=[
            {
            "role": "user",
            "content": user_prompt,
            }, 
            {
            "role": "assistant",
            "content": system_prompt,
            }
        ],
    )
    if chat_completion.content:  # This checks if the list is not empty
        final_response = chat_completion.content[0].text
        if debug_mode is True: 
            print(final_response)

    else:
        return -1

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"

    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # if "[[Yes]]" in final_response:
    #     # print(final_response)
    #     return 1
    # elif "[[No]]" in final_response:
    #     # print(final_response)
    #     return 0
    # else:
    #     print("Didn't extract Yes or No!")
    #     # print(f"Unexpected response received: {final_response}")
    #     return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1

def few_shot_answer_faithfulness_scoring_claude(system_prompt: str, query: str, document: str, answer: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
                user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                if current_label == 1 or current_label == 1.0:
                    warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                    current_label = "[[Yes]]"
                elif current_label == 0 or current_label == 0.0:
                    warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                    current_label = "[[No]]"
                user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Answer: " + answer + "\n"
    user_prompt += "Label: "

    time.sleep(2)

    client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    )

    chat_completion = client.messages.create(
        model=model_choice,
        max_tokens=1024,
        messages=[
            {
            "role": "user",
            "content": user_prompt,
            }, 
            {
            "role": "assistant",
            "content": system_prompt,
            }
        ],
    )
    if chat_completion.content:  # This checks if the list is not empty
        final_response = chat_completion.content[0].text
        if debug_mode is True: 
            print(final_response)
    else:
        return -1

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"

    
    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # if "[[Yes]]" in final_response:
    #     # print(final_response)
    #     return 1
    # elif "[[No]]" in final_response:
    #     # print(final_response)
    #     return 0
    # else:
    #     print("Didn't extract Yes or No!")
    #     # print(f"Unexpected response received: {final_response}")
    #     return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1

def few_shot_answer_relevance_scoring_claude(system_prompt: str, query: str, document: str, answer: str, model_choice, query_id: str, debug_mode: bool, few_shot_examples=None):
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    user_prompt = ""
    if few_shot_examples is not None:
        for row in range(len(few_shot_examples)):
            user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
            user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
            user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
            current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
            if current_label == 1 or current_label == 1.0:
                warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                current_label = "[[Yes]]"
            elif current_label == 0 or current_label == 0.0:
                warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                current_label = "[[No]]"
            user_prompt += "Label: " + str(current_label) + "\n\n"

    user_prompt += "Question: " + query + "\n"
    user_prompt += "Document: " + document + "\n"
    user_prompt += "Answer: " + answer + "\n"
    user_prompt += "Label: "

    time.sleep(2)

    client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    )

    chat_completion = client.messages.create(
        model=model_choice,
        max_tokens=1024,
        messages=[
            {
            "role": "user",
            "content": user_prompt,
            }, 
            {
            "role": "assistant",
            "content": system_prompt,
            }
        ],
    )
    if chat_completion.content:  # This checks if the list is not empty
        final_response = chat_completion.content[0].text
        if debug_mode is True: 
            print(final_response)
    else:
        return -1

    yes = r"\[\[Yes]]"
    no = r"\[\[No]]"

    
    if re.search(yes, final_response): 
        return 1
    elif re.search(no, final_response):
        return 0
    else:
        print("Didn't extract Yes or No!")
        # print(f"Unexpected response received: {final_response}")
        return -1
    # if "[[Yes]]" in final_response:
    #     # print(final_response)
    #     return 1
    # elif "[[No]]" in final_response:
    #     # print(final_response)
    #     return 0
    # else:
    #     print("Didn't extract Yes or No!")
    #     # print(f"Unexpected response received: {final_response}")
    #     return -1
    # except requests.exceptions.RequestException as e:
    #     print(f"HTTP Request failed: {e}")
    #     return -1

def few_shot_context_relevance_scoring_vllm(system_prompt: str, query: str, document: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # openai_api_base = "http://localhost:8000/v1"

    # if not OPENAI_API_KEY:
    #     sys.exit("OPENAI_API_KEY environment variable not set. Please set the variable.")

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Label: "

            # client = OpenAI(api_key=OPENAI_API_KEY, base_url=openai_api_base)
            
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
            
           
            # final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)


            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
            
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
        #    print("Error with querying OpenAI! Trying again...")
        #    time.sleep(60)
    
####################################################################

def few_shot_answer_faithfulness_scoring_vllm(system_prompt, query: str, document: str, answer: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_api_base = "http://localhost:8000/v1"

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY environment variable not set. Please set the variable.")

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + str(answer) + "\n"
            user_prompt += "Label: "

            client = OpenAI(api_key=OPENAI_API_KEY, base_url=openai_api_base)
            
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
            response = client.chat.completions.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)

            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
            
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)
    
####################################################################

def few_shot_answer_relevance_scoring_vllm(system_prompt: str, query: str, document: str, answer: str, gpt_model: str, query_id: str, debug_mode: bool, few_shot_examples=None):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_api_base = "http://localhost:8000/v1"

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY environment variable not set. Please set the variable.")

    for _ in range(5):
        #try:

            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += "Question: " + few_shot_examples.iloc[row][query_id] + "\n"
                    user_prompt += "Document: " + few_shot_examples.iloc[row]['Document'] + "\n"
                    user_prompt += "Answer: " + few_shot_examples.iloc[row]['Answer'] + "\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    if current_label == 1 or current_label == 1.0:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label == 0 or current_label == 0.0:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

            user_prompt += "Question: " + query + "\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Answer: " + str(answer) + "\n"
            user_prompt += "Label: "

            client = OpenAI(api_key=OPENAI_API_KEY, base_url=openai_api_base)
            
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
            response = client.chat.completions.create(
                                model=gpt_model,
                                messages=messages
                            )    
            final_response = response.choices[0].message.content

            if debug_mode is True: 
                print(final_response)


            yes = r"\[\[Yes]]"
            no = r"\[\[No]]"
            
            if re.search(yes, final_response): 
                return 1
            elif re.search(no, final_response):
                return 0
            else:
                print("Didn't extract Yes or No!")
                # print(f"Unexpected response received: {final_response}")
                return -1
        #except:
            print("Error with querying OpenAI! Trying again...")
            time.sleep(60)