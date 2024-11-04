import ast
import copy
import csv
import json
import os
import re
import sys
import time
import warnings
from typing import Dict

import anthropic
import numpy as np
import pandas as pd
import requests
import torch
import openai
from together import Together
from datasets import Dataset
from openai import OpenAI
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not imported.")

def calculate_accuracy(predictions, ground_truth) -> float:
    """
    Calculate the accuracy percentage between predictions and ground truths.

    Args:
    predictions (list): A list of predicted values.
    ground_truth (list): A list of actual values.

    Returns:
    float: The accuracy percentage rounded to two decimal places.

    Raises:
    ValueError: If the input lists have different lengths.
    """
    if len(predictions) != len(ground_truth):
        print(f"Predictions count: {len(predictions)}")
        print(f"Ground truth count: {len(ground_truth)}")
        raise ValueError("Input lists must have the same length")

    correct_count = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
    total_count = len(predictions)

    accuracy = round(correct_count * 100 / total_count, 2)
    return accuracy

def few_shot_context_relevance_scoring(
    system_prompt: str, query: str, document: str, gpt_model: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of a query given a document using few-shot examples.
    This function constructs a user prompt with few-shot examples (if provided) and the current query,
    document, and answer. It then queries an OpenAI model to determine if the query is context relevant to the
    document.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query to be evaluated.
    document (str): The document to be evaluated.
    gpt_model (str): The model identifier for OpenAI's GPT.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating context relevance (1 for relevant, 0 for not relevant).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += f"Question: {query}\nDocument: {document}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Context Relevance Evaluation:")
                print(user_prompt)

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

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"

            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_faithfulness_scoring(
    system_prompt: str, query: str, document: str, answer: str, gpt_model: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the faithfulness of an answer given a document and a query using few-shot examples.
    This function constructs a user prompt with few-shot examples (if provided) and the current query,
    document, and answer. It then queries an OpenAI model to determine if the answer is faithful.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for faithfulness.
    gpt_model (str): The model identifier for OpenAI's GPT.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer faithfulness 1 if the answer is faithful, 0 otherwise.

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    for attempt in range(5): 
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay) # Model query delay chosen by user
            
            if debug_mode: 
                print("------------------------------------------")
                print("Answer Faithfulness Evaluation")
                print(user_prompt)

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

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"

            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_relevance_scoring(
    system_prompt: str, query: str, document: str, answer: str, gpt_model: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of an answer given a document and a query using few-shot examples.
    This function constructs a user prompt with few-shot examples (if provided) and the current query,
    document, and answer. It then queries an OpenAI model to determine if the answer is faithful.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for relevance.
    gpt_model (str): The model identifier for OpenAI's GPT.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer relevance (1 for relevant, 0 for not relevant).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    for attempt in range(5): 
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay)
            
            if debug_mode: 
                print("------------------------------------------")
                print("Answer Relevance Evaluation:")
                print(user_prompt)

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

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"

            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_context_relevance_scoring_togetherai(
    system_prompt: str, query: str, document: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of a query given a document using few-shot examples with a specified model from TogetherAI.
    This function constructs a user prompt with optional few-shot examples and the current query and document.
    It then queries a TogetherAI model to determine if the query is contextually relevant to the document.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    model_choice (str): The model identifier for TogetherAI's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating context relevance (1 for relevant, 0 for not relevant).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")
    
    for attempt in range(5): 
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += f"Label: {current_label}\n\n"

            user_prompt += f"Question: {query}\nDocument: {document}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("Context Relevance Evaluation:")
                print(user_prompt)

            client = Together(api_key=TOGETHER_API_KEY)

            # Enable streaming to receive responses as they are generated, reducing wait times.
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_choice,
                stream=True
            )

            responses = [chunk.choices[0].delta.content for chunk in chat_completion]
            final_response = " ".join(responses)

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"
            
            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_faithfulness_scoring_togetherai(
    system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the faithfulness of an answer given a document using few-shot examples with a specified model from TogetherAI.
    This function constructs a user prompt with optional few-shot examples and the current query, document, and answer.
    It then queries a TogetherAI model to determine if the answer is faithful.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query to be evaluated. 
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for faithfulness.
    model_choice: The model identifier for TogetherAI's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer faithfulness (1 for faithful, 0 for not faithful).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")
    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            

            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Answer Faithfulness Evaluation:")
                print(user_prompt)

            client = Together(api_key=TOGETHER_API_KEY)

            # Enable streaming to receive responses as they are generated, reducing wait times.
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model_choice,
                stream=True
            )

            responses = []
            for chunk in chat_completion:
                responses.append(chunk.choices[0].delta.content)

            final_response = " ".join(responses)

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"

            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_relevance_scoring_togetherai(
    system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of an answer given a document using few-shot examples with a specified model from TogetherAI.
    This function constructs a user prompt with optional few-shot examples and the current query, document, and answer.
    It then queries a TogetherAI model to determine if the answer is relevant.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query to be evaluated. 
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for relevance.
    model_choice: The model identifier for TogetherAI's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer relevance (1 for faithful, 0 for not faithful).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        sys.exit("TOGETHER_API_KEY environment variable not set. Please set the variable.")
    
    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Answer Relevance Evaluation:")
                print(user_prompt)

            client = Together(api_key=TOGETHER_API_KEY)

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_choice,
                stream=True
            )

            responses = []
            
            for chunk in chat_completion:
                responses.append(chunk.choices[0].delta.content)

            final_response = " ".join(responses)
            
            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"
            
            if re.search(yes, final_response): 
                if debug_mode:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0
            
def few_shot_context_relevance_scoring_claude(
    system_prompt: str, query: str, document: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of a query given a document using few-shot examples with a specified model from TogetherAI.
    This function constructs a user prompt with optional few-shot examples and the current query and document.
    It then queries an Anthropic model to determine if the query is contextually relevant to the document.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    model_choice (str): The model identifier for Anthropic's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating context relevance (1 for relevant, 0 for not relevant).

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"
            
            user_prompt += f"Question: {query}\nDocument: {document}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Context Relevance Evaluation:")
                print(user_prompt)

            client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            )

            responses = []

            with client.messages.stream( 
                max_tokens=1024, 
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": user_prompt.strip()}], 
                model=model_choice
            ) as stream: 
                for text in stream.text_stream: 
                    responses.append(text)

            final_response = " ".join(responses)

            if debug_mode is True: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"

            if re.search(yes, final_response): 
                if debug_mode is True:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode is True:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_faithfulness_scoring_claude(
    system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the faithfulness of an answer given a document and a query using few-shot examples.
    This function constructs a user prompt with few-shot examples (if provided) and the current query,
    document, and answer. It then queries an Anthropic model to determine if the answer is faithful.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for faithfulness.
    model_choice (str): The model identifier for Anthropic's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer faithfulness 1 if the answer is faithful, 0 otherwise.

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Answer Faithfulness Evaluation")
                print(user_prompt)

            client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            )

            responses = []

            with client.messages.stream( 
                max_tokens=1024, 
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": user_prompt.strip()}], 
                model=model_choice
            ) as stream: 
                for text in stream.text_stream: 
                    responses.append(text)

            final_response = " ".join(responses)

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"
            
            if re.search(yes, final_response): 
                if debug_mode is True:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode is True:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0

def few_shot_answer_relevance_scoring_claude(
    system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
    query_id: str, debug_mode: bool, request_delay: int, 
    failed_extraction_count: Dict[str,int] = {'failed': 0}, 
    few_shot_examples=None
) -> int:
    """
    Evaluates the relevance of an answer given a document and a query using few-shot examples.
    This function constructs a user prompt with few-shot examples (if provided) and the current query,
    document, and answer. It then queries an Anthropic model to determine if the answer is relevant.

    Args:
    system_prompt (str): The prompt for the system role in the conversation.
    query (str): The query related to the document.
    document (str): The document to be evaluated.
    answer (str): The answer to be evaluated for relevance.
    model_choice (str): The model identifier for an Anthropic's API.
    query_id (str): The identifier for the query column in few_shot_examples.
    debug_mode (bool): Flag to turn on debug mode for additional output.
    request_delay (int): Time in seconds to delay the request (simulating network latency).
    failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
    few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

    Returns:
    int: The label indicating answer relevance 1 if the answer is relevant, 0 otherwise.

    Raises:
    Warning: If incorrect labels are detected in the few-shot examples.
    """
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY environment variable not set. Please set the variable.")

    for attempt in range(5):
        try:
            user_prompt = ""
            if few_shot_examples is not None:
                for row in range(len(few_shot_examples)):
                    user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                    user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                    user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                    current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                    if current_label in {1, 1.0}:
                        warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                        current_label = "[[Yes]]"
                    elif current_label in {0, 0.0}:
                        warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                        current_label = "[[No]]"
                    user_prompt += "Label: " + str(current_label) + "\n\n"

            user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

            time.sleep(request_delay)

            if debug_mode: 
                print("------------------------------------------")
                print("Answer Relevance Evaluation")
                print(user_prompt)

            client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            )

            responses = []

            with client.messages.stream( 
                max_tokens=1024, 
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": user_prompt.strip()}], 
                model=model_choice
            ) as stream: 
                for text in stream.text_stream: 
                    responses.append(text)

            final_response = " ".join(responses)

            if debug_mode: 
                print(final_response)

            yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
            no = r"\[\s*\[?\s*No\s*\]?\s*\]"
            
            if re.search(yes, final_response): 
                if debug_mode is True:
                    print("Returned label 1")
                return 1
            elif re.search(no, final_response):
                if debug_mode is True:
                    print("Returned label 0")
                return 0
            else:
                print("Didn't extract Yes or No!")
                failed_extraction_count['failed'] += 1
                return 0
        except Exception as e:
            if attempt < 4:  # Only print the error message if not on the last attempt
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(60)  # Sleep for 60 seconds before retrying
            else:
                print("All attempts failed. Last error was:", e)
                return 0
            
# Define the no-op functions
def no_op_function(*args, **kwargs):
    print("vLLM is not available. This function call is ignored.")

if VLLM_AVAILABLE:
    def few_shot_context_relevance_scoring_vllm(
        system_prompt: str, query: str, document: str, model_choice: str, 
        query_id: str, debug_mode: bool, host_url: str, request_delay: int, 
        failed_extraction_count: Dict[str,int] = {'failed': 0}, 
        few_shot_examples=None
    ) -> int:
        """
        Evaluates the relevance of a query given a document using few-shot examples.
        This function constructs a user prompt with few-shot examples (if provided) and the current query,
        document, and answer. It then queries a local vLLM model to determine if the query is context relevant to the
        document.

        Args:
        system_prompt (str): The prompt for the system role in the conversation.
        query (str): The query to be evaluated.
        document (str): The document to be evaluated.
        model_choice (str): The model identifier for vLLM's model.
        query_id (str): The identifier for the query column in few_shot_examples.
        debug_mode (bool): Flag to turn on debug mode for additional output.
        host_url (str): The URL is used to send requests to the locally running model.
        request_delay (int): Time in seconds to delay the request (simulating network latency).
        failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
        few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

        Returns:
        int: The label indicating context relevance (1 for relevant, 0 for not relevant).

        Raises:
        Warning: If incorrect labels are detected in the few-shot examples.
        """
        for attempt in range(5):
            try:
                user_prompt = ""
                if few_shot_examples is not None:
                    for row in range(len(few_shot_examples)):
                        user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                        user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                        current_label = few_shot_examples.iloc[row]['Context_Relevance_Label']
                        if current_label in {1, 1.0}:
                            warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                            current_label = "[[Yes]]"
                        elif current_label in {0, 0.0}:
                            warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                            current_label = "[[No]]"
                        user_prompt += "Label: " + str(current_label) + "\n\n"
                
                user_prompt += f"Question: {query}\nDocument: {document}\nLabel: "

                time.sleep(request_delay)

                if debug_mode: 
                    print("------------------------------------------")
                    print("Context Relevance Evaluation:")
                    print(user_prompt)
                
                openai_api_key = "EMPTY"
                client = OpenAI(
                api_key=openai_api_key,
                base_url=host_url
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

                if debug_mode: 
                    print(final_response)

                yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
                no = r"\[\s*\[?\s*No\s*\]?\s*\]"

                if re.search(yes, final_response): 
                    if debug_mode:
                        print("Returned label 1")
                    return 1
                elif re.search(no, final_response):
                    if debug_mode:
                        print("Returned label 0")
                    return 0
                else:
                    print("Didn't extract Yes or No!")
                    failed_extraction_count['failed'] += 1
                    return 0
            except Exception as e:
                if attempt < 4:  # Only print the error message if not on the last attempt
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(60)  # Sleep for 60 seconds before retrying
                else:
                    print("All attempts failed. Last error was:", e)
                    return 0
        
    def few_shot_answer_faithfulness_scoring_vllm(
        system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
        query_id: str, debug_mode: bool, host_url: str, request_delay: int, 
        failed_extraction_count: Dict[str,int] = {'failed': 0}, 
        few_shot_examples=None
    ) -> int:
        """
        Evaluates the faithfulness of an answer given a document and a query using few-shot examples.
        This function constructs a user prompt with few-shot examples (if provided) and the current query,
        document, and answer. It then queries a local vLLM model to determine if the answer is faithful.

        Args:
        system_prompt (str): The prompt for the system role in the conversation.
        query (str): The query related to the document.
        document (str): The document to be evaluated.
        answer (str): The answer to be evaluated for faithfulness.
        model_choice (str): The model identifier for vLLM's model.
        query_id (str): The identifier for the query column in few_shot_examples.
        debug_mode (bool): Flag to turn on debug mode for additional output.
        host_url (str): The URL is used to send requests to the locally running model.
        request_delay (int): Time in seconds to delay the request (simulating network latency).
        failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
        few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

        Returns:
        int: The label indicating answer faithfulness 1 if the answer is faithful, 0 otherwise.

        Raises:
        Warning: If incorrect labels are detected in the few-shot examples.
        """
        for attempt in range(5):
            try:
                user_prompt = ""
                if few_shot_examples is not None:
                    for row in range(len(few_shot_examples)):
                        user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                        user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                        user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                        current_label = few_shot_examples.iloc[row]['Answer_Faithfulness_Label']
                        if current_label in {1, 1.0}:
                            warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                            current_label = "[[Yes]]"
                        elif current_label in {0, 0.0}:
                            warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                            current_label = "[[No]]"
                        user_prompt += "Label: " + str(current_label) + "\n\n"

                user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

                time.sleep(request_delay) # Model query delay chosen by user

                if debug_mode: 
                    print("------------------------------------------")
                    print("Answer Faithfulness Evaluation")
                    print(user_prompt)

                openai_api_key = "EMPTY"
                client = OpenAI(
                api_key=openai_api_key,
                base_url=host_url
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
                
                if debug_mode: 
                    print(final_response)

                yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
                no = r"\[\s*\[?\s*No\s*\]?\s*\]"
                
                if re.search(yes, final_response): 
                    if debug_mode:
                        print("Returned label 1")
                    return 1
                elif re.search(no, final_response):
                    if debug_mode:
                        print("Returned label 0")
                    return 0
                else:
                    print("Didn't extract Yes or No!")
                    failed_extraction_count['failed'] += 1
                    return 0
            except Exception as e:
                if attempt < 4:  # Only print the error message if not on the last attempt
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(60)  # Sleep for 60 seconds before retrying
                else:
                    print("All attempts failed. Last error was:", e)
                    return 0
            
    def few_shot_answer_relevance_scoring_vllm(
        system_prompt: str, query: str, document: str, answer: str, model_choice: str, 
        query_id: str, debug_mode: bool, host_url: str, request_delay: int, 
        failed_extraction_count: Dict[str,int] = {'failed': 0}, 
        few_shot_examples=None
    ) -> int:
        """
        Evaluates the relevance of an answer given a document and a query using few-shot examples.
        This function constructs a user prompt with few-shot examples (if provided) and the current query,
        document, and answer. It then queries a local vLLM model to determine if the answer is relevant.

        Args:
        system_prompt (str): The prompt for the system role in the conversation.
        query (str): The query related to the document.
        document (str): The document to be evaluated.
        answer (str): The answer to be evaluated for relevance.
        model_choice (str): The model identifier for vLLM's model.
        query_id (str): The identifier for the query column in few_shot_examples.
        debug_mode (bool): Flag to turn on debug mode for additional output.
        host_url (str): The URL is used to send requests to the locally running model.
        request_delay (int): Time in seconds to delay the request (simulating network latency).
        failed_extraction_count (Dict[str, int]): A dictionary to count the number of failed extractions.
        few_shot_examples (DataFrame, optional): A DataFrame containing few-shot examples.

        Returns:
        int: The label indicating answer relevance 1 if the answer is relevant, 0 otherwise.

        Raises:
        Warning: If incorrect labels are detected in the few-shot examples.
        """
        for attempt in range(5):
            try:
                user_prompt = ""
                if few_shot_examples is not None:
                    for row in range(len(few_shot_examples)):
                        user_prompt += f"Question: {few_shot_examples.iloc[row][query_id]}\n"
                        user_prompt += f"Document: {few_shot_examples.iloc[row]['Document']}\n"
                        user_prompt += f"Answer: {few_shot_examples.iloc[row]['Answer']}\n"
                        current_label = few_shot_examples.iloc[row]['Answer_Relevance_Label']
                        if current_label in {1, 1.0}:
                            warnings.warn("Incorrect label '1' detected. Please use '[[Yes]]' instead.")
                            current_label = "[[Yes]]"
                        elif current_label in {0, 0.0}:
                            warnings.warn("Incorrect label '0' detected. Please use '[[No]]' instead.")
                            current_label = "[[No]]"
                        user_prompt += "Label: " + str(current_label) + "\n\n"
                
                user_prompt += f"Question: {query}\nDocument: {document}\nAnswer: {answer}\nLabel: "

                time.sleep(request_delay) # Model query delay chosen by user

                if debug_mode: 
                    print("------------------------------------------")
                    print("Answer Relevance Evaluation")
                    print(user_prompt)

                openai_api_key = "EMPTY"
                client = OpenAI(
                api_key=openai_api_key,
                base_url=host_url
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

                yes = r"\[\s*\[?\s*Yes\s*\]?\s*\]"
                no = r"\[\s*\[?\s*No\s*\]?\s*\]"
                
                if re.search(yes, final_response): 
                    if debug_mode is True:
                        print("Returned label 1")
                    return 1
                elif re.search(no, final_response):
                    if debug_mode is True:
                        print("Returned label 0")
                    return 0
                else:
                    print("Didn't extract Yes or No!")
                    failed_extraction_count['failed'] += 1
                    return 0
            except Exception as e:
                if attempt < 4:  # Only print the error message if not on the last attempt
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(60)  # Sleep for 60 seconds before retrying
                else:
                    print("All attempts failed. Last error was:", e)
                    return 0
else:
    # Override the functions with the no-op function if vLLM is not available
    few_shot_context_relevance_scoring_vllm = no_op_function
    few_shot_answer_faithfulness_scoring_vllm = no_op_function
    few_shot_answer_relevance_scoring_vllm = no_op_function