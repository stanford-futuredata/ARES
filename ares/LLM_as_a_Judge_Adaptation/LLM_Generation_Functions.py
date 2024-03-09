
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import re
import random
import math

#################################################################

def generate_synthetic_query_llm_approach(document: str, prompt: str, length_of_fewshot_prompt: int, device, tokenizer, model, percentiles, for_fever_dataset=False, for_wow_dataset=False):

    synthetic_queries = []

    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) +":\n"
    if for_fever_dataset:
        prompt_without_document += "Document: \nStatement: "
    elif for_wow_dataset:
        prompt_without_document += "Document: \nDialogue: "
    else:
        prompt_without_document += "Document: \nQuestion: "
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors='pt').to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors='pt').to(device).shape[1]

    if prompt_tokens_length + document_length + 100 >= 2048:
        # Added buffer for truncation
        encoded_input = tokenizer(document, max_length=2048 - prompt_tokens_length - 100, truncation=True, return_tensors='pt')
        truncated_document = tokenizer.decode(encoded_input['input_ids'][0][:2048 - prompt_tokens_length - 100]) 
        document = truncated_document.replace("</s>", "")

    prompt += "Example " + str(length_of_fewshot_prompt + 1) +":\n"
    prompt += "Document: " + document + "\n"
    if for_fever_dataset:
        prompt += "Statement: "
    elif for_wow_dataset:
        prompt += "Dialogue: "
    else:
        prompt += "Question: "

    input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

    max_length = 32
    if for_wow_dataset:
        max_length = 256

    for percentile in percentiles:

        if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
            print("Length of problematic input ids: " + str(input_ids.shape))
            print("Length of problematic document: " + str(len(encoded_input['input_ids'][0])))
            assert False
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=percentile,
            num_return_sequences=1)

        query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        synthetic_queries.append(query)

    return synthetic_queries

#################################################################

def generate_answer_llm_approach(document: str, question: str, prompt: str, length_of_fewshot_prompt: int, device, tokenizer, model, for_fever_dataset=False, for_wow_dataset=False):

    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) +":\n"
    if for_fever_dataset:
        prompt_without_document += "Document: \nStatement: \nAnswer: "
    elif for_wow_dataset:
        prompt_without_document += "Document: \nDialogue: \nResponse: "
    else:
        prompt_without_document += "Document: \nQuestion: \nAnswer: "
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors='pt').to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors='pt').to(device).shape[1]
    question_length = tokenizer.encode(question, return_tensors='pt').to(device).shape[1]

    if prompt_tokens_length + document_length + question_length + 100 >= 2048:
        # Added buffer for truncation
        reduction_length = prompt_tokens_length + question_length + 100
        encoded_input = tokenizer(document, max_length=2048 - reduction_length, truncation=True, return_tensors='pt')
        truncated_document = tokenizer.decode(encoded_input['input_ids'][0][:2048 - reduction_length]) 
        document = truncated_document.replace("</s>", "")

    prompt += "Example " + str(length_of_fewshot_prompt + 1) +":\n"
    prompt += "Document: " + document + "\n"
    if for_fever_dataset:
        prompt += "Statement: " + question + "\n"
        prompt += "Answer: " 
    elif for_wow_dataset:
        prompt += "Dialogue: " + question + "\n"
        prompt += "Response: " 
    else:
        prompt += "Question: " + question + "\n"
        prompt += "Answer: " 

    input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

    if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
        print("Length of problematic input ids: " + str(input_ids.shape))
        print("Length of problematic document: " + str(len(encoded_input['input_ids'][0])))
        assert False
    outputs = model.generate(
        input_ids=input_ids,
        max_length=256,
        do_sample=True,
        top_p=0.05,
        num_return_sequences=1)

    query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return query

#################################################################

def generate_synthetic_query_openai_approach(document: str, system_prompt: str, few_shot_examples: str, temperatures: list, length_of_fewshot_prompt: int):

    time.sleep(1)

    synth_documents_generated = []
    #try:
    for temp in temperatures:
        try:
            user_prompt = few_shot_examples
            #user_prompt += "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
            user_prompt += "Document: " + document + "\n"
            user_prompt += "Question: "

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
                    model="gpt-3.5-turbo-16k",
                    #model="gpt-4",
                    messages=messages,
                    temperature=temp
                )
                
            final_response = response["choices"][0]["message"]["content"]
            synth_documents_generated.append(final_response)
        except Exception as E:
            print("Error with OpenAI! Waiting one minute...")
            print("Error: " + str(E))
            time.sleep(30)
        #return final_response
    #except:
    #    print("Error Querying OpenAI! Attempting again...")

    assert len(synth_documents_generated) >= 1
    return synth_documents_generated

#################################################

def generate_answer_from_context(document: str, synth_question: str):

    time.sleep(1)

    for _ in range(5):

        try:

            system_prompt = "You are a helpful assistant built by Databricks, you are not human, you are good at helping to answer a query based on the context step by step, the context is a document. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context as the answer. If you don't know the answer, just say I don't know. If there is no enough information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know."
            user_prompt = f"Here is the question {synth_question}\nHere is the context: {(document)}?"

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
                        model="gpt-3.5-turbo-16k",
                        messages=messages,
                        temperature=0.0
                    )
                    
            final_response = response["choices"][0]["message"]["content"]
            return final_response

        except:

            print("Error querying OpenAI! Attempting again...")

#################################################

def generate_contradictory_answer_from_context(document: str, synth_question: str):

    time.sleep(1)

    for _ in range(5):

        try:

            #system_prompt = "You are a helpful assistant built by Databricks, you are not human, you are good at helping to answer a query based on the context step by step, the context is a document. If the query doesn't form a complete question, just say I don't know. If there is a good answer from the context, try to summarize the context as the answer. If you don't know the answer, just say I don't know. If there is no enough information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know."
            system_prompt = "Create an answer for the given question that contradicts the provided document. You should create false information that disagrees with what exists within the content of the document."
            user_prompt = f"Question: {synth_question}\nDocument:{(document)}"
            user_prompt = system_prompt + "\n\n" + user_prompt

            messages = [
                    #{
                    #    "role": "system",
                    #    "content": system_prompt
                    #},
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]

            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        messages=messages
                    )
                    
            final_response = response["choices"][0]["message"]["content"]
            return final_response

        except:

            print("Error querying OpenAI! Attempting again...")

#################################################

def check_generated_answer(answer: str):
    problematic_phrases = ["I don't know", "don't know", "i don't know"]
    for phrase in problematic_phrases:
        if phrase in answer:
            return "No"
    return "Yes"

def generate_contradictory_answer_examples(queries_dataset, number_of_contradictory_answers_to_generate: int, few_shot_examples_for_contradictory_answers=None, 
                                           device=None, tokenizer=None, model=None, for_fever_dataset=None, for_wow_dataset=None):

    def remove_problematic_contradictory_phrases(text):
        if text is None:
            return text

        problematic_phrases = ["Contradictory Answer:", "The false information created is:", "Incorrect Answer: "]
        text_split = text.split(":")
        if len(text_split) > 1:
            return text_split[1]
        else:
            return text

    queries_dataset_copy = queries_dataset.copy()
    #queries_dataset_copy = queries_dataset_copy[queries_dataset_copy['Context_Relevance_Label'] == "Yes"]
    queries_dataset_copy = queries_dataset_copy.drop_duplicates(subset=['synthetic_query'])

    number_of_contradictory_answers_to_generate = min(number_of_contradictory_answers_to_generate, len(queries_dataset_copy))
    queries_dataset_copy = queries_dataset_copy.sample(n=number_of_contradictory_answers_to_generate, random_state=42)

    contradictory_answers = []
    contradictory_labels = []
    for i in tqdm(range(len(queries_dataset_copy))):

        if few_shot_examples_for_contradictory_answers is None:
            contradictory_answer_generated = generate_contradictory_answer_from_context(queries_dataset_copy.iloc[i]['document'], queries_dataset_copy.iloc[i]['synthetic_query'])
        else:
            contradictory_answer_generated = generate_contradictory_answer_llm_approach(queries_dataset_copy.iloc[i]['document'], queries_dataset_copy.iloc[i]['synthetic_query'], few_shot_examples_for_contradictory_answers, device, tokenizer, model, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset)

        contradictory_answer_generated = remove_problematic_contradictory_phrases(contradictory_answer_generated)

        contradictory_answers.append(contradictory_answer_generated)   
        contradictory_labels.append("No")

    queries_dataset_copy['generated_answer'] = contradictory_answers
    queries_dataset_copy['Answer_Faithfulness_Label'] = contradictory_labels
    queries_dataset_copy['generated_answer'] = contradictory_answers
    queries_dataset_copy['Answer_Relevance_Label'] = contradictory_labels

    print("Contradictory Answers Added using Contradiction Generation")
    print(len(queries_dataset_copy))

    #################################################

    queries_dataset_copy_2 = queries_dataset.copy()
    queries_dataset_copy_2 = queries_dataset_copy_2.drop_duplicates(subset=['synthetic_query'])

    queries_dataset_copy_2 = queries_dataset_copy_2.sample(n=number_of_contradictory_answers_to_generate, random_state=42)
    total_answers = queries_dataset[queries_dataset['Answer_Relevance_Label'] == "Yes"]['generated_answer'].tolist()
    total_answers = [answer for answer in total_answers if isinstance(answer, str)]
    total_answers = [str(answer) for answer in total_answers]
    total_answers = [answer for answer in total_answers if len(answer) > 5]

    contradictory_answers_2 = []
    contradictory_labels_2 = []
    for i in tqdm(range(len(queries_dataset_copy_2))):

        random_answer = random.choice(total_answers)
        contradictory_answers_2.append(random_answer)   
        contradictory_labels_2.append("No")

    queries_dataset_copy_2['generated_answer'] = contradictory_answers_2
    queries_dataset_copy_2['Answer_Relevance_Label'] = contradictory_labels_2

    print("Contradictory Answers Added using Answer Randomization")
    print(len(queries_dataset_copy_2))

    #################################################

    # Shuffle dataframe
    queries_dataset = pd.concat([queries_dataset, queries_dataset_copy, queries_dataset_copy_2], axis=0, ignore_index=True)
    queries_dataset = queries_dataset.sample(n=len(queries_dataset), random_state=42)

    # contradictory_answers_df = queries_dataset[queries_dataset['Answer_Relevance_Label'] == "No"]

    # # Print each contradictory answer with its query
    # for index, row in contradictory_answers_df.iterrows():
    #     print(f"Query: {row['synthetic_query']}, Contradictory Answer: {row['generated_answer']}")

    # breakpoint()

    return queries_dataset

#################################################

def generate_contradictory_answer_llm_approach(document: str, question: str, prompt: str, device, tokenizer, model, for_fever_dataset=False, for_wow_dataset=False):

    prompt_without_document = prompt + "Example " + str(prompt.count("Example") + 1) +":\n"
    if for_fever_dataset:
        prompt_without_document += "Document: \nStatement: \nIncorrect Answer: "
    elif for_wow_dataset:
        prompt_without_document += "Document: \nDialogue: \nIncorrect Response: "
    else:
        prompt_without_document += "Document: \nQuestion: \nIncorrect Answer: "
    prompt_tokens_length = tokenizer.encode(prompt_without_document, return_tensors='pt').to(device).shape[1]
    document_length = tokenizer.encode(document, return_tensors='pt').to(device).shape[1]
    question_length = tokenizer.encode(question, return_tensors='pt').to(device).shape[1]

    if prompt_tokens_length + document_length + question_length + 100 >= 2048:
        # Added buffer for truncation
        reduction_length = prompt_tokens_length + question_length + 100
        encoded_input = tokenizer(document, max_length=2048 - reduction_length, truncation=True, return_tensors='pt')
        truncated_document = tokenizer.decode(encoded_input['input_ids'][0][:2048 - reduction_length]) 
        document = truncated_document.replace("</s>", "")

    prompt += "Example " + str(prompt.count("Example") + 1) +":\n"
    prompt += "Document: " + document + "\n"
    if for_fever_dataset:
        prompt += "Statement: " + question + "\n"
        prompt += "Incorrect Answer: " 
    elif for_wow_dataset:
        prompt += "Dialogue: " + question + "\n"
        prompt += "Incorrect Response: " 
    else:
        prompt += "Question: " + question + "\n"
        prompt += "Incorrect Answer: " 

    input_ids = tokenizer.encode(prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)

    #print("New input ids")
    #print(input_ids.shape)
    #assert False

    if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
        print("Length of problematic input ids: " + str(input_ids.shape))
        print("Length of problematic document: " + str(len(encoded_input['input_ids'][0])))
        assert False
    outputs = model.generate(
        input_ids=input_ids,
        max_length=256,
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1)

    query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return query

