
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

from Evaluation_Functions import calculate_accuracy, few_shot_context_relevance_scoring
from Evaluation_Functions import few_shot_answer_faithfulness_scoring, few_shot_answer_relevance_scoring

####################################################################

def clean_document(document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")) #[:512]
    return cleaned_document

def clean_query(query: str):
    cleaned_query = query.replace("\n", " ").replace("\r"," ").replace("\t"," ").strip()
    return cleaned_query

#################################################

#evaluation_set_filepath = "../datasets_v2/LLM_Judge_Test_Set_V4_Reannotated.tsv"
#evaluation_set_filepath = "../datasets_v2/LLM_Judge_Test_Set_Human_Annotations.tsv"
#evaluation_set_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2.tsv"
#evaluation_set_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives.tsv"
#evaluation_set_filepath = "../datasets_v2/wow_reformatted_validation_with_negatives.tsv"
#evaluation_set_filepath = "../datasets_v2/fever_reformatted_validation_with_negatives.tsv"
#evaluation_set_filepath = "../datasets_v2/qa_logs_with_logging_details.tsv"
#evaluation_set_filepath = "../datasets_v2/LLM_Judge_Test_Set_Human_Annotations_V1.1_Filtered.tsv"
evaluation_set_filepath = "../datasets_v2/nq_reformatted_full_articles_True_validation_with_negatives.tsv"

gpt_model = "gpt-3.5-turbo-16k" #"gpt-3.5-turbo-16k" #"gpt-4"
ratio_sampled = 1.00 #0.04
perform_zero_shot = False
context_relevance_grading = True
answer_faithfulness_grading = True
answer_relevance_grading = False

#few_shot_examples_filepath = "../datasets_v2/human_examples_for_few-shot_automatic_evaluation_v1.tsv"
few_shot_examples_filepath = "../datasets_v2/NQ_Few_shot_prompt_v1.tsv"
#few_shot_examples_filepath = "../datasets_v2/HotPotQA_Few_shot_prompt_v1.tsv"
#few_shot_examples_filepath = "../datasets_v2/WoW_Few_shot_prompt_v1.tsv"
#few_shot_examples_filepath = "../datasets_v2/FEVER_Few_shot_prompt_v1.tsv"

evaluation_set = pd.read_csv(evaluation_set_filepath, sep="\t")
#evaluation_set = evaluation_set[evaluation_set["Context_Relevance_Label"].notna()]
#evaluation_set['Query'] = evaluation_set['Question']
#evaluation_set['Context_Relevance_Label'] = evaluation_set['Context_Relevance']
#evaluation_set = evaluation_set[:10]

if "nq" in evaluation_set_filepath.lower() or "hotpotqa" in evaluation_set_filepath.lower() or "wow" in evaluation_set_filepath.lower() or "fever" in evaluation_set_filepath.lower():
    evaluation_set = evaluation_set[evaluation_set["wikipedia_id"].notna()]
    evaluation_set = evaluation_set[evaluation_set["Answer"].notna()]
    #evaluation_set = evaluation_set.drop_duplicates(['Query',"Document"])

    evaluation_set = evaluation_set.sample(n=len(evaluation_set), random_state=43)
    #evaluation_set_sample = evaluation_set.sample(n=len(evaluation_set), random_state=43)
    #evaluation_set_non_sample = evaluation_set.drop(evaluation_set_sample.index)
    #evaluation_set = evaluation_set_sample
elif evaluation_set_filepath == "../datasets_v2/qa_logs_with_logging_details.tsv":
    evaluation_set = evaluation_set[evaluation_set["similarity_score"] != -1]
    evaluation_set = evaluation_set[evaluation_set["retrieval_contexts_used"].str.len() > 5]
    evaluation_set["retrieval_contexts_used"] = evaluation_set["retrieval_contexts_used"].apply(lambda x: clean_document(x))

    evaluation_set['Query'] = evaluation_set['question']
    evaluation_set['Document'] = evaluation_set['retrieval_contexts_used']
    evaluation_set['Answer'] = evaluation_set['new_answer']
    evaluation_set['Context_Relevance_Label'] = [1 for _ in range(len(evaluation_set))]
    evaluation_set['Answer_Faithfulness_Label'] = [1 for _ in range(len(evaluation_set))]
    evaluation_set['Answer_Relevance_Label'] = [1 for _ in range(len(evaluation_set))]


evaluation_set = evaluation_set[:2000]
print("evaluation_set")
print(len(evaluation_set))
print(evaluation_set.head())

few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
print("few_shot_examples")
print(len(few_shot_examples))
print(few_shot_examples.head())

print("Using gpt model: " + gpt_model)
print("Ratio sample for split: " + str(ratio_sampled))
if perform_zero_shot:
    print("Performing zero-shot!")
    print("Setting few-shot example to None...")
    few_shot_examples = None

####################################################################

if "wow" in evaluation_set_filepath.lower():
    context_relevance_system_prompt = "You are an expert dialogue agent. "
    context_relevance_system_prompt += "Given the following dialogue and document, you must analyze the provided document and determine whether it is relevant for responding to the dialogue. "
    context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
    context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
    context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
if "fever" in evaluation_set_filepath.lower():
    context_relevance_system_prompt = "You are an expert fact-checking agent. "
    context_relevance_system_prompt += "Given the following statement and document, you must analyze the provided document and determine whether it is sufficient for determining the statement's factuality. "
    context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided statement's factuality. "
    context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document is not sufficient. '
    context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
else:
    context_relevance_system_prompt = "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
    context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
    context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
    context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

####################################################################

print("Beginning automatic evaluation with few-shot approach!")

if context_relevance_grading:
    tqdm.pandas(desc="Generating context relevance scores...", total=evaluation_set.shape[0])
    evaluation_set["Context_Relevance_Prediction"] = evaluation_set.progress_apply(lambda x: few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(x["Query"]), x["Document"], gpt_model, few_shot_examples), axis=1)

if answer_faithfulness_grading:
    tqdm.pandas(desc="Generating answer faithfulness scores...", total=evaluation_set.shape[0])
    evaluation_set["Answer_Faithfulness_Prediction"] = evaluation_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)

if answer_relevance_grading:
    tqdm.pandas(desc="Generating answer relevance scores...", total=evaluation_set.shape[0])
    evaluation_set["Answer_Relevance_Prediction"] = evaluation_set.progress_apply(lambda x: few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)

####################################################################

print("Beginning scoring!")

if context_relevance_grading:
    evaluation_set_filtered = evaluation_set[evaluation_set['Context_Relevance_Label'].notna()]
    context_relevance_accuracy = calculate_accuracy(evaluation_set_filtered["Context_Relevance_Prediction"].tolist(), evaluation_set_filtered["Context_Relevance_Label"].tolist())
    print("Context Relevance Accuracy: " + str(context_relevance_accuracy) + "%")
if answer_faithfulness_grading:
    evaluation_set_filtered = evaluation_set[evaluation_set['Answer_Faithfulness_Label'].notna()]
    answer_faithfulness_accuracy = calculate_accuracy(evaluation_set_filtered["Answer_Faithfulness_Prediction"].tolist(), evaluation_set_filtered["Answer_Faithfulness_Label"].tolist())
    print("Answer Faithfulness Accuracy: " + str(answer_faithfulness_accuracy) + "%")
if answer_relevance_grading:
    evaluation_set_filtered = evaluation_set[evaluation_set['Answer_Relevance_Label'].notna()]
    answer_relevance_accuracy = calculate_accuracy(evaluation_set_filtered["Answer_Relevance_Prediction"].tolist(), evaluation_set_filtered["Answer_Relevance_Label"].tolist())
    print("Answer Relevance Accuracy: " + str(answer_relevance_accuracy) + "%")

####################################################################

#if "nq" in evaluation_set_filepath or "hotpotqa" in evaluation_set_filepath:

evaluation_set_sample = evaluation_set.sample(n=int(len(evaluation_set) * ratio_sampled), random_state=43)
evaluation_set_non_sample = evaluation_set.drop(evaluation_set_sample.index)
    
added_tag = "_" + str(perform_zero_shot) + "_" + str(context_relevance_grading) + "_" + str(answer_faithfulness_grading) + "_" + str(answer_relevance_grading)
evaluation_set_filepath = evaluation_set_filepath.replace(".tsv", added_tag + "_sampled.tsv")
evaluation_set_sample.to_csv(evaluation_set_filepath, sep="\t")
print("Saving results back to: " + evaluation_set_filepath)

evaluation_set_non_sample_filepath = evaluation_set_filepath.replace("_sampled.tsv", added_tag + "_non_sampled.tsv")
evaluation_set_non_sample.to_csv(evaluation_set_non_sample_filepath, sep="\t")
print("Saving nonsampled dataframe to: " + evaluation_set_non_sample_filepath)

#else:
#    evaluation_set_filepath = evaluation_set_filepath.replace(".tsv", "_sampled.tsv")
#    evaluation_set.to_csv(evaluation_set_filepath, sep="\t")
#    print("Saving results back to: " + evaluation_set_filepath)



