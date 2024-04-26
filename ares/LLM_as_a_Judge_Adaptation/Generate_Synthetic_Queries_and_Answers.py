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
import argparse
import pdb
import sys
import warnings

from ares.LLM_as_a_Judge_Adaptation.LLM_Generation_Functions import generate_synthetic_query_openai_approach, generate_answer_from_context, generate_contradictory_answer_from_context
from ares.LLM_as_a_Judge_Adaptation.LLM_Generation_Functions import check_generated_answer, generate_contradictory_answer_examples, generate_synthetic_query_llm_approach, generate_answer_llm_approach

from ares.LLM_as_a_Judge_Adaptation.Filter_Synthetic_Queries import get_embedding, generate_index, filter_synthetic_queries, generate_additional_negatives, generate_additional_positives
#from Instruction_Finetune import instruction_finetune
#from Instruction_Finetune_V2 import instruction_finetune_v2
#from Instruction_Finetune_v3 import instruction_finetune_v3

pd.set_option('display.max_columns', None)  # Show all columns - TEST
pd.set_option('display.max_rows', None)  # Show all rows - TEST
pd.set_option('display.max_colwidth', None)  # Show full content of each column - TEST

#################################################

def clean_document(document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")) #[:512]
    return cleaned_document

    # Load model for synthetic query generation 

def validate_input_file(df, required_columns) -> bool: 
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        sys.exit(f"Error: The DataFrame is missing the following required column(s): {', '.join(missing_columns)}.")
############

def load_model(flan_approach, model_choice):
    if flan_approach:
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

        torch.no_grad()
        model.eval()

        device = "cuda:0"
        device = torch.device(device)
        model.to(device)
    return model, tokenizer, device

#############

# Load documents for synthetic query and answer generation
def load_documents(document_filepath, clean_documents, documents_sampled):
    documents = []
    required_columns = ['Query', 'Document', 'Answer']

    if "docs_aws" in document_filepath:

        with open(document_filepath, "r") as json_file:
            json_data = json.load(json_file)
            documents = [x['text'] for x in json_data]

            # Clean document
            if clean_documents:
                documents = [clean_document(text) for text in documents]

        documents = pd.DataFrame(documents, columns=["document"])

    else:
        if not document_filepath.endswith('.tsv'):
            sys.exit(f"Error: The file {document_filepath} is not a TSV file.")
        try:
            documents = pd.read_csv(document_filepath, sep="\t")
            validate_input_file(documents, required_columns)
            documents.rename(columns={"Document": "document"}, inplace=True)
            documents['document'] = documents['document'].str.strip()
        except Exception as e:
            sys.exit(f"Error reading the file {document_filepath}: {e}")


    initial_count = len(documents)
    documents = documents[documents['document'].str.split().apply(len) >= 50] # Filter documents w/ less than 50 words.
    after_filter_count = len(documents)

    count = initial_count - after_filter_count

    if(after_filter_count == 0): 
        sys.exit(f"All documents were less than 50 words, please provide dataset with documents containing more than 50 words")

    if documents_sampled > initial_count:
        print(f"\nThe `documents_sampled` parameter ({documents_sampled}) exceeds the available number of documents ({initial_count}). Sampling will be adjusted to the maximum available documents ({initial_count}).\n")
        documents_sampled = initial_count

    if count > 0: 
        print(f"Filtered out {count} documents because they had less than 50 words. Sampling will be be adjusted to {after_filter_count} documents\n")
        documents_sampled = after_filter_count

    documents = documents.sample(n=documents_sampled, random_state=43)

    #documents = documents[:10]
    # print("documents") - CHANGED
    # print(len(documents)) - CHANGED
    # print(documents.head()) - CHANGED
    return documents

################

# Load few-shot prompt
def load_few_shot_prompt(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset):
    few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt = few_shot_prompt[few_shot_prompt['Context_Relevance_Label'] == "[[Yes]]"]
    if "Query" not in few_shot_prompt:
        few_shot_prompt['Query'] = few_shot_prompt['Question']

    length_of_fewshot_prompt = len(few_shot_prompt)
    few_shot_examples = ""
    for row in range(len(few_shot_prompt)):
        few_shot_examples += "Example " + str(row + 1) + ":\n"
        few_shot_examples += "Document: " + clean_document(few_shot_prompt.iloc[row]['Document']) + "\n"
        if for_fever_dataset:
            few_shot_examples += "Statement: " + few_shot_prompt.iloc[row]['Query'] + "\n\n"
        elif for_wow_dataset:
            few_shot_examples += "Dialogue: " + few_shot_prompt.iloc[row]['Query'] + "\n\n"
        else:
            few_shot_examples += "Question: " + few_shot_prompt.iloc[row]['Query'] + "\n\n"

    # print("Fewshot Prompt") - CHANGED
    # print(few_shot_examples) - CHANGED 
    # print("Finished loading dataset + model") - CHANGED
    return few_shot_examples, length_of_fewshot_prompt

#################################################
def generate_contradictory_answers(few_shot_prompt_filename,generate_contradictory_answers_with_flan,for_fever_dataset,for_wow_dataset): 
    few_shot_prompt_for_contradictory_answers = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt_for_contradictory_answers = few_shot_prompt_for_contradictory_answers[few_shot_prompt_for_contradictory_answers['Contradictory_Answer'].str.len() > 4]

    if generate_contradictory_answers_with_flan:
        few_shot_examples_for_contradictory_answers = ""
        for row in range(len(few_shot_prompt_for_contradictory_answers)):
            few_shot_examples_for_contradictory_answers += "Example " + str(row + 1) +":\n"
            few_shot_examples_for_contradictory_answers += "Document: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Document'] + "\n"
            if for_fever_dataset:
                few_shot_examples_for_contradictory_answers += "Statement: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Query'] + "\n"
                few_shot_examples_for_contradictory_answers += "Incorrect Answer: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer'] + "\n\n"
            elif for_wow_dataset:
                few_shot_examples_for_contradictory_answers += "Dialogue: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Query'] + "\n"
                few_shot_examples_for_contradictory_answers += "Incorrect Response: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer'] + "\n\n"
            else:
                few_shot_examples_for_contradictory_answers += "Question: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Query'] + "\n"
                few_shot_examples_for_contradictory_answers += "Incorrect Answer: " + few_shot_prompt_for_contradictory_answers.iloc[row]['Contradictory_Answer'] + "\n\n"
    return few_shot_examples_for_contradictory_answers

def generate_few_shot_prompts(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset):
    answer_gen_few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[answer_gen_few_shot_prompt['Answer_Relevance_Label'] == "[[Yes]]"]
    answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[answer_gen_few_shot_prompt['Answer_Faithfulness_Label'] == "[[Yes]]"]
    #answer_gen_few_shot_prompt = answer_gen_few_shot_prompt[:4]
    length_of_fewshot_prompt_answer_gen = len(answer_gen_few_shot_prompt)
    if "Query" in answer_gen_few_shot_prompt.columns:
        answer_gen_few_shot_prompt['Question'] = answer_gen_few_shot_prompt['Query']

    answer_gen_few_shot_examples = ""
    for row in range(len(answer_gen_few_shot_prompt)):
        answer_gen_few_shot_examples += "Example " + str(row + 1) +":\n"
        answer_gen_few_shot_examples += "Document: " + answer_gen_few_shot_prompt.iloc[row]['Document'] + "\n"
        if for_fever_dataset:
            answer_gen_few_shot_examples += "Statement: " + answer_gen_few_shot_prompt.iloc[row]['Query'] + "\n"
            answer_gen_few_shot_examples += "Answer: " + answer_gen_few_shot_prompt.iloc[row]['Answer'] + "\n\n"
        elif for_wow_dataset:
            answer_gen_few_shot_examples += "Dialogue: " + answer_gen_few_shot_prompt.iloc[row]['Query'] + "\n"
            answer_gen_few_shot_examples += "Response: " + answer_gen_few_shot_prompt.iloc[row]['Answer'] + "\n\n"
        else:
            answer_gen_few_shot_examples += "Question: " + answer_gen_few_shot_prompt.iloc[row]['Query'] + "\n"
            answer_gen_few_shot_examples += "Answer: " + answer_gen_few_shot_prompt.iloc[row]['Answer'] + "\n\n"

    # print("answer_gen_few_shot_examples") - CHANGED
    # print(answer_gen_few_shot_examples) - CHANGED
    return answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen

##############################

 # Save Synthetic Queries
def save_synthetic_queries(documents, regenerate_synth_questions, flan_approach, few_shot_examples, 
    length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset, for_wow_dataset, 
    synthetic_query_prompt, synthetic_queries_filename, question_temperatures): 

    # print("documents") - CHANGED 
    # print(len(documents)) - CHANGED
    # print(documents.head()) - CHANGED

    if regenerate_synth_questions:
        print("Beginning synthetic query generation!")
        tqdm.pandas(desc="Generating synthetic queries...", total=documents.shape[0])
        if flan_approach:
            # Testing
            # subset_documents = documents.head(10)  #
            # subset_documents["synthetic_query"] = subset_documents.apply(lambda x: generate_synthetic_query_llm_approach(x["document"], few_shot_examples, length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset), axis=1)
            # print(subset_documents[["document", "synthetic_query"]])
            # breakpoint()
            # synthetic_query = generate_synthetic_query_llm_approach(test_doc["document"], few_shot_examples, length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset) #
            # print(synthetic_query)
            documents["synthetic_query"] = documents.progress_apply(lambda x: generate_synthetic_query_llm_approach(x["document"], few_shot_examples, length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset), axis=1)
        else:
            documents["synthetic_query"] = documents.progress_apply(lambda x: generate_synthetic_query_openai_approach(x["document"], synthetic_query_prompt, few_shot_examples, question_temperatures, length_of_fewshot_prompt), axis=1)
        documents = documents.explode("synthetic_query", ignore_index=True)
        documents = documents.drop_duplicates(subset=['synthetic_query'])
        documents.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print("Saved synthetic queries to: " + synthetic_queries_filename)

################################################

# Generate synthetic queries and their answers for training set
def Generate_Synthetic_Answers(synthetic_queries_filename,
    regenerate_answers, flan_approach, answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen, 
    device, tokenizer, model, for_fever_dataset, for_wow_dataset, generate_contradictory_answers_with_flan,
    few_shot_examples_for_contradictory_answers, number_of_negatives_added_ratio,
    lower_bound_for_negatives, number_of_contradictory_answers_added_ratio, number_of_positives_added_ratio, regenerate_embeddings): 
    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    # print("Filtered synth queries")
    # print(len(synth_queries))
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]
    # print(len(synth_queries))

    if regenerate_answers:
        print("Beginning answer generation!")
        
        tqdm.pandas(desc="Generating answers...", total=synth_queries.shape[0])
        if flan_approach:
            synth_queries["generated_answer"] = synth_queries.progress_apply(lambda x: generate_answer_llm_approach(x["document"], x["synthetic_query"], answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen, device, tokenizer, model, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset), axis=1)
        else:
            synth_queries["generated_answer"] = synth_queries.progress_apply(lambda x: generate_answer_from_context(x["document"], x["synthetic_query"]), axis=1)
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print("Saved answers to: " + synthetic_queries_filename)

        #################################################

        #  for idx, row in synth_queries.iterrows():
        #     print(f"Document: {row['document']}\nSynthetic Query: {row['synthetic_query']}\nGenerated Answer: {row['generated_answer']}\n---\n")
        # breakpoint()

        synth_queries["Answer_Faithfulness_Label"] = [check_generated_answer(synth_queries.iloc[i]['generated_answer']) for i in range(len(synth_queries))]
        synth_queries["Answer_Relevance_Label"] = [check_generated_answer(synth_queries.iloc[i]['generated_answer']) for i in range(len(synth_queries))]
        print("Generating contradictory answers!")
        if generate_contradictory_answers_with_flan:
            synth_queries = generate_contradictory_answer_examples(synth_queries, int(len(synth_queries) * number_of_contradictory_answers_added_ratio), few_shot_examples_for_contradictory_answers=few_shot_examples_for_contradictory_answers, device=device, tokenizer=tokenizer, model=model, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset)
        else:
            synth_queries = generate_contradictory_answer_examples(synth_queries, int(len(synth_queries) * number_of_contradictory_answers_added_ratio))
        # print(synth_queries.columns)  # List all column names to understand the structure
        # print(synth_queries.head()) 
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print("Saved answers to: " + synthetic_queries_filename)

    #################################################

    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]

    # print("synth_queries")
    # print(len(synth_queries))
    # print(synth_queries.head())
    

    if regenerate_embeddings:
        print("Generating index and negatives!")
        documentation_index = generate_index(synth_queries)
        synth_queries = filter_synthetic_queries(synth_queries, documentation_index)
        synth_queries = generate_additional_negatives(synth_queries, documentation_index, number_of_negatives_added_ratio, lower_bound_for_negatives)
        synth_queries = generate_additional_positives(synth_queries, documentation_index, number_of_positives_added_ratio)
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")

    #################################################

    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")

    #Shuffle queries
    synth_queries = synth_queries.sample(n=len(synth_queries), random_state=42)

    # print("Label Sets for each Metric")
    # print(set(synth_queries['Context_Relevance_Label'].tolist()))
    # print(set(synth_queries['Answer_Faithfulness_Label'].tolist()))
    # print(set(synth_queries['Answer_Relevance_Label'].tolist()))

    # print("synth_queries filtered")
    # print(len(synth_queries))
    # print(synth_queries.head())

    # print("Positive and Negative Counts")
    # print("Context Relevance")
    # print(len(synth_queries[synth_queries['Context_Relevance_Label'] == "Yes"]))
    # print(len(synth_queries[synth_queries['Context_Relevance_Label'] == "No"]))
    # print("Answer Faithfulness")
    # print(len(synth_queries[synth_queries['Answer_Faithfulness_Label'] == "Yes"]))
    # print(len(synth_queries[synth_queries['Answer_Faithfulness_Label'] == "No"]))
    # print("Answer Relevance")
    # print(len(synth_queries[synth_queries['Answer_Relevance_Label'] == "Yes"]))
    # print(len(synth_queries[synth_queries['Answer_Relevance_Label'] == "No"]))

    synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
    print("Completed synthetic generation!")
    print("Saved synthetic queries file to: " + synthetic_queries_filename)

# def print_synthetic_queries(filename): 
#     queries = pd.read_csv(filename,sep='\t')
    
#     for index, row in queries.head(6).iterrows(): 
#         print(f"Document {index}: {row['document']}")
#         print(f"Synthetic Query: {row['synthetic_query']}")
#         print(f"Answer: {row['generated_answer']}")
#         print("-" * 50)

# def synthetic_generator_config(document_filepath: str, few_shot_prompt_filename: str,
#                                synthetic_queries_filename: str, documents_sampled: int,
#                                flan_approach: bool = True, clean_documents: bool = False,
#                                regenerate_synth_questions: bool = True, 
#                                percentiles: list = [0.05, 0.25, 0.5, 0.95], 
#                                question_temperatures: list = [2.0, 1.5, 1.0, 0.5, 0.0],
#                                regenerate_answers: bool = True,
#                                generate_contradictory_answers_with_flan: bool = True, 
#                                number_of_negatives_added_ratio: float = 0.5, # Check whether can also be an int
#                                lower_bound_for_negatives: int = 5, # Need to be an int value
#                                number_of_contradictory_answers_added_ratio: float = 0.67, # Check whether can also be an int
#                                number_of_positives_added_ratio: float = 0.0, # Check whether can also be an int
#                                regenerate_embeddings: float = True, 
#                                synthetic_query_prompt: str = "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n"
#                                ): 
#     for_fever_dataset = False
#     if "fever" in document_filepath.lower():
#         for_fever_dataset = True
#     for_wow_dataset = False
#     if "wow" in document_filepath.lower():
#         for_wow_dataset = True

#     model, model_choice, tokenizer, device = load_model(flan_approach)

#     documents = load_documents(document_filepath, clean_documents, documents_sampled)

#     few_shot_examples, length_of_fewshot_prompt = load_few_shot_prompt(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset)

#     few_shot_examples_for_contradictory_answers = generate_contradictory_answers(few_shot_prompt_filename,generate_contradictory_answers_with_flan,for_fever_dataset,for_wow_dataset)

#     answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen = generate_few_shot_prompts(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset)

#     save_synthetic_queries(documents, regenerate_synth_questions, flan_approach, few_shot_examples, 
#     length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset, for_wow_dataset, 
#     synthetic_query_prompt, synthetic_queries_filename, question_temperatures)

#     Generate_Synthetic_Queries_and_Answers(synthetic_queries_filename,
#     regenerate_answers, flan_approach, answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen, 
#     device, tokenizer, model, for_fever_dataset, for_wow_dataset, generate_contradictory_answers_with_flan,
#     few_shot_examples_for_contradictory_answers, number_of_negatives_added_ratio, lower_bound_for_negatives, number_of_contradictory_answers_added_ratio, number_of_positives_added_ratio, regenerate_embeddings)

#     print_synthetic_queries(synthetic_queries_filename)

#################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--document_filepath", type=str, required=True)
    parser.add_argument("--few_shot_prompt_filename", type=str, required=True)
    parser.add_argument("--synthetic_queries_filename", type=str, required=True)
    parser.add_argument("--flan_approach", type=bool, default="True", required=False)
    parser.add_argument("--documents_sampled", type=int, required=True)

    args = parser.parse_args()

    ### Instructions

    document_filepath = args.document_filepath
    few_shot_prompt_filename = args.few_shot_prompt_filename
    synthetic_queries_filename = args.synthetic_queries_filename
    flan_approach = args.flan_approach
    if flan_approach == "True":
        flan_approach = True 
    else:
        flan_approach = False
    documents_sampled = args.documents_sampled

    ########################################################

    for_fever_dataset = False
    if "fever" in document_filepath.lower():
        for_fever_dataset = True
    for_wow_dataset = False
    if "wow" in document_filepath.lower():
        for_wow_dataset = True

    regenerate_synth_questions = True
    regenerate_answers = True
    regenerate_embeddings = True

    lower_bound_for_negatives = 20
    number_of_negatives_added_ratio = 0.5
    number_of_positives_added_ratio = 0.0 #0.2
    number_of_contradictory_answers_added_ratio = 0.67
    clean_documents = False

    question_temperatures = [2.0, 1.5, 1.0, 0.5, 0.0] #[2.0, 0.9, 0.7, 0.5]
    percentiles = [0.05, 0.25, 0.5, 0.95] #[0.01, 0.05, 0.25, 0.5, 0.95]
    clean_few_shot_prompt_docs = False

    flan_approach = True
    generate_contradictory_answers_with_flan = True

    #################################################

    synthetic_query_prompt = "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n"

    print("------------------------------------------------------------")
    print("Document File: " + document_filepath)
    print("Synthetic File Path: " + synthetic_queries_filename)
    print("number_of_negatives_added_ratio: " + str(number_of_negatives_added_ratio))
    print("number_of_positives_added_ratio: " + str(number_of_positives_added_ratio))
    print("number_of_contradictory_answers_added_ratio: " + str(number_of_contradictory_answers_added_ratio))
    print("clean_documents: " + str(clean_documents))
    print("question_temperatures: " + str(question_temperatures))
    print("percentiles: " + str(percentiles))
    print("lower_bound_for_negatives: " + str(lower_bound_for_negatives))
    print("for_fever_dataset: " + str(for_fever_dataset))
    print("for_wow_dataset: " + str(for_wow_dataset))
    print("------------------------------------------------------------")

    # model, model_choice, tokenizer, device = load_model(flan_approach)

    # documents = load_documents(document_filepath, clean_documents)

    # few_shot_examples, length_of_fewshot_prompt = load_few_shot_prompt(few_shot_prompt_filename,for_fever_dataset)

    # few_shot_examples_for_contradictory_answers = generate_contradictory_answers(few_shot_prompt_filename)

    # answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen = generate_few_shot_prompts(few_shot_prompt_filename)

    # save_synthetic_queries(documents, regenerate_synth_questions, flan_approach, few_shot_examples, 
    # length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset, for_wow_dataset, 
    # synthetic_query_prompt, synthetic_queries_filename, question_temperatures)

    # Generate_Synthetic_Queries_and_Answers(synthetic_queries_filename,
    # regenerate_answers, flan_approach, answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen, 
    # device, tokenizer, model, for_fever_dataset, for_wow_dataset, generate_contradictory_answers_with_flan,
    # number_of_contradictory_answers_added_ratio, few_shot_examples_for_contradictory_answers, number_of_negatives_added_ratio
    # lower_bound_for_negatives, number_of_contradictory_answers_added_ratio, number_of_positives_added_ratio, regenerate_embeddings): 

    ######################################################################