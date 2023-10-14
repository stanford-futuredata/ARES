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

from LLM_Generation_Functions import generate_synthetic_query_openai_approach, generate_answer_from_context, generate_contradictory_answer_from_context
from LLM_Generation_Functions import check_generated_answer, generate_contradictory_answer_examples, generate_synthetic_query_llm_approach, generate_answer_llm_approach

from Filter_Synthetic_Queries import get_embedding, generate_index, filter_synthetic_queries, generate_additional_negatives, generate_additional_positives
#from Instruction_Finetune import instruction_finetune
#from Instruction_Finetune_V2 import instruction_finetune_v2
#from Instruction_Finetune_v3 import instruction_finetune_v3

#################################################

def clean_document(document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")) #[:512]
    return cleaned_document

#################################################





#################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--document_filepath", type=str, required=True)
    parser.add_argument("--few_shot_prompt_filename", type=str, required=True)
    parser.add_argument("--answer_gen_few_shot_prompt_filename", type=str, required=True)
    parser.add_argument("--synthetic_queries_filename", type=str, required=True)
    parser.add_argument("--flan_approach", type=bool, default=True, required=False)
    parser.add_argument("--documents_sampled", type=int, required=True)

    args = parser.parse_args()

    ### Instructions

    document_filepath = args.document_filepath
    few_shot_prompt_filename = args.few_shot_prompt_filename
    answer_gen_few_shot_prompt_filename = args.answer_gen_few_shot_prompt_filename
    synthetic_queries_filename = args.synthetic_queries_filename
    flan_approach = args.flan_approach
    documents_sampled = args.documents_sampled

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
    chosen_score_threshold = 0.01
    number_of_contradictory_answers_added_ratio = 0.67
    clean_documents = False

    question_temperatures = [2.0, 1.5, 1.0, 0.5, 0.0] #[2.0, 0.9, 0.7, 0.5]
    percentiles = [0.05, 0.25, 0.5, 0.95] #[0.01, 0.05, 0.25, 0.5, 0.95]
    clean_few_shot_prompt_docs = False

    flan_approach = True
    generate_contradictory_answers_with_flan = True

    training_filename = synthetic_queries_filename.replace(".tsv", "_train.tsv")
    test_filename = synthetic_queries_filename.replace(".tsv", "_test.tsv")

    #################################################







    #################################################

    synthetic_query_prompt = "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n"

    instruction_model = "microsoft/deberta-v3-large"
    device = "cuda:0"
    use_eval_set = True

    print("Model selected: " + instruction_model)
    print("Document File: " + document_filepath)
    print("Synthetic File Path: " + synthetic_queries_filename)
    print("number_of_negatives_added_ratio: " + str(number_of_negatives_added_ratio))
    print("number_of_positives_added_ratio: " + str(number_of_positives_added_ratio))
    print("chosen_score_threshold: " + str(chosen_score_threshold))
    print("number_of_contradictory_answers_added_ratio: " + str(number_of_contradictory_answers_added_ratio))
    print("clean_documents: " + str(clean_documents))
    print("question_temperatures: " + str(question_temperatures))
    print("percentiles: " + str(percentiles))
    print("lower_bound_for_negatives: " + str(lower_bound_for_negatives))
    print("for_fever_dataset: " + str(for_fever_dataset))
    print("for_wow_dataset: " + str(for_wow_dataset))

    #################################################

    # Load model for synthetic query generation 
    if flan_approach:
        model_choice = "google/flan-t5-xxl" #google/flan-t5-xxl
        tokenizer = AutoTokenizer.from_pretrained(model_choice)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

        torch.no_grad()
        model.eval()

        device = "cuda:0"
        device = torch.device(device)
        model.to(device)

    #################################################

    # Load documents for synthetic query and answer generation
    documents = []
    if "docs_aws" in document_filepath:

        with open(document_filepath, "r") as json_file:
            json_data = json.load(json_file)
            documents = [x['text'] for x in json_data]

            # Clean document
            if clean_documents:
                documents = [clean_document(text) for text in documents]

        documents = pd.DataFrame(documents, columns=["document"])

    else:
        documents = pd.read_csv(document_filepath, sep="\t")
        if "multirc" not in document_filepath and "record" not in document_filepath:
            documents = documents[documents["wikipedia_id"].notna()]
        documents = documents[documents["Answer"].notna()]
        documents.rename(columns={"Document": "document"}, inplace=True)
        documents['document'] = documents['document'].str.strip()
        if "nq" in document_filepath:
            documents = documents[documents["document"].str.len() > 100]
        documents = documents.sample(n=documents_sampled, random_state=43)

    #documents = documents[:10]
    print("documents")
    print(len(documents))
    print(documents.head())

    #################################################

    # Load few-shot prompt
    few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
    few_shot_prompt = few_shot_prompt[few_shot_prompt['Context_Relevance_Label'] == "[[Yes]]"]

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

    print("Fewshot Prompt")
    print(few_shot_examples)

    print("Finished loading dataset + model")

    #################################################

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

    #################################################

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

    print("answer_gen_few_shot_examples")
    print(answer_gen_few_shot_examples)

    #################################################

    # Generate synthetic queries and their answers for training set

    print("documents")
    print(len(documents))
    print(documents.head())

    if regenerate_synth_questions:
        print("Beginning synthetic query generation!")
        tqdm.pandas(desc="Generating synthetic queries...", total=documents.shape[0])
        if flan_approach:
            documents["synthetic_query"] = documents.progress_apply(lambda x: generate_synthetic_query_llm_approach(x["document"], few_shot_examples, length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset), axis=1)
        else:
            documents["synthetic_query"] = documents.progress_apply(lambda x: generate_synthetic_query_openai_approach(x["document"], synthetic_query_prompt, few_shot_examples, question_temperatures, length_of_fewshot_prompt), axis=1)
        documents = documents.explode("synthetic_query", ignore_index=True)
        documents = documents.drop_duplicates(subset=['synthetic_query'])
        documents.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print("Saved synthetic queries to: " + synthetic_queries_filename)

    #################################################

    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    print("Filtered synth queries")
    print(len(synth_queries))
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]
    print(len(synth_queries))

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

        synth_queries["Answer_Faithfulness_Label"] = [check_generated_answer(synth_queries.iloc[i]['generated_answer']) for i in range(len(synth_queries))]
        synth_queries["Answer_Relevance_Label"] = [check_generated_answer(synth_queries.iloc[i]['generated_answer']) for i in range(len(synth_queries))]
        print("Generating contradictory answers!")
        if generate_contradictory_answers_with_flan:
            synth_queries = generate_contradictory_answer_examples(synth_queries, int(len(synth_queries) * number_of_contradictory_answers_added_ratio), few_shot_examples_for_contradictory_answers=few_shot_examples_for_contradictory_answers, device=device, tokenizer=tokenizer, model=model, for_fever_dataset=for_fever_dataset, for_wow_dataset=for_wow_dataset)
        else:
            synth_queries = generate_contradictory_answer_examples(synth_queries, int(len(synth_queries) * number_of_contradictory_answers_added_ratio))
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
        print("Saved answers to: " + synthetic_queries_filename)

    #################################################

    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")
    #synth_queries = synth_queries[:10]
    #synth_queries = synth_queries.drop('__index_level_0__', axis=1)

    print("synth_queries")
    print(len(synth_queries))
    print(synth_queries.head())

    print("Filtered synth queries")
    print(len(synth_queries))
    synth_queries = synth_queries[synth_queries["synthetic_query"].str.len() > 10]
    print(len(synth_queries))

    if regenerate_embeddings:
        print("Generating index and negatives!")
        documentation_index = generate_index(synth_queries)
        synth_queries = filter_synthetic_queries(synth_queries, documentation_index)
        synth_queries = generate_additional_negatives(synth_queries, documentation_index, number_of_negatives_added_ratio, lower_bound_for_negatives)
        synth_queries = generate_additional_positives(synth_queries, documentation_index, number_of_positives_added_ratio, chosen_score_threshold)
        synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")

    #################################################

    synth_queries = pd.read_csv(synthetic_queries_filename, sep="\t")

    #Shuffle queries
    synth_queries = synth_queries.sample(n=len(synth_queries), random_state=42)

    print("synth_queries filtered")
    print(len(synth_queries))
    print(set(synth_queries['Context_Relevance_Label'].tolist()))
    print(set(synth_queries['Answer_Faithfulness_Label'].tolist()))
    print(set(synth_queries['Answer_Relevance_Label'].tolist()))
    print(synth_queries.head())

    print("Positive and Negative Counts")
    print(len(synth_queries[synth_queries['Context_Relevance_Label'] == "Yes"]))
    print(len(synth_queries[synth_queries['Context_Relevance_Label'] == "No"]))
    print(len(synth_queries[synth_queries['Answer_Faithfulness_Label'] == "Yes"]))
    print(len(synth_queries[synth_queries['Answer_Faithfulness_Label'] == "No"]))
    print(len(synth_queries[synth_queries['Answer_Relevance_Label'] == "Yes"]))
    print(len(synth_queries[synth_queries['Answer_Relevance_Label'] == "No"]))

    synth_queries.to_csv(synthetic_queries_filename, index=False, sep="\t")
    print("Completed synthetic generation!")
    print("Saved synthetic queries file to: " + synthetic_queries_filename)

    ######################################################################

