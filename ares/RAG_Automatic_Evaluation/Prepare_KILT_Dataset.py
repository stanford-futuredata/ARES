
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

def join_wikipedia_passages_by_paragraph(given_paragraph_sections, given_paragraph_ids):

    collected_paragraphs = []
    current_id = given_paragraph_ids[0]
    current_text = ""

    assert len(given_paragraph_sections) == len(given_paragraph_ids)
    for given_paragraph, given_paragraph_id in zip(given_paragraph_sections, given_paragraph_ids):
        if current_id == given_paragraph_id:
            current_text += given_paragraph + " "
        else:
            current_text = current_text.strip()
            collected_paragraphs.append(current_text)

            current_text = ""
            current_id = given_paragraph_id
            current_text += given_paragraph + " "

    current_text = current_text.strip()
    collected_paragraphs.append(current_text)

    return collected_paragraphs


####################################################################

def run_kilt(dataset_choice):

    # Get the pre-processed Wikipedia knowledge source for kilt
    kilt_wiki = load_dataset("kilt_wikipedia")['full']

    print("Preparing kilt wiki...")
    kilt_wiki = kilt_wiki.to_pandas()
    kilt_wiki['index_column'] = kilt_wiki['wikipedia_id']
    kilt_wiki = kilt_wiki.set_index("index_column")

    print("Finished loading KILT wiki!")

    ####################################################################

    dataset_choices = [dataset_choice] #"nq", "hotpotqa", "wow", "fever", 
    process_wikipedia = False
    gather_full_wikipedia_articles = False

    #positive_negative_ratios = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]
    positive_negative_ratios = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7]

    ####################################################################

    for dataset_chosen in dataset_choices:

        # Get the KILT task datasets
        kilt_dataset_total = load_dataset("kilt_tasks", name=dataset_choice)
        print("Printing kilt dataset total")
        print(kilt_dataset_total)

        if dataset_chosen in ['nq', "fever", 'hotpotqa', "wow"]:

            folder_path = "../datasets_v2/" + dataset_chosen + "/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            #for split in ['train', 'validation']:
            for split in ['validation']:   

                print("Starting on " + split + " split")

                kilt_dataset = kilt_dataset_total[split]
                kilt_dataset = kilt_dataset.to_pandas()

                wikipedia_ids = []
                wikipedia_passages = []
                paragraph_number = []
                wikipedia_answers = []

                for row in tqdm(range(len(kilt_dataset))):
                    
                    if len(kilt_dataset.iloc[row]['output'][0]['provenance']) > 0:
                        
                        if len(kilt_dataset.iloc[row]['output'][0]['provenance']) == 1 or dataset_chosen == "fever":
                            wiki_id = kilt_dataset.iloc[row]['output'][0]['provenance'][0]['wikipedia_id']
                            start_paragraph_with_answer = kilt_dataset.iloc[row]['output'][0]['provenance'][0]['start_paragraph_id']
                            end_paragraph_with_answer = kilt_dataset.iloc[row]['output'][0]['provenance'][0]['end_paragraph_id']
                            answer = kilt_dataset.iloc[row]['output'][0]['answer']

                            assert start_paragraph_with_answer == end_paragraph_with_answer

                            if not gather_full_wikipedia_articles:
                                wikipedia_passages.append(kilt_wiki.loc[wiki_id]['text']['paragraph'][start_paragraph_with_answer])
                            else:
                                paragraph_sections = kilt_wiki.loc[wiki_id]['text']['paragraph'].tolist()
                                paragraph_sections = [item for item in paragraph_sections if "BULLET" not in item]
                                full_article = (" ").join(paragraph_sections)
                                full_article = full_article.strip()
                                wikipedia_passages.append(full_article)
                            
                            wikipedia_ids.append(wiki_id)
                            paragraph_number.append(start_paragraph_with_answer)
                            wikipedia_answers.append(answer)

                        elif len(kilt_dataset.iloc[row]['output'][0]['provenance']) >= 1 and dataset_chosen == "hotpotqa":

                            provenances = kilt_dataset.iloc[row]['output'][0]['provenance']
                            wiki_ids = [provenances[j]['wikipedia_id'] for j in range(len(provenances))]
                            answer = kilt_dataset.iloc[row]['output'][0]['answer']
                            
                            paragraph_ids = []
                            for provenance in provenances:
                                ids = []
                                for id in range(provenance['start_paragraph_id'], provenance['end_paragraph_id'] + 1):
                                    ids.append(id)
                                assert len(ids) >= 1
                                paragraph_ids.append(ids)

                            assert len(wiki_ids) == len(paragraph_ids)

                            total_context = ""
                            for w_id, included_para_ids in zip(wiki_ids, paragraph_ids):
                                if not gather_full_wikipedia_articles:
                                    paragraph_sections = kilt_wiki.loc[w_id]['text']['paragraph'].tolist()
                                    for paragraph_id in included_para_ids:
                                        total_context += paragraph_sections[paragraph_id] + " "
                                else:
                                    paragraph_sections = kilt_wiki.loc[w_id]['text']['paragraph'].tolist()
                                    paragraph_sections = [item for item in paragraph_sections if "BULLET" not in item]
                                    combined_paragraph_sections = (" ").join(paragraph_sections)
                                    total_context += combined_paragraph_sections + " "
                            
                            total_context.strip()
                            assert len(total_context) > 0

                            wikipedia_ids.append(wiki_ids)
                            wikipedia_passages.append(total_context)
                            paragraph_number.append(paragraph_ids)
                            wikipedia_answers.append(answer)
                    
                    else:
                        wikipedia_ids.append(None)
                        wikipedia_passages.append(None)
                        paragraph_number.append(None)
                        wikipedia_answers.append(None)

                print("Total missing wikipedia ids")
                print(wikipedia_ids.count(None))
                print(wikipedia_passages.count(None))

                kilt_dataset['wikipedia_id'] = wikipedia_ids
                kilt_dataset['Document'] = wikipedia_passages
                kilt_dataset['paragraph_number'] = paragraph_number
                kilt_dataset['Answer'] = wikipedia_answers
                kilt_dataset['Query'] = kilt_dataset['input']

                print("Number of " + dataset_chosen + " question with answer vs. total")
                print(len(kilt_dataset[kilt_dataset["wikipedia_id"].notna()]))
                print(len(kilt_dataset))

                #kilt_dataset = kilt_dataset.sample(n=len(kilt_dataset), random_state=42)
                file_path = folder_path + dataset_chosen + "_reformatted_full_articles_" + str(gather_full_wikipedia_articles) + "_" + split + ".tsv"
                kilt_dataset.to_csv(file_path, sep="\t", index=False)
                print("Saved file to: " + file_path)

                ####################################################################

                #if split in ["train", "validation"] and dataset_chosen in ["nq", "hotpotqa", "wow", "fever"]:
                if split in ["validation"] and dataset_chosen in ["nq", "hotpotqa", "wow", "fever"]:
                    
                    kilt_dataset = kilt_dataset[kilt_dataset["paragraph_number"].notna()]
                    kilt_dataset = kilt_dataset[kilt_dataset["Document"].notna()]
                    kilt_dataset = kilt_dataset[kilt_dataset["wikipedia_id"].notna()]
                    kilt_dataset = kilt_dataset[kilt_dataset["Answer"].notna()]

                    kilt_dataset_copy_1 = kilt_dataset.copy()
                    kilt_dataset_copy_2 = kilt_dataset.copy()
                    
                    incorrect_passages = []
                    context_relevance_labels = []

                    incorrect_answers = []
                    answer_faithfulness_labels = []
                    answer_relevance_labels = []

                    # Gather negatives for context relevance and answer faithfulness
                    for row in tqdm(range(len(kilt_dataset_copy_1))):
                        
                        wiki_id = kilt_dataset_copy_1.iloc[row]['wikipedia_id']
                        if dataset_chosen == "hotpotqa":
                            wiki_id = wiki_id[0]
                            if len(kilt_wiki.loc[wiki_id]['text']['paragraph'][1:].tolist()) == 0:
                                wiki_id = wiki_id[1]
                        answer = kilt_dataset_copy_1.iloc[row]['Answer']



                        if not gather_full_wikipedia_articles:
                            current_wikipedia_passages = kilt_wiki.loc[wiki_id]['text']['paragraph'].tolist()
                        else:
                            #current_wikipedia_passages = [(" ").join(kilt_wiki.loc[wiki_id]['text']['paragraph'].tolist())]
                            paragraphs_to_join = kilt_wiki.loc[wiki_id]['text']['paragraph'].tolist()
                            paragraphs_to_join = [item for item in paragraphs_to_join if "BULLET" not in item]
                            current_wikipedia_passages = [(" ").join(paragraphs_to_join)]
                        
                        
                        filtered_list = [item for item in current_wikipedia_passages if answer not in item] # Filter out paragraphs with answer in it
                        filtered_list = [item for item in current_wikipedia_passages if len(item.strip().split(" ")) >= 50] # Filter out paragraphs with not enough text
                        
                        if row % 2 == 0 and len(filtered_list) > 0 and "wow" != dataset_chosen:
                            incorrect_passages.append(random.choice(filtered_list))
                            context_relevance_labels.append(0)
                        else:
                            random_int = random.randint(0, len(kilt_wiki))
                            cutoff = 100
                            while random_int >= row - cutoff and random_int <= row + cutoff:
                                random_int = random.randint(0, len(kilt_wiki))
                            
                            


                            
                            if not gather_full_wikipedia_articles: 
                                current_wikipedia_passages = kilt_wiki.iloc[random_int]['text']['paragraph'].tolist()
                                for k in range(random_int + 1, min(random_int + cutoff, len(kilt_wiki))):
                                    current_wikipedia_passages.extend(kilt_wiki.iloc[k]['text']['paragraph'].tolist())
                                for k in range(max(random_int - cutoff, 0), random_int - 1):
                                    current_wikipedia_passages.extend(kilt_wiki.iloc[k]['text']['paragraph'].tolist())
                            else:
                                paragraphs_to_join = kilt_wiki.iloc[random_int]['text']['paragraph'].tolist()
                                paragraphs_to_join = [item for item in paragraphs_to_join if "BULLET" not in item]
                                current_wikipedia_passages = [(" ").join(paragraphs_to_join)]
                                for k in range(random_int + 1, min(random_int + cutoff, len(kilt_wiki))):
                                    current_wikipedia_passages.append((" ").join(kilt_wiki.iloc[k]['text']['paragraph'].tolist()))
                                for k in range(max(random_int - cutoff, 0), random_int - 1):
                                    current_wikipedia_passages.append((" ").join(kilt_wiki.iloc[k]['text']['paragraph'].tolist()))
                            
                            

                            
                            
                            filtered_list = [item for item in current_wikipedia_passages if answer not in item]
                            filtered_list = [item for item in current_wikipedia_passages if len(item.strip().split(" ")) >= 50]
                            
                            incorrect_passages.append(random.choice(filtered_list))
                            context_relevance_labels.append(0)

                        random_int = random.randint(0, len(kilt_dataset_copy_1) - 1)
                        if random_int == row:
                            random_int = row - 1
                        random_answer = kilt_dataset_copy_1.iloc[random_int]['output'][0]['answer']
                        incorrect_answers.append(random_answer)
                        answer_faithfulness_labels.append(0)
                        answer_relevance_labels.append(0)


                    kilt_dataset_copy_1['Document'] = incorrect_passages
                    kilt_dataset_copy_1['Context_Relevance_Label'] = context_relevance_labels

                    kilt_dataset_copy_2['Answer'] = incorrect_answers
                    kilt_dataset_copy_2['Answer_Faithfulness_Label'] = answer_faithfulness_labels
                    kilt_dataset_copy_2['Answer_Relevance_Label'] = answer_relevance_labels

                    kilt_dataset['Context_Relevance_Label'] = [1 for _ in range(len(kilt_dataset))]
                    kilt_dataset['Answer_Faithfulness_Label'] = [1 for _ in range(len(kilt_dataset))]
                    kilt_dataset['Answer_Relevance_Label'] = [1 for _ in range(len(kilt_dataset))]

                    total_filepaths = []
                    for ratio in positive_negative_ratios:

                        negatives_to_add = (1 - ratio) / ratio
                        negatives_to_add = int(negatives_to_add * len(kilt_dataset_copy_1))

                        kilt_dataset_combined = pd.concat([kilt_dataset, kilt_dataset_copy_1[:negatives_to_add], kilt_dataset_copy_2[:negatives_to_add]], axis=0, ignore_index=True)
                        kilt_dataset_combined = kilt_dataset_combined.sample(n=len(kilt_dataset_combined), random_state=42) #Shuffled

                        #print("Length before and after adding negatives")
                        #print(len(kilt_dataset))
                        #print(len(kilt_dataset_combined))
                        print("Positive - Negative Ratio")
                        print(str(ratio))
                        print(len(kilt_dataset) / (len(kilt_dataset) + len(kilt_dataset_copy_1[:negatives_to_add])))
                        if dataset_choice == "nq": 
                            file_path = folder_path + "nq_ratio_" + str(ratio) + ".tsv"
                        elif dataset_choice == "hotpotqa": 
                            file_path = folder_path + "hotpotqa_ratio_" + str(ratio) + ".tsv"
                        elif dataset_choice == "wow": 
                            file_path = folder_path + "wow_ratio_" + str(ratio) + ".tsv"
                        elif dataset_choice == "fever": 
                            file_path = folder_path + "fever_ratio_" + str(ratio) + ".tsv"
                        # + "_reformatted_full_articles_" + str(gather_full_wikipedia_articles) + "_" + split + "_with_negatives.tsv"
                        kilt_dataset_combined.to_csv(file_path, sep="\t", index=False)
                        print("Saved file to: " + file_path)
                        print("-------------------------------------------------------")
                        total_filepaths.append(file_path)

                    print("Total Filepaths for Evaluation:")
                    print(str(total_filepaths))

                







                



    ####################################################################

    if process_wikipedia:

        wikipedia_paragraphs = []
        wikipedia_page_ids = []
        wikipedia_paragraph_numbers = []
        #kilt_wiki = kilt_wiki[:10000]

        for row in tqdm(range(len(kilt_wiki))):
            paragraphs = kilt_wiki.iloc[row]['text']['paragraph']
            for paragraph, i in zip(paragraphs, range(len(paragraphs))):
                wikipedia_paragraphs.append(paragraph)
                wikipedia_page_ids.append(kilt_wiki.iloc[row]['wikipedia_id'])
                wikipedia_paragraph_numbers.append(i)

        decompressed_wiki_dataset = {
            "text": wikipedia_paragraphs,
            "wikipedia_id": wikipedia_page_ids,
            "paragraph_number": wikipedia_paragraph_numbers
        }

        decompressed_wiki_dataset = pd.DataFrame(decompressed_wiki_dataset)
        decompressed_wiki_dataset.to_csv("../datasets_v2/decompressed_wikipedia_paragraphs.tsv", sep="\t")


