
from tabnanny import check
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, MptForSequenceClassification, AutoModelForSequenceClassification
import torch
from peft import PeftModel

from Instruction_Finetune import format_text_for_fine_tuning_content_relevance

########################################################################

def extract_answer(output_text: str):
    
    yes_found_index = output_text.lower().find("yes")
    no_found_index = output_text.lower().find("no")

    if yes_found_index > -1 and no_found_index > -1:
        if yes_found_index < no_found_index:
            return "Yes"
        else:
            return "No"
    elif yes_found_index > -1:
        return "Yes"
    elif no_found_index > -1:
        return "No"
    else:
        return "Yes"

########################################################################

""" evaluation_dataset_filename = "../datasets_v2/qa_logs_with_logging_details.tsv"
predictions_filename = "../datasets_v2/qa_logs_with_logging_details_and_model_predictions.tsv"
evaluation_dataset = pd.read_csv(evaluation_dataset_filename, sep="\t")
evaluation_dataset = evaluation_dataset[evaluation_dataset[document_column].str.len() >= 10]

question_column = "question"
document_column = "retrieval_contexts_used"
answer_column = "new_answer" 
evaluation_dataset = evaluation_dataset[evaluation_dataset[document_column].str.len() >= 10] """

########################################################################

evaluation_dataset_filename = "../../datasets/synthetic_queries_v3_test.tsv"
predictions_filename = "../datasets_v2/synthetic_queries_v3_test_with_model_predictions.tsv"
evaluation_dataset = pd.read_csv(evaluation_dataset_filename, sep="\t")
evaluation_dataset['Context_Relevance_Label'] = evaluation_dataset['Label']

question_column = "synthetic_query"
document_column = "document"
answer_column = "generated_answer"
evaluation_dataset = evaluation_dataset[evaluation_dataset[document_column].str.len() >= 10]

print("evaluation_dataset")
print(len(evaluation_dataset))
print(evaluation_dataset.head())

#checkpoint_path = "results/mosaicml-mpt-7b-8k-instruct_final_checkpoint"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v10"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v4.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v5.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v6.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v2.1"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v1.7"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_final_checkpoint_v1.5"

#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_checkpoint_v3.1"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_checkpoint_v3.2"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_checkpoint_v5.1"
#checkpoint_path = "./results/mosaicml-mpt-7b-8k-instruct_checkpoint_v5.3"

#checkpoint_path = "./results/mosaicml-mpt-7b-instruct_sequence_classification_checkpoint_v2.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-instruct_sequence_classification_checkpoint_v1.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-instruct_sequence_classification_checkpoint_v7.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-instruct_sequence_classification_checkpoint_v6.0"

#checkpoint_path = "./results/microsoft-deberta-v3-large_sequence_classification_checkpoint_v1.0"
#checkpoint_path = "./results/microsoft-deberta-v3-large_sequence_classification_checkpoint_v2.0"
checkpoint_path = "./results/microsoft-deberta-v3-large_sequence_classification_checkpoint_v2.0"

###################################################################################

instruction_model = "microsoft/deberta-v3-large" #"mosaicml/mpt-7b-instruct" #"mosaicml/mpt-1b-redpajama-200b-dolly" #"mosaicml/mpt-7b-8k-instruct" #"mosaicml/mpt-30b-instruct"
device = "cuda:0"

########################################################################

#model = PeftModel.from_pretrained(
#    model,
#    checkpoint_path
#)

tqdm.pandas(desc="Generating instructions formatting...", total=evaluation_dataset.shape[0])
evaluation_dataset["content_relevance_text"] = evaluation_dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x[question_column], x[document_column], ""), axis=1)

tokenizer = AutoTokenizer.from_pretrained(instruction_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token






dataset = load_dataset("glue", "qnli")['validation']
dataset = dataset.to_pandas()
dataset["content_relevance_text"] = dataset.progress_apply(lambda x: x["question"] + " | " + x["sentence"], axis=1)
dataset = Dataset.from_pandas(dataset)






max_token_length = 0
for i in range(len(evaluation_dataset)):
    input_ids = tokenizer.encode(evaluation_dataset.iloc[i]["content_relevance_text"], return_tensors='pt')
    max_token_length = max(len(input_ids[0]), max_token_length)

print("max_token_length")
print(max_token_length)
max_token_length = 512
print("Setting max token length to: " + str(max_token_length))

tokenizer = AutoTokenizer.from_pretrained(instruction_model, max_length=max_token_length, 
                                          padding='max_length', truncation=True, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token








device = torch.device(device)
#model = MptForSequenceClassification.from_pretrained(instruction_model, max_seq_len=max_token_length, 
#                                                     num_labels=2, torch_dtype=torch.bfloat16)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2, torch_dtype=torch.bfloat16)
model.to(device)












#model = PeftModel.from_pretrained(
#    model,
#    checkpoint_path
#)

########################################################################

model_predictions = []
model.eval()
for i in tqdm(range(len(evaluation_dataset))):
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
        
                inputs = tokenizer(evaluation_dataset.iloc[i]["content_relevance_text"], return_tensors="pt").to('cuda')
                logits = model(**inputs).logits
                predictions = torch.argmax(logits, dim=-1)
                #print("Predictions")
                #print(predictions)
                if predictions[0] == 1:
                    model_predictions.append("Yes")
                elif predictions[0] == 0:
                    model_predictions.append("No")  

########################################################################

evaluation_dataset["Model_Predictions_for_Context_Relevance"] = model_predictions

print("Model Preciction Counts")
print("Yes: " + str(len(evaluation_dataset[evaluation_dataset["Model_Predictions_for_Context_Relevance"] == "Yes"])))
print("No: " + str(len(evaluation_dataset[evaluation_dataset["Model_Predictions_for_Context_Relevance"] == "No"])))

evaluation_dataset.to_csv(predictions_filename, index=False, sep="\t")
print("Saved prediction to: " + predictions_filename)

########################################################################

print("Beginning accuracy scoring...")

evaluation_dataset['context_relevance_correct_classification'] = evaluation_dataset['Context_Relevance_Label'] == evaluation_dataset['Model_Predictions_for_Context_Relevance']

correct = len(evaluation_dataset[evaluation_dataset['context_relevance_correct_classification'] == True])
incorrect = len(evaluation_dataset[evaluation_dataset['context_relevance_correct_classification'] == False])

print("Correct: " + str(correct))
print("Incorrect: " + str(incorrect))
print("Accuracy Percentage: " + str(round((correct * 100) / (correct + incorrect), 2)))