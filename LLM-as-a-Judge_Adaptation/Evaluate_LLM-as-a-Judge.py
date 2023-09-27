
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
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
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

checkpoint_path = "./results/mosaicml-mpt-7b-instruct_generation_checkpoint_v1.0"
#checkpoint_path = "./results/mosaicml-mpt-7b-instruct_generation_checkpoint_v2.0"

###################################################################################

instruction_model = "mosaicml/mpt-7b-instruct" #"mosaicml/mpt-1b-redpajama-200b-dolly" #"mosaicml/mpt-7b-8k-instruct" #"mosaicml/mpt-30b-instruct"
device = "cuda:0"
output_length = 10

########################################################################

#model = PeftModel.from_pretrained(
#    model,
#    checkpoint_path
#)

tqdm.pandas(desc="Generating instructions formatting...", total=evaluation_dataset.shape[0])
evaluation_dataset["content_relevance_text"] = evaluation_dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x[question_column], x[document_column], ""), axis=1)

tokenizer = AutoTokenizer.from_pretrained(instruction_model, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token

max_token_length = 0
for i in range(len(evaluation_dataset)):
    input_ids = tokenizer.encode(evaluation_dataset.iloc[i]["content_relevance_text"], return_tensors='pt')
    max_token_length = max(len(input_ids[0]), max_token_length)

print("max_token_length")
print(max_token_length)
print("Changing max token length...")








#max_token_length = 2930









tokenizer = AutoTokenizer.from_pretrained(instruction_model, max_length=max_token_length, 
                                          padding='max_length', truncation=True, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(instruction_model, trust_remote_code=True)
#config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = device #"cuda:0" #'meta'
config.max_seq_len = max_token_length + output_length #3036 #4096
config.trust_remote_code = True

#bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.float16,
#)

model = AutoModelForCausalLM.from_pretrained(
  instruction_model,
  config=config,
  #quantization_config=bnb_config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True,
  use_auth_token=True
)

model = PeftModel.from_pretrained(
    model,
    checkpoint_path
)

########################################################################

print("Testing dummy example...")
with torch.autocast('cuda', dtype=torch.bfloat16):
    inputs = tokenizer('Here is a recipe for vegan banana bread:\n', return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=100)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(output)
    print("-----------------------------------------------------------")

########################################################################

model_predictions = []
model.eval()
for i in tqdm(range(len(evaluation_dataset))):
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            try:
                #inputs = tokenizer(evaluation_dataset.iloc[i]["content_relevance_text"], return_tensors="pt", 
                #                   max_length=config.max_seq_len, padding='max_length', truncation=True).to('cuda')
                inputs = tokenizer(evaluation_dataset.iloc[i]["content_relevance_text"], return_tensors="pt").to('cuda')
                outputs = model.generate(**inputs, max_new_tokens=output_length)
                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                output_text_extracted = extract_answer(output_text.split("### Response:")[1])
                model_predictions.append(output_text_extracted)

                #print(evaluation_dataset.iloc[i]["content_relevance_text"])
                print("------------------------------------------")
                print(output_text)
                print("Extracted response: " + output_text_extracted)
                print("Correct Label: " + evaluation_dataset.iloc[i]["Label"])
                print("------------------------------------------\n\n\n\n")
            except:
                print("Error for row " + str(i))
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