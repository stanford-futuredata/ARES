
from cProfile import label
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
#import tensorflow as tf

import pandas as pd
import numpy as np
import ast
import datasets
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import pyarrow as pa
import pyarrow.dataset as ds

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoModelForCausalLM, AutoConfig, MptForSequenceClassification, AutoModelForSequenceClassification

import torch
from tqdm.auto import tqdm
import statistics
import time

import subprocess as sp
import os

from sklearn.model_selection import train_test_split
import json
import random

import re

#############################################################

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice, dropout_layer, frozen, 
                 frozen_layer_count, average_hidden_state, frozen_embeddings):

          super(CustomBERTModel, self).__init__()
          #self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
          if model_choice in ["mosaicml/mpt-7b-instruct", "mosaicml/mpt-7b"]:

            config = AutoConfig.from_pretrained(model_choice, trust_remote_code=True)
            config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
            config.max_seq_len = max_token_length

            model_encoding = AutoModelForCausalLM.from_pretrained(
                model_choice,
                config=config,
                #max_seq_len=max_token_length,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_auth_token=True
            )
            embedding_size = 4096
            self.encoderModel = model_encoding.transformer

          elif model_choice in ['mosaicml/mpt-1b-redpajama-200b']:

            #model_encoding = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True, 
            #                                                      torch_dtype=torch.bfloat16) #attn_impl='triton', 
            model_encoding = MptForSequenceClassification.from_pretrained("mosaicml/mpt-1b-redpajama-200b", trust_remote_code=True)
            embedding_size = 2048
            self.encoderModel = model_encoding
          
          elif model_choice in ["google/t5-large-lm-adapt", "google/t5-xl-lm-adapt"]:

            model_encoding = AutoModelForSequenceClassification.from_pretrained(model_choice)
            #model_encoding = AutoModel.from_pretrained(model_choice, torch_dtype=torch.bfloat16)
            embedding_size = 1024
            self.encoderModel = model_encoding#.transformer

          elif model_choice in ["roberta-large", "microsoft/deberta-v3-large"]:

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1024
            self.encoderModel = model_encoding

          elif model_choice in ["microsoft/deberta-v2-xlarge", "microsoft/deberta-v2-xxlarge"]:

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 1536
            self.encoderModel = model_encoding
          
          else:

            model_encoding = AutoModel.from_pretrained(model_choice)
            embedding_size = 768
            self.encoderModel = model_encoding



          if frozen == True:
            print("Freezing the model parameters")
            for param in self.encoderModel.parameters():
                param.requires_grad = False



          if frozen_layer_count > 0:

            if model_choice in ["t5-3b", "google/t5-large-lm-adapt"]:

                #print(self.encoderModel.__dict__)

                print("Freezing T5-3b")
                print("Number of Layers: " + str(len(self.encoderModel.encoder.block)))

                for parameter in self.encoderModel.encoder.embed_tokens.parameters():
                    parameter.requires_grad = False

                for i, m in enumerate(self.encoderModel.encoder.block):        
                    #Only un-freeze the last n transformer blocks
                    if i+1 > 24 - frozen_layer_count:
                        print(str(i) + " Layer")
                        for parameter in m.parameters():
                            parameter.requires_grad = True

            elif model_choice == "distilbert-base-uncased":

                #print(self.encoderModel.__dict__)
                print("Number of Layers: " + str(len(list(self.encoderModel.transformer.layer))))

                layers_to_freeze = self.encoderModel.transformer.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False

            else:

                print("Number of Layers: " + str(len(list(self.encoderModel.encoder.layer))))

                layers_to_freeze = self.encoderModel.encoder.layer[:frozen_layer_count]
                for module in layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False



          
          if frozen_embeddings == True:

            if model_choice != "google/t5-large-lm-adapt":

                print("Frozen Embeddings Layer")
                #print(self.encoderModel.__dict__)
                for param in self.encoderModel.embeddings.parameters():
                    param.requires_grad = False





          ### New layers:

          self.classifier = nn.Sequential(nn.Linear(embedding_size, 256), #, dtype=torch.bfloat16
                                          nn.Linear(256, number_of_labels))


          #self.linear1 = nn.Linear(embedding_size, 256)
          #self.linear2 = nn.Linear(256, number_of_labels)

          self.embedding_size = embedding_size
          self.average_hidden_state = average_hidden_state

          #print(self.encoderModel.__dict__)


          

    def forward(self, ids, mask, labels=None, decoder_input_ids=None):
          
        if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt", "mosaicml/mpt-1b-redpajama-200b"]:
            total_output = self.encoderModel(input_ids=ids, attention_mask=mask) #labels=labels
            #print("total_output")
            #print(total_output.keys)
            #assert False
            return total_output['logits']
        else:
            total_output = self.encoderModel(ids, attention_mask=mask)

            #print("total_output")
            #print(total_output.keys())
            #print(total_output['logits'].shape)

            sequence_output = total_output['last_hidden_state']

            #linear1_output = self.linear1(sequence_output[:,0,:].view(-1, self.embedding_size))
            #linear2_output = self.linear2(linear1_output)

            last_hidden_state_formatted = sequence_output[:,0,:].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)

            return linear2_output

############################################################

def combine_query_document(query: str, document: str, answer=None):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")[:512])

    if len(query.split(" ")) > 100:
        query = (" ").join(query.split(" ")[:30])

    if answer is None:
        return query + " | " + cleaned_document
    else:
        return query + " | " + cleaned_document + " | " + answer

def tokenize_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################

def prepare_dataset_for_evaluation(dataframe, label_column: str, text_column: str):
    test_set_text = [dataframe.iloc[i][text_column] for i in range(len(dataframe))]
    test_set_label = [dataframe.iloc[i][label_column] for i in range(len(dataframe))]

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    classification_dataset = datasets.DatasetDict({'test' : test_dataset_arrow})
    tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)
    return eval_dataloader

############################################################

# Context Relevance
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v4.tsv/2e-05_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v4.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v4.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/google-t5-large-lm-adapt/datasets-synthetic_queries_v4.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v5.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/mosaicml-mpt-1b-redpajama-200b/datasets-synthetic_queries_v5.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v5.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/mosaicml-mpt-7b/datasets-synthetic_queries_v6.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v8.1.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v8.tsv/1e-05_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v15.tsv/5e-06_0_False_1True_False.pt"
model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-hotpotqa_synthetic_queries_v1.tsv/5e-06_0_False_1True_False.pt"

# Answer Faithfulness
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-synthetic_queries_v8.tsv/5e-06_0_False_1True_False.pt"
#model_checkpoint = "checkpoints/microsoft-deberta-v3-large/datasets-hotpotqa_synthetic_queries_v1.tsv/5e-06_0_False_1True_False_Answer_Faithfulness_Label.pt"

############################################################

#test_set_selection = "../datasets_v2/LLM_Judge_Test_Set.tsv"
#test_set_selection = "../datasets_v2/LLM_Judge_Test_Set_v3_Filtered.tsv"
#test_set_selection = "../datasets_v2/LLM_Judge_Test_Set_V4_Reannotated.tsv"
#test_set_selection = "../../datasets/synthetic_queries_v10.tsv"
#test_set_selection = "../datasets_v2/nq_reformatted_validation_with_negatives.tsv"
test_set_selection = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives.tsv"

ratio_sampled = 0.05

test_set = pd.read_csv(test_set_selection, sep="\t")
text_column = 'concat_text'
label_column = "Context_Relevance_Label" #"Answer_Faithfulness_Label" #"Context_Relevance_Label"

test_set['synthetic_query'] = test_set['Query']
test_set['document'] = test_set['Document']
test_set['synthetic_answer'] = test_set['Answer']
#test_set = test_set[:100]

conversion_dict = {"Yes": 1, "No": 0}
test_set = test_set[test_set[label_column].notna()]
#test_set[label_column] = test_set[label_column].apply(lambda x: conversion_dict[x])
if "Context" in label_column:
    test_set[text_column] = [combine_query_document(test_set.iloc[i]['synthetic_query'], test_set.iloc[i]['document']) for i in range(len(test_set))]
else:
    test_set[text_column] = [combine_query_document(test_set.iloc[i]['synthetic_query'], test_set.iloc[i]['document'], test_set.iloc[i]['synthetic_answer']) for i in range(len(test_set))]

print("Example Text")
print(test_set.iloc[10][text_column])

############################################################

assigned_batch_size = 1
number_of_labels = 2

model_choice = "microsoft/deberta-v3-large"
max_token_length = 2048
tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length)

#model_choice = "google/t5-large-lm-adapt"
#max_token_length = 2048
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length)

#model_choice = "mosaicml/mpt-1b-redpajama-200b" #"mosaicml/mpt-7b" #"mosaicml/mpt-1b-redpajama-200b" #"mosaicml/mpt-7b-instruct"
#max_token_length = 2048
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length, padding="max_length", truncation=True)
#tokenizer.pad_token = tokenizer.eos_token

#model_choice = "mosaicml/mpt-7b" #"mosaicml/mpt-1b-redpajama-200b" #"mosaicml/mpt-7b-instruct"
#max_token_length = 2048
#tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length, padding="max_length", truncation=True)
#tokenizer.pad_token = tokenizer.eos_token

device = "cuda:0"
device = torch.device(device)

############################################################

model = CustomBERTModel(number_of_labels, model_choice, True, False, 0, False, False)
model.to(device)

############################################################

eval_dataloader = prepare_dataset_for_evaluation(test_set, label_column, text_column)

print("Loading the Best Model")
model.load_state_dict(torch.load(model_checkpoint))

############################################################

print("Beginning Evaluation")

metric = load_metric("accuracy")
#model.eval()

total_predictions = torch.FloatTensor([]).to(device)
total_references = torch.FloatTensor([]).to(device)
total_logits = torch.FloatTensor([]).to(device)

inference_start = time.time()

#progress_bar = tqdm(range(len(eval_dataloader)))
#for batch in eval_dataloader:

progress_bar = tqdm(range(len(eval_dataloader)))
for batch in eval_dataloader:

    with torch.no_grad():

        if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
        else:
            new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

        if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
            new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device)

        outputs = model(**new_batch)

        #labels = batch['labels'].to(device)

        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

        total_predictions = torch.cat((total_predictions, predictions), 0)
        total_references = torch.cat((total_references, batch['labels'].to(device)), 0)
        total_logits = torch.cat((total_logits, logits), 0)

        progress_bar.update(1)



    inference_end = time.time()
    total_inference_time = inference_end - inference_start

    ############################################################

print("--------------------------")
print("Predictions Shapes")
print(total_predictions.shape)
print(total_references.shape)

""" for i in range(total_predictions.shape[0]):
    if total_predictions[i] != total_references[i]:
        print("Question")
        print(test_set.iloc[i]['Question'])
        print("Document")
        print(test_set.iloc[i]['Document'])
        print("Reference Label")
        print(total_references[i])
        print("Prediction Label")
        print(total_predictions[i])
        print("Logits")
        print(total_logits[i])
        print("---------------------------------------------\n\n") """


results = metric.compute(references=total_references, predictions=total_predictions)
print("Accuracy for Test Set: " + str(results['accuracy']))

f_1_metric = load_metric("f1")
macro_f_1_results = f_1_metric.compute(average='macro', references=total_references, predictions=total_predictions)
print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
micro_f_1_results = f_1_metric.compute(average='micro', references=total_references, predictions=total_predictions)
print("Micro F1 for Test Set: " + str(micro_f_1_results['f1'] * 100))

print("-------------------------------------------------------")
print("Positive Reference Label Count")
print(total_references.tolist().count(1))
print("Negative Reference Label Count")
print(total_references.tolist().count(0))

print("-------------------------------------------------------")
print("Positive Prediction Label Count")
print(total_predictions.tolist().count(1))
print("Negative Prediction Label Count")
print(total_predictions.tolist().count(0))

########################################################################

test_set[label_column + "_Model_Predictions"] = total_predictions.tolist()

test_set_sample = test_set.sample(n=int(len(test_set) * ratio_sampled), random_state=43)
test_set_non_sample = test_set.drop(test_set_sample.index)

test_set_selection = test_set_selection.replace(".tsv", "_" + label_column + "_with_Model_Predictions_Sampled.tsv")
test_set_sample.to_csv(test_set_selection, sep="\t", index=False)
print("Saved test set sampled predictions to: " + test_set_selection)

test_set_selection = test_set_selection.replace("_Sampled.tsv", "_NonSampled.tsv")
test_set_non_sample.to_csv(test_set_selection, sep="\t", index=False)
print("Saved test set non-sampled predictions to: " + test_set_selection)

