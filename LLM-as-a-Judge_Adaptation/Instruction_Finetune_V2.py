

import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MptForSequenceClassification, AutoModelForSequenceClassification
import numpy as np

from torch.optim import AdamW, Adam
from transformers import get_scheduler
import torch.nn as nn

from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
import re

from torch.utils.data import DataLoader

from Instruction_Finetune import format_text_for_fine_tuning_content_relevance

#################################################

def combine_query_document(query: str, document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" "))#[:512])
    return query + " | " + cleaned_document

def instruction_finetune_v2(checkpoint_path: str, model_name: str, device, dataset):
    
    tqdm.pandas(desc="Generating instructions formatting...", total=dataset.shape[0])
    #dataset["content_relevance_text"] = dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x["synthetic_query"], x["document"], x['Label']), axis=1)
    dataset["content_relevance_text"] = dataset.progress_apply(lambda x: combine_query_document(x["synthetic_query"], x["document"]), axis=1)
    dataset = Dataset.from_pandas(dataset)

    







    
    dataset = load_dataset("glue", "qnli")['train']
    dataset = dataset.to_pandas()
    dataset["content_relevance_text"] = dataset.progress_apply(lambda x: x["question"] + " | " + x["sentence"], axis=1)
    dataset = dataset.sample(n=30000, random_state=42)
    dataset = Dataset.from_pandas(dataset)














    print("context relevance example")
    print(dataset[1555]["content_relevance_text"])

    ##################################################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    max_token_length = 0
    for i in range(len(dataset)):
        input_ids = tokenizer.encode(dataset[i]["content_relevance_text"], return_tensors='pt')
        max_token_length = max(len(input_ids[0]), max_token_length)

    print("max_token_length")
    print(max_token_length)
    max_token_length = 512
    print("Setting max token length to: " + str(max_token_length))
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_token_length,
                                              padding='max_length', truncation=True, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token

    ##################################################################################################

    #peft_config = LoraConfig(
    #    #lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM",
    #    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    #)

    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.float16,
    #)

    ##################################################################################################

    output_dir = "./results/" + checkpoint_path

    #config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    #config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    #config.max_seq_len = max_token_length
    #config.init_device = device #"cuda:0" #'meta'

    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    config=config,
        #quantization_config=bnb_config,
    #    torch_dtype=torch.bfloat16, # Load model weights in bfloat16
    #    trust_remote_code=True,
    #    use_auth_token=True
    #)

    #model = get_peft_model(model, peft_config)
    #print("Model Training Overview")
    #model.print_trainable_parameters()

    ##################################################################################################

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, task_type="SEQ_CLS",
    )







    #max_token_length = 2048
    #model = MptForSequenceClassification.from_pretrained(model_name, max_seq_len=max_token_length, 
    #                                                     num_labels=2, torch_dtype=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.bfloat16)
    
    #model = get_peft_model(model, peft_config)
    #print("Model Training Overview")
    #model.print_trainable_parameters()
    model.to(device)










    num_epochs = 1
    selected_batch_size = 1
    gradient_accumulation_multiplier = 32
    learning_rate = 1e-5
    number_of_warmup_steps = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Settings for Instruction-Finetuning Run")
    print("model: " + str(model_name))
    print("learning_rate: " + str(learning_rate))
    print("gradient_accumulation_multiplier: " + str(gradient_accumulation_multiplier))
    print("selected_batch_size: " + str(selected_batch_size))
    print("num_epochs: " + str(num_epochs))
    print("number_of_warmup_steps: " + str(number_of_warmup_steps))
    print("Peft Config: " + str(peft_config))

    ##################################################################################################

    def tokenize_function(examples):
        tokenized_docs = tokenizer(examples["content_relevance_text"], max_length=max_token_length, 
                                   padding="max_length", truncation=True)
        for input_ids in tokenized_docs['input_ids']:
            assert len(input_ids) == max_token_length
        return tokenized_docs

    #new_labels = []
    #for i in tqdm(range(len(dataset))):
    #    if dataset[i]['Label'] == "Yes":
    #        new_labels.append(1)
    #    elif dataset[i]['Label'] == "No":
    #        new_labels.append(0)
    #dataset = dataset.add_column('label', new_labels)

    classification_dataset = DatasetDict({'train' : dataset})
    classification_dataset = classification_dataset.map(tokenize_function, batched=True)

    #tokenized_datasets = classification_dataset.remove_columns(['content_relevance_text', '__index_level_0__', 'document', 
    #                                                            'synthetic_query', 'filtered_questions', 'Label'])
    tokenized_datasets = classification_dataset.remove_columns(['content_relevance_text', 'question', 'sentence', "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=selected_batch_size)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=number_of_warmup_steps, num_training_steps=num_training_steps
    )

    ##################################################################################################

    gradient_accumulation_count = 0
    model.train()
    for epoch in range(num_epochs):
        print("Epoch #" + str(epoch))
        progress_bar = tqdm(range(len(train_dataloader)))
        for batch in train_dataloader:

            #outputs = model(**batch).logits
            #loss = criterion(outputs, batch['labels'].to(device))
            
            new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}#, 
                         #'labels': batch['labels'].to(device)}
            logits = model(**new_batch).logits
            #print("logits")
            #print(logits)
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()

            gradient_accumulation_count += 1
            if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)

    print("Dataset before training")
    print(len(dataset))
    print(dataset)
    print(dataset[0]['content_relevance_text'])

    model.save_pretrained(output_dir)
    print("Saved checkpoint to: " + output_dir)

    return model