
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MptForSequenceClassification
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

from torch.distributed.pipeline.sync import Pipe
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '29500'
#torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

#################################################

def instruction_finetune_v3(checkpoint_path: str, model_name: str, device, dataset):
    
    tqdm.pandas(desc="Generating instructions formatting...", total=dataset.shape[0])
    dataset["content_relevance_text"] = dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x["synthetic_query"], x["document"], x['Label']), axis=1)
    dataset = Dataset.from_pandas(dataset)

    ##################################################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    max_token_length = 0
    for i in range(len(dataset)):
        input_ids = tokenizer.encode(dataset[i]["content_relevance_text"], return_tensors='pt')
        max_token_length = max(len(input_ids[0]), max_token_length)

    print("max_token_length")
    print(max_token_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_token_length,
                                              padding='max_length', truncation=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ##################################################################################################

    output_dir = "./results/" + checkpoint_path

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    config.max_seq_len = max_token_length
    config.init_device = device #"cuda:0" #'meta'

    ##################################################################################################

    peft_config = LoraConfig(
        #lora_alpha=16, lora_dropout=0.1, r=64, task_type="SEQ_CLS",
        lora_alpha=8, lora_dropout=0.1, r=32, task_type="SEQ_CLS",
    )

    model = MptForSequenceClassification.from_pretrained(model_name, max_seq_len=max_token_length, 
                                                         num_labels=2, torch_dtype=torch.bfloat16)
    

    modules = []
    modules.append(model.transformer.wte.cuda(0))
    for block, count in zip(model.transformer.blocks[:16], range(16)):
        modules.append(block.cuda(0))

    for block, count in zip(model.transformer.blocks[16:], range(16, len(model.transformer.blocks))):
        modules.append(block.cuda(1))

    modules.append(model.transformer.norm_f.cuda(1))
    modules.append(model.score.cuda(1))
    model_parallelized = nn.Sequential(*modules)

    model_pipe = Pipe(model_parallelized, chunks=8)
    model = model_pipe

    print("Model Parallelization complete!")


    #model = get_peft_model(model, peft_config)
    #print("Model Training Overview")
    #model.print_trainable_parameters()
    #model.to(device)

    num_epochs = 1
    selected_batch_size = 1
    gradient_accumulation_multiplier = 32
    learning_rate = 2e-4
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

    new_labels = []
    for i in tqdm(range(len(dataset))):
        if dataset[i]['Label'] == "Yes":
            new_labels.append(1)
        elif dataset[i]['Label'] == "No":
            new_labels.append(0)
    dataset = dataset.add_column('label', new_labels)

    classification_dataset = DatasetDict({'train' : dataset})
    classification_dataset = classification_dataset.map(tokenize_function, batched=True)

    tokenized_datasets = classification_dataset.remove_columns(['content_relevance_text', '__index_level_0__', 'document', 
                                                                'synthetic_query', 'filtered_questions', 'Label'])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=selected_batch_size)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=number_of_warmup_steps, num_training_steps=num_training_steps
    )

    ##################################################################################################

    #from accelerate import Accelerator
    #accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_multiplier)
    #train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, model, optimizer, lr_scheduler)

    ##################################################################################################

    gradient_accumulation_count = 0
    progress_bar = tqdm(range(len(train_dataloader)))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            #outputs = model(**batch).logits
            #loss = criterion(outputs, batch['labels'].to(device))
            
            #new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}#, 
                         #'labels': batch['labels'].to(device)}
            logits = model(batch['input_ids']).logits
            #print("logits")
            #print(logits)
            loss = criterion(logits, batch['labels'].to(device))
            loss.backward()
            #accelerator.backward(loss)

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