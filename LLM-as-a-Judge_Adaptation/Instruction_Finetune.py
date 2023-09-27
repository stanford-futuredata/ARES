
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

from datasets import load_dataset, Dataset, load_from_disk
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
import re

#################################################

def format_text_for_fine_tuning_content_relevance(question: str, document: str, label: str):
    instruction = "You are an expert judge for evaluating question answering systems. "
    instruction += "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
    instruction += "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
    instruction += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
    instruction += 'Do not provide any additional explanation for your decision. Solely output "[[Yes]]" or "[[No]]"\n\n'

    #instruction += "Please explain why the provided document is relevant or not relevant for answering the question.\n"
    #instruction += 'After providing your explanation, output your final verdict by strictly following this format: "[[Yes]]" if the document provided is sufficient for answering the question and "[[No]]" if the document provided is not sufficient for answering the question.'
    #and "[[Unknown]]" if you are not sure if the document is sufficient or not to answer the question.
    
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document)

    instruction += "### Instruction:\n"
    instruction += "Question: " + question + "\n"
    instruction += "Document: " + cleaned_document + "\n"
    instruction += "### Response:\n"
    if label != "" and len(label) > 1:
        instruction += "[[" + label + "]]"
    return instruction

def format_text_for_fine_tuning_answer_relevance(question: str, document: str, answer: str, label: str):
    instruction = "Below is an instruction that describes a question-answering task...\n\n"
    instruction += "### Instruction:\n"
    instruction += "Given the following question, context, and answer, analyze the provided answer and determine whether it is sufficient for addressing the question. Your decision should state Yes or No. Do not provide any additional explanation for your decision.\n\n"
    instruction += "### Input:\n"
    instruction += "Question: " + question + "\n"
    instruction += "Context: " + re.sub(r'\n+', '\n', document) + "\n"
    instruction += "Answer: " + re.sub(r'\n+', '\n', answer) + "\n"
    instruction += "### Response:\n"
    instruction += label
    return instruction

def format_text_for_fine_tuning_answer_faithfulness(question: str, document: str, answer: str, label: str):
    instruction = "Below is an instruction that describes a question-answering task...\n\n"
    instruction += "### Instruction:\n"
    instruction += "Given the following question, context, and answer, determine whether the answer is supported by information provided in the context. Your decision should state Yes or No. Do not provide any additional explanation for your decision.\n\n"
    instruction += "### Input:\n"
    instruction += "Question: " + question + "\n"
    instruction += "Context: " + re.sub(r'\n+', '\n', document) + "\n"
    instruction += "Answer: " + re.sub(r'\n+', '\n', answer) + "\n"
    instruction += "### Response:\n"
    instruction += label
    return instruction


####################################################################################


def instruction_finetune(checkpoint_path: str, model_name: str, device, dataset, eval_dataset=None):
    
    tqdm.pandas(desc="Generating instructions formatting...", total=dataset.shape[0])
    dataset["content_relevance_text"] = dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x["synthetic_query"], x["document"], x['Label']), axis=1)
    dataset = Dataset.from_pandas(dataset)

    if eval_dataset is not None:
        tqdm.pandas(desc="Generating instructions formatting...", total=eval_dataset.shape[0])
        eval_dataset["content_relevance_text"] = eval_dataset.progress_apply(lambda x: format_text_for_fine_tuning_content_relevance(x["synthetic_query"], x["document"], x['Label']), axis=1)
        eval_dataset = Dataset.from_pandas(eval_dataset)

    ##################################################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token

    max_token_length = 0
    for i in range(len(dataset)):
        input_ids = tokenizer.encode(dataset[i]["content_relevance_text"], return_tensors='pt')
        max_token_length = max(len(input_ids[0]), max_token_length)
    if eval_dataset is not None:
        for i in range(len(eval_dataset)):
            input_ids = tokenizer.encode(eval_dataset[i]["content_relevance_text"], return_tensors='pt')
            max_token_length = max(len(input_ids[0]), max_token_length)

    print("max_token_length")
    print(max_token_length)

    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_token_length, 
                                              padding='max_length', truncation=True, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token

    ##################################################################################################

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, task_type="CAUSAL_LM", #bias="none",
        #task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )

    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.float16,
    #)

    ##################################################################################################

    output_dir = "./results/" + checkpoint_path

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    config.max_seq_len = max_token_length
    config.init_device = device #"cuda:0" #'meta'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True,
        use_auth_token=True
    )

    model = get_peft_model(model, peft_config)
    print("Model Training Overview")
    model.print_trainable_parameters()

    ##################################################################################################

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=100,
        max_steps=len(dataset),
        save_steps=15000,
        num_train_epochs=3 #3
    )

    print("Settings for Instruction-Finetuning Run")
    print(training_args)
    print(peft_config)

    #if eval_dataset != None:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="content_relevance_text",
        max_seq_length=max_token_length,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Dataset before training")
    print(len(dataset))
    print(dataset)
    print(dataset[0]['content_relevance_text'])

    print("Starting training...")
    trainer.train()
    print("Completed training!")

    trainer.model.save_pretrained(output_dir)
    print("Saved checkpoint to: " + output_dir)

    return model

