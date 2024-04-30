

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
from transformers import get_scheduler, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification #MptForSequenceClassification

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
import argparse

from tqdm import tqdm, tqdm_pandas

import warnings
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.")
#############################################################

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
        try:
            return query + " | " + cleaned_document + " | " + answer
        except:
            breakpoint()
            print("Error with combine_query_document")
            print("Query: " + str(query))
            print("Cleaned Document: " + str(cleaned_document))
            print("Answer: " + str(answer))
            return str(query) + " | " + str(cleaned_document) + " | " + str(answer)

def format_text_for_fine_tuning_content_relevance_sequence_classification(question: str, document: str):
    instruction = "You are an expert judge for evaluating question answering systems. "
    instruction += "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. \n\n"

    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")[:512])

    instruction += "### Instruction:\n"
    instruction += "Question: " + question + "\n"
    instruction += "Document: " + cleaned_document + "\n"
    instruction += "### Response:\n"

    return instruction

#############################################################

random_state = 42

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice):
          self.model_choice = model_choice
          super(CustomBERTModel, self).__init__()
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

          #########################################

          self.classifier = nn.Sequential(nn.Linear(embedding_size, 256), nn.Linear(256, number_of_labels))
          self.embedding_size = embedding_size

########################################################

    def forward(self, ids, mask, labels=None, decoder_input_ids=None):
        model_choice = self.model_choice # Possibly change this to self.model_choice = model_choice
        if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt", "mosaicml/mpt-1b-redpajama-200b"]:
            total_output = self.encoderModel(input_ids=ids, attention_mask=mask) #labels=labels
            return total_output['logits']
        else:
            total_output = self.encoderModel(ids, attention_mask=mask)
            sequence_output = total_output['last_hidden_state']

            last_hidden_state_formatted = sequence_output[:,0,:].view(-1, self.embedding_size)
            linear2_output = self.classifier(last_hidden_state_formatted)

            return linear2_output

############################################################

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) #.input_ids

########################################################

def checkpoints(classification_dataset, model_choice):
    checkpoints_folder_path = "checkpoints/"
    if not os.path.isdir(checkpoints_folder_path):
        os.mkdir(checkpoints_folder_path)

    dataset_folder_path = "checkpoints/" + model_choice.replace("/", "-")
    if not os.path.isdir(dataset_folder_path):
        print("Creating folder: " + dataset_folder_path)
        os.mkdir(dataset_folder_path)

    for dataset in classification_datasets:
        try:
            os.mkdir(dataset_folder_path + "/" + dataset.replace("../", "").replace("/", "-"))
        except:
            print("Already exists")
            print(dataset_folder_path + "/" + dataset.replace("../", "").replace("/", "-"))

########################################################

def load_model(model_choice):
    max_token_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length)
    
    return tokenizer, max_token_length

########################################################

def prepare_and_clean_data(dataset, learning_rate_choices, chosen_learning_rate, model_choice, 
                number_of_runs, validation_set_scoring, 
                label_column, validation_set, patience_value, 
                num_epochs, num_warmup_steps, gradient_accumulation_multiplier, 
                assigned_batch_size, tokenizer): 

    print("--------------------------------------------------------------------------")
    print("Starting new learning rate: " + str(chosen_learning_rate))
    print("--------------------------------------------------------------------------")

    import datetime

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    parent_dir = "checkpoints/" + model_choice.replace("/", "-")

    if not os.path.exists(parent_dir):
        print(f"Creating parent checkpoint directory: {parent_dir}")
        print("--------------------------------------------------------------------------")
        os.makedirs(parent_dir)

    checkpoint_path = "checkpoints/" + model_choice.replace("/", "-") + "/" + label_column + "_" + str(validation_set.split("/")[-1].replace(".tsv", "")) + "_" + current_datetime + ".pt"

    # checkpoint_path = "checkpoints/" + model_choice.replace("/", "-") + "/" + dataset.replace("../", "").replace("/", "-") + "/" + str(chosen_learning_rate) + "_"
    # checkpoint_path += str(number_of_runs) + "_" + str(validation_set_scoring) + "_" + label_column + "_" + str(validation_set.split("/")[-1].replace(".tsv", "")) + "_" + str(random_int) + ".pt"

    # checkpoint_path = "checkpoints/" + model_choice.replace("/", "-") + "/" + dataset.replace("../", "").replace("/", "-") + "/" + str(chosen_learning_rate) + "_"
    # checkpoint_path += str(number_of_runs) + "_" + str(validation_set_scoring) + "_" + label_column + "_" + str(validation_set.split("/")[-1].replace(".tsv", "")) + "_" + str(random_int) + ".pt"
    
    execution_start = time.time()

    print("Dataset: " + dataset)
    print("Model: " + model_choice)
    print("Test Set Selection: " + validation_set)
    print("Number of Runs: " + str(number_of_runs))
    print('Learning Rate: ' + str(chosen_learning_rate))
    print("Checkpoint Path: " + checkpoint_path)
    print("Patience: " + str(patience_value))
    print("Validation Set Choice: " + str(validation_set_scoring))
    print("Number of Epochs: " + str(num_epochs))
    print("Number of warmup steps: " + str(num_warmup_steps))
    print("--------------------------------------------------------------------------")

    return checkpoint_path, patience_value

########################################################

def analyze_and_report_data(dataset, label_column, tokenizer, max_token_length): 

    synth_queries = pd.read_csv(dataset, sep="\t")
    
    if "nq_reformatted" in dataset:
        synth_queries['synthetic_query'] = synth_queries['Query']
        synth_queries['generated_answer'] = synth_queries['Answer']
        synth_queries['document'] = synth_queries['Document']

    # print("Positive and Negative Label Count")
    # print(synth_queries[label_column].tolist().count("Yes"))
    # print(synth_queries[label_column].tolist().count("No"))
    # print(set(synth_queries[label_column].tolist()))
    
    synth_queries = synth_queries[synth_queries[label_column] != "NaN"]
    synth_queries = synth_queries[synth_queries["synthetic_query"].notna()]
    synth_queries = synth_queries[synth_queries["document"].notna()]
    synth_queries = synth_queries[synth_queries['generated_answer'].notna()]
    synth_queries = synth_queries[synth_queries[label_column].notna()]
    synth_queries = synth_queries.sample(n=len(synth_queries), random_state=42)
    #synth_queries = synth_queries[:40000]

    # print("Number of unique questions")
    # print(len(set(synth_queries['synthetic_query'].tolist())))
    # print("Positive and Negative Label Count")
    # print(synth_queries[label_column].tolist().count("Yes"))
    # print(synth_queries[label_column].tolist().count("No"))
    # print(set(synth_queries[label_column].tolist()))
    
    if "Context" in label_column:
        synth_queries["concat_text"] = [combine_query_document(synth_queries.iloc[i]['synthetic_query'], synth_queries.iloc[i]['document']) for i in range(len(synth_queries))]
    else:
        synth_queries["concat_text"] = [combine_query_document(synth_queries.iloc[i]['synthetic_query'], synth_queries.iloc[i]['document'], synth_queries.iloc[i]['generated_answer']) for i in range(len(synth_queries))]
    synth_queries['token_length'] = [len(tokenizer.encode(text, return_tensors='pt')[0]) for text in tqdm(synth_queries['concat_text'], desc="Tokenizing")]
    synth_queries = synth_queries.drop_duplicates(["concat_text"])

    # print("Max token length")
    # print(synth_queries['token_length'].max())
    # print("Total inputs over max token length set of " + str(max_token_length))
    # print(len(synth_queries[synth_queries['token_length'] > max_token_length]))

    synth_queries = synth_queries[synth_queries['token_length'] <= 2048]

    return synth_queries

########################################################

def transform_data(synth_queries, validation_set, label_column):

    train_df = synth_queries

    test_set = pd.read_csv(validation_set, sep="\t")
    test_set['Question'] = test_set['Query']
    test_set['Document'] = test_set['Document'].str.strip()
    test_set = test_set[test_set["Document"].str.len() > 100]
    test_set = test_set[test_set[label_column].notna()]

    train_df['document'] = train_df['document'].astype(str).str.strip()
    train_df = train_df[train_df["document"].str.len() > 100]
    train_df = train_df[train_df[label_column].notna()]

    if "Context" in label_column:
        test_set['concat_text'] = [combine_query_document(test_set.iloc[i]['Question'], test_set.iloc[i]['Document']) for i in range(len(test_set))]
    else:
        test_set['concat_text'] = [combine_query_document(test_set.iloc[i]['Question'], test_set.iloc[i]['Document'], test_set.iloc[i]['Answer']) for i in range(len(test_set))]

    train_df = train_df.drop_duplicates(["concat_text"])
    test_set = test_set.drop_duplicates(["concat_text"])

    if "Faith" in label_column:
        print("Refining data for Answer_Faithfulness classification!")
        train_df = train_df[train_df["Context_Relevance_Label"].notna()]
        train_df = train_df[train_df["Answer_Faithfulness_Label"].notna()]
        error_strings = ['answer', 'contrad', 'false', 'information', 'unanswer', 'Answer', 'Contrad', 'False', 'Information', 'Unanswer']
        train_df['generated_answer'] = train_df['generated_answer'].astype(str)
        train_df = train_df[~train_df['generated_answer'].str.contains('|'.join(error_strings))]

    return train_df, test_set

########################################################

def split_dataset(train_df, dataset, test_set, label_column):
    conversion_dict = {"Yes": 1, "No": 0}
    train_set_text = [train_df.iloc[i]['concat_text'] for i in range(len(train_df))]
    if "nq_reformatted" not in dataset:
        train_set_label = [conversion_dict[train_df.iloc[i][label_column]] for i in range(len(train_df))]
    else:
        train_set_label = [int(train_df.iloc[i][label_column]) for i in range(len(train_df))]
    
    #dev_set_text = [validation_df.iloc[i]['concat_text'] for i in range(len(validation_df))]
    #dev_set_label = [conversion_dict[validation_df.iloc[i]['Label']] for i in range(len(validation_df))]
    dev_set_text = [test_set.iloc[i]['concat_text'] for i in range(len(test_set))]
    dev_set_label = [int(test_set.iloc[i][label_column]) for i in range(len(test_set))]
    
    #test_set_text = [test_df.iloc[i]['concat_text'] for i in range(len(test_df))]
    #test_set_label = [conversion_dict[test_df.iloc[i]['Label']] for i in range(len(test_df))]
    test_set_text = [test_set.iloc[i]['concat_text'] for i in range(len(test_set))]
    test_set_label = [int(test_set.iloc[i][label_column]) for i in range(len(test_set))]

    print('---------------------------------------------------')
    # print("Lengths of train, dev, and test")
    # print(len(train_set_text))
    # print(len(dev_set_text))
    # print(len(test_set_text))
    # print("Training example")
    # print(train_set_text[100])
    # print('---------------------------------------------------')
    # print(train_set_label[100])
    # print('---------------------------------------------------')

    labels_list = sorted(list(set(train_set_label + dev_set_label + test_set_label)))

    # print("Label List")
    # print(labels_list)
    # print(set(train_set_label))
    # print(set(dev_set_label))
    # print(set(test_set_label))

    return train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, test_set_label, labels_list

    ############################################################

def prepare_dataset(validation_set_scoring, train_set_label, train_set_text, dev_set_label, dev_set_text):
    if validation_set_scoring == True:

        training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:100]
        training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
        training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

        validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:100]
        validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
        validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

        test_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})
        test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
        test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    else:

        training_dataset_pandas = pd.DataFrame({'label': train_set_label, 'text': train_set_text})#[:1000]
        training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
        training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

        validation_dataset_pandas = pd.DataFrame({'label': dev_set_label, 'text': dev_set_text})#[:1000]
        validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
        validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

        test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
        test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
        test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    return training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas


def initalize_dataset_for_tokenization(tokenizer, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow):

    classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
                                    'validation': validation_dataset_arrow, 
                                    'test' : test_dataset_arrow})
    
    tokenized_datasets = classification_dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

    ############################################################

def train_and_evaluate_model(number_of_runs, tokenized_datasets, assigned_batch_size, train_set_label, model_choice, chosen_learning_rate, 
                            device, checkpoint_path, patience_value, num_epochs, num_warmup_steps, gradient_accumulation_multiplier): 

    micro_averages = []
    macro_averages = []
    inference_times = []

    for i in range(0, number_of_runs):

        run_start = time.time()

        print("Loading Model")

        train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
        validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
        eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

        # print("Number of labels: " + str(len(set(train_set_label))))

        ############################################################

        model = CustomBERTModel(len(set(train_set_label)), model_choice)

        model.to(device)

        ############################################################

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=chosen_learning_rate)

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        ############################################################

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        # import EarlyStopping
        from ares.LLM_as_a_Judge_Adaptation.pytorchtools import EarlyStopping

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience_value, verbose=True, path=checkpoint_path)
        #early_stopping = EarlyStopping(patience=10, verbose=True)

        print("Checkpoint Path: " + checkpoint_path)

        print("Beginning Training")

        total_epochs_performed = 0

        for epoch in range(num_epochs):

            total_epochs_performed += 1

            print("Current Epoch: " + str(epoch))

            progress_bar = tqdm(range(len(train_dataloader)))

            gradient_accumulation_count = 0

            model.train()
            for batch in train_dataloader:

                if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
                else:
                    new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                outputs = model(**new_batch)

                loss = criterion(outputs, batch['labels'].to(device))

                loss.backward()

                gradient_accumulation_count += 1
                if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                progress_bar.update(1)
                train_losses.append(loss.item())

            ######################################################

            progress_bar = tqdm(range(len(validation_dataloader)))

            model.eval()
            for batch in validation_dataloader:

                with torch.no_grad():
                
                    if model_choice in ["mosaicml/mpt-1b-redpajama-200b"]:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].bool().to(device)}
                    else:
                        new_batch = {'ids': batch['input_ids'].to(device), 'mask': batch['attention_mask'].to(device)}

                    if model_choice in ["t5-small", "google/t5-xl-lm-adapt", "google/t5-large-lm-adapt"]:
                        new_batch['decoder_input_ids'] = batch['labels'].reshape(batch['labels'].shape[0], 1).to(device), 

                    outputs = model(**new_batch)

                    loss = criterion(outputs, batch['labels'].to(device))
                    progress_bar.update(1)

                    valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(num_epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')
            
            print(print_msg)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        return model, avg_train_losses, avg_valid_losses, eval_dataloader, inference_times

        ############################################################

def evaluate_model(model, model_choice, checkpoint_path, device, eval_dataloader, inference_times):

        print("Loading the Best Model")

        model.load_state_dict(torch.load(checkpoint_path))

        ############################################################

        print("Beginning Evaluation")

        metric = load_metric("accuracy")

        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)

        inference_start = time.time()

        ############################################################

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

                logits = outputs
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch['labels'].to(device))

                total_predictions = torch.cat((total_predictions, predictions), 0)
                total_references = torch.cat((total_references, batch['labels'].to(device)), 0)

                progress_bar.update(1)

        ############################################################

        inference_end = time.time()
        total_inference_time = inference_end - inference_start
        inference_times.append(total_inference_time)

        return total_predictions, total_references, metric

        ############################################################

def print_and_save_model(total_predictions, total_references, checkpoint_path, metric):
        print("--------------------------")
        print("Predictions and Reference Shapes")
        print(total_predictions.shape)
        print(total_references.shape)

        results = metric.compute(references=total_references, predictions=total_predictions)
        print("Accuracy for Test Set: " + str(results['accuracy']))

        f_1_metric = load_metric("f1", trust_remote_code=True)
        macro_f_1_results = f_1_metric.compute(average='macro', references=total_references, predictions=total_predictions)
        print("Macro F1 for Test Set: " + str(macro_f_1_results['f1'] * 100))
        micro_f_1_results = f_1_metric.compute(average='micro', references=total_references, predictions=total_predictions)
        print("Micro F1 for Test Set: " + str(micro_f_1_results['f1']  * 100))

        print("Positive / Negative Reference Ratio: " + str(round(total_references.tolist().count(1) / len(total_references.tolist()), 3)))

        print("Saved classification checkpoint to: " + str(checkpoint_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--classification_dataset", type=str, required=True)
    parser.add_argument("--test_set_selection", type=str, required=True)
    parser.add_argument("--label_column", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--patience_value", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)

    args = parser.parse_args()

    classification_datasets = [args.classification_dataset] # 
    test_set_selection = args.test_set_selection #
    label_column = args.label_column # 
    num_epochs = args.num_epochs # 
    patience_value = args.patience_value # 
    learning_rate_choices = [args.learning_rate] # 

    ### Instructions

    device = "cuda:0"
    device = torch.device(device)

    validation_set_scoring = True # 
    assigned_batch_size = 1
    gradient_accumulation_multiplier = 32

    number_of_runs = 1 # 
    num_warmup_steps = 100 #

    ############################################################
