
from cProfile import label
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer
#import tensorflow as tf
import sys

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
import scipy.stats as stats
import argparse

import subprocess
import json

import openai

from tqdm import tqdm
tqdm.pandas()


from ares.RAG_Automatic_Evaluation.ppi import clt_iid, binomial_iid, pp_mean_iid_asymptotic
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import calculate_accuracy, few_shot_context_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring, few_shot_answer_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_togetherai, few_shot_answer_faithfulness_scoring_togetherai, few_shot_answer_relevance_scoring_togetherai
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_claude, few_shot_answer_faithfulness_scoring_claude, few_shot_answer_relevance_scoring_claude

#############################################################

random_state = 44

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)
os.environ["HUGGINGFACE_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
          
          self.encoderModel.eval()
          self.classifier = nn.Sequential(nn.Linear(embedding_size, 256), nn.Linear(256, number_of_labels))
          self.embedding_size = embedding_size


    def forward(self, ids, mask, labels=None, decoder_input_ids=None):
        model_choice = self.model_choice
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
            print("Error with combine_query_document")
            print("Query: " + str(query))
            print("Cleaned Document: " + str(cleaned_document))
            print("Answer: " + str(answer))
            #return str(query) + " | " + str(cleaned_document) + " | " + str(answer)
            return "Error"

def tokenize_function(tokenizer, examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)#.input_ids

############################################################

def prepare_dataset_for_evaluation(dataframe, label_column: str, text_column: str, assigned_batch_size, tokenizer):
    from datasets.utils.logging import disable_progress_bar
    disable_progress_bar()

    test_set_text = [dataframe.iloc[i][text_column] for i in range(len(dataframe))]
    test_set_label = [dataframe.iloc[i][label_column] for i in range(len(dataframe))]

    test_dataset_pandas = pd.DataFrame({'label': test_set_label, 'text': test_set_text})
    test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
    test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

    classification_dataset = datasets.DatasetDict({'test' : test_dataset_arrow})
    tokenized_datasets = classification_dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)
    return eval_dataloader

############################################################

def calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials):

    n_max = Y_labeled.shape[0]
    # ns = np.linspace(100,n_max,20).astype(int)
    ns = np.linspace(0, n_max, 20).astype(int)

    # Imputed-only estimate
    imputed_estimate = (Yhat_labeled.sum() + Yhat_unlabeled.sum()) / (Yhat_labeled.shape[0] + Yhat_unlabeled.shape[0])

    # Run prediction-powered inference and classical inference for many values of n
    ci = np.zeros((num_trials, ns.shape[0], 2))
    ci_classical = np.zeros((num_trials, ns.shape[0], 2))

    for j in tqdm(range(num_trials), desc="Trials"):  # Wrap the outer loop with tqdm for the progress bar
        for i, n in enumerate(ns):  # Iterate over ns with an index
            rand_idx = np.random.permutation(Y_labeled.shape[0])
            f = Yhat_labeled.astype(float)[rand_idx[:n]]
            y = Y_labeled.astype(float)[rand_idx[:n]]
            output = pp_mean_iid_asymptotic(y, f, Yhat_unlabeled, alpha)
            ci[j, i, :] = output
            # Classical interval
            try:
                ci_classical[j, i, :] = binomial_iid(n, alpha, y.mean())
            except:
                avg_ci_classical = None

    avg_ci = ci.mean(axis=0)[-1]

    try:
        ci_imputed = binomial_iid(Yhat_unlabeled.shape[0], alpha, imputed_estimate)
    except:
        ci_imputed = None
    try:
        avg_ci_classical = ci_classical.mean(axis=0)[-1]
    except:
        avg_ci_classical = None

    return avg_ci, avg_ci_classical, ci_imputed

######################################################################

def begin(evaluation_datasets, checkpoints, labels, GPT_scoring, few_shot_examples_filepath):
    print("--------------------------------------------------------")
    print("Evaluation Sets: " + str(evaluation_datasets))
    print("Checkpoints: " + str(checkpoints))
    print("Labels: "  + str(labels))
    print("--------------------------------------------------------")

    ######################################################################

    # if GPT_scoring:
    #     checkpoint = ["" for _ in range(len(labels))]

    if few_shot_examples_filepath is not None:
        few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
        # print("few_shot_examples")
        # print(len(few_shot_examples))
        # print(few_shot_examples.head())
    return few_shot_examples

####################################################################

def clean_document(document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" ")) #[:512]
    return cleaned_document

def clean_query(query: str):
    cleaned_query = query.replace("\n", " ").replace("\r"," ").replace("\t"," ").strip()
    return cleaned_query

#################################################
def filter_dataset(rag_type: str = "question_answering"):
    if rag_type == "question_answering": 
        context_relevance_system_prompt = "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    
        answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
        answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
        answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
        answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

        answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

    elif rag_type == "fact_checking": 
        context_relevance_system_prompt = "You are an expert fact-checking agent. "
        context_relevance_system_prompt += "Given the following statement and document, you must analyze the provided document and determine whether it is sufficient for determining the statement's factuality. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided statement's factuality. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document is not sufficient. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

        answer_faithfulness_system_prompt = "Given the following statement, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "

        answer_relevance_system_prompt = "Given the following statement, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided statement. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the statement and provides only correct information from the document for answering the statement. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given statement and "[[No]]" if the answer is not relevant for the given statement. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

    elif rag_type == "dialogue_agent":
        context_relevance_system_prompt = "You are an expert dialogue agent. "
        context_relevance_system_prompt += "Given the following dialogue and document, you must analyze the provided document and determine whether it is relevant for responding to the dialogue. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

        answer_faithfulness_system_prompt = "Given the following dialogue, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        
        answer_relevance_system_prompt = "Given the following dialogue, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided dialogue. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the dialogue and provides only correct information from the document for responding to the dialogue. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given dialogue and "[[No]]" if the answer is not relevant for the given dialogue. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    
    else: 
        sys.exit("Error: rag_type not specified, please specifiy in configuration with 'rag_type'")
    
    return context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt

def preprocess_data(test_set_selection, label_column, labels): 
    test_set = pd.read_csv(test_set_selection, sep="\t")
    text_column = 'concat_text'
    test_set = test_set[test_set[label_column].notna()]
    if "Context" in label_column:
        test_set[text_column] = [combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document']) for i in range(len(test_set))]
    else:
        test_set[text_column] = [combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document'], test_set.iloc[i]['Answer']) for i in range(len(test_set))]

    test_set = test_set[test_set[text_column] != "Error"]
    # print("Example Text for " + label_column + " Scoring")
    # print(test_set.iloc[10][text_column])
    if len(test_set) < 10:
        # print("Example Text for " + label_column + " Scoring")
        # print(test_set.iloc[10][text_column])
        raise ValueError("Insufficient Data: Dataset has fewer than 10 rows after filtering!")
    
    return test_set, text_column

        ############################################################

def togetherai_list_models(api_key):
    if not api_key:
        return []
    try:
        # Running the command to list models
        result = subprocess.run(['together', 'models', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        models = []
        for line in lines[3:]:
            if '|' in line:
                parts = line.split('|')
                if len(parts) > 1:
                    model_name = parts[1].strip()  
                    models.append(model_name)
        return models
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return ["N/A"]  
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ["N/A"]

def load_model(model_identifier, number_of_labels, checkpoint=None):
    check = False
    tokenizer = None
    api_key = os.getenv("TOGETHER_API_KEY")
    
    together_models = togetherai_list_models(api_key)
    if "gpt" in model_identifier:
        check = True
    elif "claude" in model_identifier: 
        check = True
    elif model_identifier in together_models: 
        check = True 
    else: 
        check = False
    
    if check == False:
        max_token_length = 2048
        tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_token_length)

    # Prepare device setup
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")

    if check is False: 
        # Initialize model
        model = CustomBERTModel(number_of_labels, model_identifier)
        model.to(device)
    else: 
        model = model_identifier

    # Load the model from a checkpoint if provided
    if checkpoint:
        checkpoint_dict = torch.load(checkpoint)
        if "encoderModel.embeddings.position_ids" in checkpoint_dict:
            del checkpoint_dict["encoderModel.embeddings.position_ids"]
        model.load_state_dict(checkpoint_dict)
        print("Loaded model from checkpoint:", checkpoint)
    else:
        print("Loaded model based on model identifier:", model_identifier)
    
    return model, tokenizer, device

    ############################################################

def evaluate_model(test_set, label_column, text_column, device, checkpoint, tokenizer, model, assigned_batch_size, model_choice, 
context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt, few_shot_examples_filepath, llm_judge): 

    metric = load_metric("accuracy")

    if checkpoint:
        total_predictions = torch.FloatTensor([]).to(device)
        total_references = torch.FloatTensor([]).to(device)
        total_logits = torch.FloatTensor([]).to(device)
        eval_dataloader = prepare_dataset_for_evaluation(test_set, label_column, text_column, assigned_batch_size, tokenizer)
        model.eval()
        with tqdm(eval_dataloader, desc="Evaluating", leave=False) as progress_bar:
            for batch in progress_bar:

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
                    total_logits = torch.cat((total_logits, logits), 0)

                    progress_bar.update(1)

    else:
        print("Performing Model scoring!")
        # print("Using gpt model: " + gpt_model)
        # if perform_zero_shot:
        #     # print("Using zero-shot approach")
        #     # print("Setting few-shot example to None...")
        #     few_shot_examples = None

        debug_mode = False

        few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
        # Few Shot Dataset Edge Check: Query ID 
        try:
                _ = few_shot_examples.iloc[0]['Query']
                query_id = "Query"
        except KeyError:
            try:
                _ = few_shot_examples.iloc[0]['Question']
                query_id = "Question"
            except KeyError:
                sys.exit("Both 'Query' and 'Question' keys are missing for the given row in few shot dataset.")
        # # Labeled Dataset Edge Check: Query ID 
        # try:
        #         _ = Y_labeled_dataset.iloc[0]['Query']
        #         query_labeled_id = "Query"
        # except KeyError:
        #     try:
        #         _ = Y_labeled_dataset.iloc[0]['Question']
        #         query_labeled_id = "Question"
        #     except KeyError:
        #         sys.exit("Both 'Query' and 'Question' keys are missing in labeled dataset.")

        if "Context_Relevance_Label" == label_column:
            # tqdm.pandas(desc="Generating context relevance scores...", total=test_set.shape[0])
            test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(x["Query"]), x["Document"], model, query_id, debug_mode, few_shot_examples), axis=1)
        elif "Answer_Faithfulness_Label" == label_column:
            # tqdm.pandas(desc="Generating answer faithfulness scores...", total=test_set.shape[0])
            test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], model, query_id, debug_mode, few_shot_examples), axis=1)
        if "Answer_Relevance_Label" == label_column:
            # tqdm.pandas(desc="Generating answer relevance scores...", total=test_set.shape[0])
            test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], model, query_id, debug_mode, few_shot_examples), axis=1)

        total_predictions = test_set[label_column.replace("_Label", "_Prediction")].to_numpy()
        total_references = test_set[label_column].to_numpy()
     
    results = metric.compute(references=total_references, predictions=total_predictions)

    return total_predictions, total_references, results, metric

        ########################################################################
def post_process_predictions(checkpoint, test_set, label_column, total_predictions, labels, gold_label_path, tokenizer, assigned_batch_size, device): 
    prediction_column = label_column + "_Model_Predictions"
    test_set[prediction_column] = total_predictions.tolist()
    test_set = test_set[test_set[label_column].notna()]
    for label in labels:
        if label != label_column:
            test_set = test_set[test_set[label] != 0]

        # print("Gathering ML predictions for Y_labeled_dataset in PPI!")

        Y_labeled_dataset = pd.read_csv(gold_label_path, sep="\t")
        Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[label_column].notna()]
        Y_labeled_dataset = Y_labeled_dataset.head(300)

        text_column = 'concat_text'
        if "Context" in label_column:
            Y_labeled_dataset[text_column] = [combine_query_document(Y_labeled_dataset.iloc[i]['Query'], Y_labeled_dataset.iloc[i]['Document']) for i in range(len(Y_labeled_dataset))]
        else:
            Y_labeled_dataset[text_column] = [combine_query_document(Y_labeled_dataset.iloc[i]['Query'], Y_labeled_dataset.iloc[i]['Document'], Y_labeled_dataset.iloc[i]['Answer']) for i in range(len(Y_labeled_dataset))]

        Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[text_column] != "Error"]
        
        if checkpoint:
            Y_labeled_dataloader = prepare_dataset_for_evaluation(Y_labeled_dataset, label_column, text_column, assigned_batch_size, tokenizer)
        else: 
            Y_labeled_dataloader = None
            
        Y_labeled_predictions = torch.FloatTensor([]).to(device)

        Yhat_unlabeled_dataset = test_set
        ####################################

    return test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column

def evaluate_and_scoring_data(test_set, Y_labeled_predictions, Y_labeled_dataset, Y_labeled_dataloader, Yhat_unlabeled_dataset, 
alpha, num_trials, model, device, model_choice, swap_human_labels_for_gpt4_labels, context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt, few_shot_examples, metric, prediction_column, 
label_column, test_set_selection, LLM_judge_ratio_predictions, validation_set_lengths, validation_set_ratios, ppi_confidence_intervals, accuracy_scores, results, checkpoint, llm_judge):
    # progress_bar = tqdm(range(len(Y_labeled_dataloader)))

    if checkpoint:
        model.eval()
        with tqdm(Y_labeled_dataloader, desc="Scoring", leave=False) as progress_bar:
            for batch in progress_bar:

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

                    Y_labeled_predictions = torch.cat((Y_labeled_predictions, predictions), 0)

                    progress_bar.update(1)

        Y_labeled_dataset[prediction_column] = Y_labeled_predictions.detach().cpu().numpy().tolist()
        
        Yhat_unlabeled_dataset = test_set

    ############################################################

    else:
        if llm_judge == "None":
            sys.exit("Error: No llm_judge provided")
        
        elif "gpt" in llm_judge:
            Y_labeled_predictions = []
            debug_mode = False # Hard coded - FIX

            # Few Shot Dataset Edge Check: Query ID 
            try:
                    _ = few_shot_examples.iloc[0]['Query']
                    query_id = "Query"
            except KeyError:
                try:
                    _ = few_shot_examples.iloc[0]['Question']
                    query_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing for the given row in few shot dataset.")

            # Labeled Dataset Edge Check: Query ID 
            try:
                    _ = Y_labeled_dataset.iloc[0]['Query']
                    query_labeled_id = "Query"
            except KeyError:
                try:
                    _ = Y_labeled_dataset.iloc[0]['Question']
                    query_labeled_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing in labeled dataset.")
            with tqdm(total=len(Y_labeled_dataset), desc="Evaluating", leave=False) as progress_bar:
                for _, row in Y_labeled_dataset.iterrows():
                    query = row[query_labeled_id]
                    document = row["Document"]
                    answer = row["Answer"]
                    if "Context_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(query), document, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
                    elif "Answer_Faithfulness_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
                    elif "Answer_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
            Y_labeled_predictions_np = np.array(Y_labeled_predictions)
            Y_labeled_dataset[prediction_column] = Y_labeled_predictions_np.tolist()
        elif "claude" in llm_judge: 
            Y_labeled_predictions = []
            debug_mode = False # Hard coded - FIX

            # Few Shot Dataset Edge Check: Query ID 
            try:
                    _ = few_shot_examples.iloc[0]['Query']
                    query_id = "Query"
            except KeyError:
                try:
                    _ = few_shot_examples.iloc[0]['Question']
                    query_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing for the given row in few shot dataset.")

            # Labeled Dataset Edge Check: Query ID 
            try:
                    _ = Y_labeled_dataset.iloc[0]['Query']
                    query_labeled_id = "Query"
            except KeyError:
                try:
                    _ = Y_labeled_dataset.iloc[0]['Question']
                    query_labeled_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing in labeled dataset.")
            with tqdm(total=len(Y_labeled_dataset), desc="Evaluating", leave=False) as progress_bar:
                for _, row in Y_labeled_dataset.iterrows():
                    query = row[query_labeled_id]
                    document = row["Document"]
                    answer = row["Answer"]
                    if "Context_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_context_relevance_scoring_claude(context_relevance_system_prompt, clean_query(query), document, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
                    elif "Answer_Faithfulness_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_faithfulness_scoring_claude(answer_faithfulness_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
                    elif "Answer_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_relevance_scoring_claude(answer_relevance_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
            Y_labeled_predictions_np = np.array(Y_labeled_predictions)
            Y_labeled_dataset[prediction_column] = Y_labeled_predictions_np.tolist()
        
        else: 
            Y_labeled_predictions = []
            debug_mode = False # Hard coded - FIX

            # Few Shot Dataset Edge Check: Query ID 
            try:
                    _ = few_shot_examples.iloc[0]['Query']
                    query_id = "Query"
            except KeyError:
                try:
                    _ = few_shot_examples.iloc[0]['Question']
                    query_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing for the given row in few shot dataset.")

            # Labeled Dataset Edge Check: Query ID 
            try:
                    _ = Y_labeled_dataset.iloc[0]['Query']
                    query_labeled_id = "Query"
            except KeyError:
                try:
                    _ = Y_labeled_dataset.iloc[0]['Question']
                    query_labeled_id = "Question"
                except KeyError:
                    sys.exit("Both 'Query' and 'Question' keys are missing in labeled dataset.")
            with tqdm(total=len(Y_labeled_dataset), desc="Evaluating", leave=False) as progress_bar:
                for _, row in Y_labeled_dataset.iterrows():
                    query = row[query_labeled_id]
                    document = row["Document"]
                    answer = row["Answer"]
                    if "Context_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_context_relevance_scoring_togetherai(context_relevance_system_prompt, clean_query(query), document, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
                    elif "Answer_Faithfulness_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_faithfulness_scoring_togetherai(answer_faithfulness_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
                    elif "Answer_Relevance_Label" == label_column:
                        Y_labeled_predictions.append(few_shot_answer_relevance_scoring_togetherai(answer_relevance_system_prompt, clean_query(query), document, answer, model, query_id, debug_mode, few_shot_examples))
                        progress_bar.update(1)
            Y_labeled_predictions_np = np.array(Y_labeled_predictions)
            Y_labeled_dataset[prediction_column] = Y_labeled_predictions_np.tolist()

    ############################################################

    # if swap_human_labels_for_gpt4_labels:
    #     if "Context_Relevance_Label" == label_column:
    #         Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(x["Query"]), x["Document"], gpt_model, few_shot_examples), axis=1)
    #     elif "Answer_Faithfulness_Label" == label_column:
    #         Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)
    #     elif "Answer_Relevance_Label" == label_column:
    #         Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)
    #     else:
    #         print("Error! Could not generate GPT labels for PPI.")
    #         assert False 
        
    Y_labeled = Y_labeled_dataset[label_column].values.astype(int)
    Yhat_labeled = Y_labeled_dataset[prediction_column].values.astype(int)
    Yhat_unlabeled = Yhat_unlabeled_dataset[prediction_column].values.astype(int)
    
    # print("Y_labeled, Yhat_labeled, Yhat_unlabeled for " + test_set_selection + " - " + label_column)
    # print(len(Y_labeled))
    # print(len(Yhat_labeled))
    # print(len(Yhat_unlabeled))
    # print("Y_labeled_dataset Label Distribution: ")
    # print(Y_labeled_dataset[label_column].tolist().count(1))
    # print(Y_labeled_dataset[label_column].tolist().count(0))
    # print("Y_labeled_dataset Prediction Distribution: ")
    # print(Y_labeled_dataset[prediction_column].tolist().count(1))
    # print(Y_labeled_dataset[prediction_column].tolist().count(0))
    # print("Yhat_unlabeled_dataset Prediction Distribution: ")
    # print(Yhat_unlabeled_dataset[prediction_column].tolist().count(1))
    # print(Yhat_unlabeled_dataset[prediction_column].tolist().count(0))

    ######################################################################

    avg_ci, avg_ci_classical, ci_imputed = calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials)
    LLM_judge_prediction = sum(avg_ci) / len(avg_ci)

    LLM_judge_ratio_predictions.append(LLM_judge_prediction)
    validation_set_lengths.append(len(test_set))
    validation_set_ratios.append(round(Yhat_unlabeled_dataset[label_column].tolist().count(1) / len(Yhat_unlabeled_dataset), 3))
    ppi_confidence_intervals.append([round(value, 3) for value in avg_ci])
    accuracy_scores.append(round(results['accuracy'], 3))

    ######################################################################

    indexed_list = list(enumerate(LLM_judge_ratio_predictions))
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    sorted_indices = [index for index, _ in sorted_list]

    print("--------------------------------------------------")
    print(label_column + " Scoring")
    print("ARES Ranking")
    print("ARES Prediction: " + str(LLM_judge_ratio_predictions))
    print("ARES Confidence Interval: " + str(ppi_confidence_intervals))
    print("Number of Examples in Evaluation Set: " + str(validation_set_lengths))
    print("Ground Truth Performance: " + str(validation_set_ratios))
    print("ARES LLM Judge Accuracy on Ground Truth Labels: " + str(accuracy_scores))
    print("Annotated Examples used for PPI: " + str(len(Y_labeled)))
    print("--------------------------------------------------\n")

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--alpha", type=float, required=True)
#     parser.add_argument("--num_trials", type=int, required=True)
#     parser.add_argument("--evaluation_datasets", nargs='+', required=True)
#     parser.add_argument("--checkpoints", nargs='+', required=True)
#     parser.add_argument("--labels", nargs='+', required=True)

#     parser.add_argument("--GPT_scoring", type=str, default="False", required=True)
#     parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo-16k", required=False)
#     parser.add_argument("--perform_zero_shot", type=str, default="False", required=False)
#     parser.add_argument("--few_shot_examples_filepath", type=str, required=True)

#     parser.add_argument("--Y_labeled_count", type=int, default=300, required=False)
#     parser.add_argument("--use_pseudo_human_labels", type=str, default="False", required=False)
#     parser.add_argument("--gold_label_path", type=str, required=False)
#     parser.add_argument("--swap_human_labels_for_gpt_labels", type=str, default="False", required=False)

#     args = parser.parse_args()

#     ### Instructions

#     # Settings for Human-labeled gold set for PPI
#     alpha = args.alpha
#     num_trials = args.num_trials
#     evaluation_datasets = args.evaluation_datasets
#     checkpoints = args.checkpoints
#     labels = args.labels
    
#     # Settings for zero/few-shot GPT scoring
#     GPT_scoring = args.GPT_scoring
#     if GPT_scoring == "True":
#         GPT_scoring = True
#     else:
#         GPT_scoring = False
    
#     gpt_model = args.gpt_model
#     perform_zero_shot = args.perform_zero_shot
#     if perform_zero_shot == "True":
#         perform_zero_shot = True
#     else:
#         perform_zero_shot = False
#     few_shot_examples_filepath = args.few_shot_examples_filepath

#     Y_labeled_count = args.Y_labeled_count
#     use_pseudo_human_labels = args.use_pseudo_human_labels
#     if use_pseudo_human_labels == "True":
#         use_pseudo_human_labels = True
#     else:
#         use_pseudo_human_labels = False
#     gold_label_path = args.gold_label_path
    
#     swap_human_labels_for_gpt4_labels = args.swap_human_labels_for_gpt_labels
#     if swap_human_labels_for_gpt4_labels == "True":
#         swap_human_labels_for_gpt4_labels = True
#     else:
#         swap_human_labels_for_gpt4_labels = False

#     assigned_batch_size = 1
#     number_of_labels = 2

    ############################################################



####################################################################

# for checkpoint, label_column in zip(checkpoints, labels):

#     LLM_judge_ratio_predictions = []
#     validation_set_lengths = []
#     validation_set_ratios = []
#     ppi_confidence_intervals = []
#     accuracy_scores = []
#     for test_set_selection in evaluation_datasets: