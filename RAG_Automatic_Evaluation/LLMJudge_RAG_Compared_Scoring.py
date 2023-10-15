
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

from ppi import clt_iid, binomial_iid, pp_mean_iid_asymptotic
from Evaluation_Functions import calculate_accuracy, few_shot_context_relevance_scoring
from Evaluation_Functions import few_shot_answer_faithfulness_scoring, few_shot_answer_relevance_scoring

#############################################################

random_state = 44

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

class CustomBERTModel(nn.Module):
    def __init__(self, number_of_labels, model_choice):

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

          self.classifier = nn.Sequential(nn.Linear(embedding_size, 256), nn.Linear(256, number_of_labels))
          self.embedding_size = embedding_size


    def forward(self, ids, mask, labels=None, decoder_input_ids=None):
          
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

def calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials):

    n_max = Y_labeled.shape[0]
    #ns = np.linspace(100,n_max,20).astype(int)
    ns = np.linspace(0,n_max,20).astype(int)

    # Imputed-only estimate
    imputed_estimate = (Yhat_labeled.sum() + Yhat_unlabeled.sum())/(Yhat_labeled.shape[0] + Yhat_unlabeled.shape[0])

    # Run prediction-powered inference and classical inference for many values of n
    ci = np.zeros((num_trials, ns.shape[0], 2))
    ci_classical = np.zeros((num_trials, ns.shape[0], 2))
    for i in tqdm(range(ns.shape[0])):
        for j in range(num_trials):
            # Prediction-Powered Inference
            n = ns[i]
            rand_idx = np.random.permutation(n)
            f = Yhat_labeled.astype(float)[rand_idx[:n]]
            y = Y_labeled.astype(float)[rand_idx[:n]]    
            output = pp_mean_iid_asymptotic(y,f,Yhat_unlabeled,alpha)
            ci[j,i,:] = output
            # Classical interval
            try:
                ci_classical[j,i,:] = binomial_iid(n,alpha,y.mean())
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







######################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    parser.add_argument("--evaluation_datasets", nargs='+', required=True)
    parser.add_argument("--checkpoints", nargs='+', required=True)
    parser.add_argument("--labels", nargs='+', required=True)

    parser.add_argument("--GPT_scoring", type=str, default="False", required=True)
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo-16k", required=False)
    parser.add_argument("--perform_zero_shot", type=str, default="False", required=False)
    parser.add_argument("--few_shot_examples_filepath", type=str, required=False)

    parser.add_argument("--Y_labeled_count", type=int, default=300, required=False)
    parser.add_argument("--use_pseudo_human_labels", type=str, default="False", required=False)
    parser.add_argument("--gold_label_path", type=str, required=False)
    parser.add_argument("--swap_human_labels_for_gpt_labels", type=str, default="False", required=False)

    args = parser.parse_args()

    ### Instructions

    # Settings for Human-labeled gold set for PPI
    alpha = args.alpha
    num_trials = args.num_trials
    evaluation_datasets = args.evaluation_datasets
    checkpoints = args.checkpoints
    labels = args.labels
    correct_ranking = [i for i in range(0, len(evaluation_datasets))]
    
    # Settings for zero/few-shot GPT scoring
    GPT_scoring = args.GPT_scoring
    if GPT_scoring == "True":
        GPT_scoring = True
    else:
        GPT_scoring = False
    
    gpt_model = args.gpt_model
    perform_zero_shot = args.perform_zero_shot
    if perform_zero_shot == "True":
        perform_zero_shot = True
    else:
        perform_zero_shot = False
    few_shot_examples_filepath = args.few_shot_examples_filepath

    Y_labeled_count = args.Y_labeled_count
    use_pseudo_human_labels = args.use_pseudo_human_labels
    if use_pseudo_human_labels == "True":
        use_pseudo_human_labels = True
    else:
        use_pseudo_human_labels = False
    gold_label_path = args.gold_label_path
    
    swap_human_labels_for_gpt4_labels = args.swap_human_labels_for_gpt_labels
    if swap_human_labels_for_gpt4_labels == "True":
        swap_human_labels_for_gpt4_labels = True
    else:
        swap_human_labels_for_gpt4_labels = False

    assigned_batch_size = 1
    number_of_labels = 2

    ############################################################









    ######################################################################

    print("--------------------------------------------------------")
    print("Evaluation Sets: " + str(evaluation_datasets))
    print("Checkpoints: " + str(checkpoints))
    print("Labels: "  + str(labels))
    print("GPT Scoring: " + str(GPT_scoring))
    print("--------------------------------------------------------")

    ######################################################################

    if GPT_scoring:
        checkpoint = ["" for _ in range(len(labels))]

    if few_shot_examples_filepath is not None:
        few_shot_examples = pd.read_csv(few_shot_examples_filepath, sep="\t")
        print("few_shot_examples")
        print(len(few_shot_examples))
        print(few_shot_examples.head())

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

    if "wow" in evaluation_datasets[0].lower():
        context_relevance_system_prompt = "You are an expert dialogue agent. "
        context_relevance_system_prompt += "Given the following dialogue and document, you must analyze the provided document and determine whether it is relevant for responding to the dialogue. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    if "fever" in evaluation_datasets[0].lower():
        context_relevance_system_prompt = "You are an expert fact-checking agent. "
        context_relevance_system_prompt += "Given the following statement and document, you must analyze the provided document and determine whether it is sufficient for determining the statement's factuality. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided statement's factuality. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document is not sufficient. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    else:
        context_relevance_system_prompt = "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    
    if "wow" in evaluation_datasets[0].lower():
        answer_faithfulness_system_prompt = "Given the following dialogue, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
    if "fever" in evaluation_datasets[0].lower():
        answer_faithfulness_system_prompt = "Given the following statement, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
    else:
        answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
    answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
    answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
    answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
    answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

    if "wow" in evaluation_datasets[0].lower():
        answer_relevance_system_prompt = "Given the following dialogue, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided dialogue. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the dialogue and provides only correct information from the document for responding to the dialogue. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given dialogue and "[[No]]" if the answer is not relevant for the given dialogue. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    if "fever" in evaluation_datasets[0].lower():
        answer_relevance_system_prompt = "Given the following statement, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided statement. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the statement and provides only correct information from the document for answering the statement. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given statement and "[[No]]" if the answer is not relevant for the given statement. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    else:
        answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"

    ####################################################################

    for checkpoint, label_column in zip(checkpoints, labels):

        LLM_judge_ratio_predictions = []
        validation_set_lengths = []
        validation_set_ratios = []
        ppi_confidence_intervals = []
        accuracy_scores = []
        for test_set_selection in evaluation_datasets:

            test_set = pd.read_csv(test_set_selection, sep="\t")
            text_column = 'concat_text'
            test_set = test_set[test_set[label_column].notna()]
            if "Context" in label_column:
                test_set[text_column] = [combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document']) for i in range(len(test_set))]
            else:
                test_set[text_column] = [combine_query_document(test_set.iloc[i]['Query'], test_set.iloc[i]['Document'], test_set.iloc[i]['Answer']) for i in range(len(test_set))]

            test_set = test_set[test_set[text_column] != "Error"]
            print("Example Text for " + label_column + " Scoring")
            print(test_set.iloc[10][text_column])

            ############################################################

            model_choice = "microsoft/deberta-v3-large"
            max_token_length = 2048
            tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=max_token_length)

            device = "cuda:0"
            device = torch.device(device)
            if not GPT_scoring:
                model = CustomBERTModel(number_of_labels, model_choice)
                model.to(device)

            ############################################################

            if GPT_scoring:
                test_set = test_set.sample(n=2000, random_state=43)
            else:
                print("Loading the Best Finetuned-LLM Checkpoint")
                model.load_state_dict(torch.load(checkpoint))

            test_set = test_set.sample(n=100, random_state=43)

            ############################################################

            eval_dataloader = prepare_dataset_for_evaluation(test_set, label_column, text_column)

            ############################################################

            metric = load_metric("accuracy")

            total_predictions = torch.FloatTensor([]).to(device)
            total_references = torch.FloatTensor([]).to(device)
            total_logits = torch.FloatTensor([]).to(device)

            if not GPT_scoring:

                progress_bar = tqdm(range(len(eval_dataloader)))
                model.eval()
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
                        total_logits = torch.cat((total_logits, logits), 0)

                        progress_bar.update(1)

            else:

                print("Performing GPT scoring!")
                print("Using gpt model: " + gpt_model)
                if perform_zero_shot:
                    print("Using zero-shot approach")
                    print("Setting few-shot example to None...")
                    few_shot_examples = None

                if "Context_Relevance_Label" == label_column:
                    tqdm.pandas(desc="Generating context relevance scores...", total=test_set.shape[0])
                    test_set["Context_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(x["Query"]), x["Document"], gpt_model, few_shot_examples), axis=1)
                elif "Answer_Faithfulness_Label" == label_column:
                    tqdm.pandas(desc="Generating answer faithfulness scores...", total=test_set.shape[0])
                    test_set["Answer_Faithfulness_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)
                if "Answer_Relevance_Label" == label_column:
                    tqdm.pandas(desc="Generating answer relevance scores...", total=test_set.shape[0])
                    test_set["Answer_Relevance_Prediction"] = test_set.progress_apply(lambda x: few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)

                total_predictions = test_set[label_column.replace("_Label", "_Prediction")]
                total_references = test_set[label_column]

                
            ############################################################

            results = metric.compute(references=total_references, predictions=total_predictions)

            ########################################################################

            prediction_column = label_column + "_Model_Predictions"
            test_set[prediction_column] = total_predictions.tolist()
            test_set = test_set[test_set[label_column].notna()]
            for label in labels:
                if label != label_column:
                    test_set = test_set[test_set[label] != 0]

            if use_pseudo_human_labels:
                y_labeled_ratio = Y_labeled_count / len(test_set)
                Yhat_unlabeled_dataset, Y_labeled_dataset = train_test_split(test_set, test_size=y_labeled_ratio, random_state=42)
                Yhat_unlabeled_dataset = test_set
            else:
                Y_labeled_dataset = pd.read_csv(gold_label_path, sep="\t")
                Yhat_unlabeled_dataset = test_set

            if swap_human_labels_for_gpt4_labels:
                if "Context_Relevance_Label" == label_column:
                    tqdm.pandas(desc="Generating context relevance labels using GPT...", total=Y_labeled_dataset.shape[0])
                    Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_context_relevance_scoring(context_relevance_system_prompt, clean_query(x["Query"]), x["Document"], gpt_model, few_shot_examples), axis=1)
                elif "Answer_Faithfulness_Label" == label_column:
                    tqdm.pandas(desc="Generating answer faithfulness labels using GPT...", total=Y_labeled_dataset.shape[0])
                    Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)
                elif "Answer_Relevance_Label" == label_column:
                    tqdm.pandas(desc="Generating answer relevance labels using GPT...", total=Y_labeled_dataset.shape[0])
                    Y_labeled_dataset[label_column] = Y_labeled_dataset.progress_apply(lambda x: few_shot_answer_relevance_scoring(answer_relevance_system_prompt, clean_query(x["Query"]), x["Document"], x["Answer"], gpt_model, few_shot_examples), axis=1)
                else:
                    print("Error! Could not generate GPT labels for PPI.")
                    assert False 
                
            Y_labeled = Y_labeled_dataset[label_column].values.astype(int)
            Yhat_labeled = Y_labeled_dataset[prediction_column].values.astype(int)
            Yhat_unlabeled = Yhat_unlabeled_dataset[prediction_column].values.astype(int)
            
            print("Y_labeled, Yhat_labeled, Yhat_unlabeled for " + test_set_selection + " - " + label_column)
            print(len(Y_labeled))
            print(len(Yhat_labeled))
            print(len(Yhat_unlabeled))
            print(Y_labeled_dataset[label_column].tolist().count(1))
            print(Y_labeled_dataset[label_column].tolist().count(0))
            print(Y_labeled_dataset[prediction_column].tolist().count(1))
            print(Y_labeled_dataset[prediction_column].tolist().count(0))
            print(Yhat_unlabeled_dataset[prediction_column].tolist().count(1))
            print(Yhat_unlabeled_dataset[prediction_column].tolist().count(0))

            ######################################################################

            avg_ci, avg_ci_classical, ci_imputed = calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials)
            LLM_judge_prediction = sum(avg_ci) / len(avg_ci)

            LLM_judge_ratio_predictions.append(LLM_judge_prediction)
            validation_set_lengths.append(len(test_set))
            validation_set_ratios.append(round(Yhat_unlabeled_dataset[label_column].tolist().count(1) / len(Yhat_unlabeled_dataset), 3))
            ppi_confidence_intervals.append(avg_ci.tolist())
            accuracy_scores.append(results['accuracy'])

        ######################################################################

        indexed_list = list(enumerate(LLM_judge_ratio_predictions))
        sorted_list = sorted(indexed_list, key=lambda x: x[1])
        sorted_indices = [index for index, _ in sorted_list]
        tau, p_value = stats.kendalltau(correct_ranking, sorted_indices)

        print("--------------------------------------------------")
        print(label_column + " Scoring")
        print("Correct Ranking v. ARES Ranking")
        print(correct_ranking)
        print(sorted_indices)
        print("Kendall's Tau: " + str(tau))
        print("P-Value: " + str(p_value))
        print("Avg. PPIs: " + str(LLM_judge_ratio_predictions))
        print("PPI Confidence Intervals: " + str(ppi_confidence_intervals))
        print("Evaluation Set Lengths: " + str(validation_set_lengths))
        print("Evaluation Set Ratio: " + str(validation_set_ratios))
        print("Test Accuracy Scores: " + str(accuracy_scores))
        print("Y-Labeled Example Count: " + str(len(Y_labeled)))
        print("--------------------------------------------------\n")

