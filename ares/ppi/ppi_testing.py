
import os, time
import sys
#sys.path.insert(1, '../')
from ppi import clt_iid, binomial_iid, pp_mean_iid_asymptotic
import numpy as np
from tqdm import tqdm
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split

######################################################################

def calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials):
    #alpha = 0.05
    #num_trials = 500
    n_max = Y_labeled.shape[0] # Total number of labeled ballots
    ns = np.linspace(100,n_max,20).astype(int)
    #ns = np.linspace(0,n_max,20).astype(int)
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
            ci_classical[j,i,:] = binomial_iid(n,alpha,y.mean())
        #print("F: " + str(f))
        #print("y: " + str(y))
        #print("Yhat_unlabeled: " + str(Yhat_unlabeled))
        #print("alpha: " + str(alpha))
        #print("output: " + str(output))  
    ci_imputed = binomial_iid(Yhat_unlabeled.shape[0], alpha, imputed_estimate)
    avg_ci = ci.mean(axis=0)[-1]
    avg_ci_classical = ci_classical.mean(axis=0)[-1]
    return avg_ci, avg_ci_classical, ci_imputed

######################################################################

column = "Context_Relevance" #"Answer_Faithfulness" #"Context_Relevance"
label_column = column + "_Label"
prediction_column = column + "_Prediction"
alpha = 0.05 #0.05
num_trials = 1000

#Y_labeled_filepath = "../datasets_v2/LLM_Judge_Test_Set_V4_Reannotated.tsv"
#Y_labeled_filepath = "../datasets_v2/LLM_Judge_Test_Set_Human_Annotations_V1.1_Filtered"
#Y_labeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/qa_logs_with_logging_details_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_with_Model_Predictions_train.tsv"
#Y_labeled_filepath = "../datasets_v2/wow_reformatted_validation_with_negatives_True_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/wow_reformatted_validation_with_negatives_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/fever_reformatted_validation_with_negatives_True_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/fever_reformatted_validation_with_negatives_False_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_True_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_False_True_False_False_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_True_True_False_False_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_False_True_False_False_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/LLM_Judge_Test_Set_Human_Annotations_V1.1_Filtered_False_True_True_True_sampled.tsv"
#Y_labeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negativesAnswer_Faithfulness_Label_with_Model_Predictions_Sampled.tsv"
Y_labeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negativesContext_Relevance_Label_with_Model_Predictions_Sampled.tsv"

Y_labeled_dataset = pd.read_csv(Y_labeled_filepath, sep="\t")
#Y_labeled_dataset[label_column] = Y_labeled_dataset["Context_Relevance"]
if 'Context_Relevance_Label_Model_Predictions' in Y_labeled_dataset.columns:
    Y_labeled_dataset["Context_Relevance_Prediction"] = Y_labeled_dataset['Context_Relevance_Label_Model_Predictions']
if 'Answer_Faithfulness_Label_Model_Predictions' in Y_labeled_dataset.columns:
    Y_labeled_dataset["Answer_Faithfulness_Prediction"] = Y_labeled_dataset['Answer_Faithfulness_Label_Model_Predictions']

Y_labeled_dataset = Y_labeled_dataset[Y_labeled_dataset[label_column].notna()]
Y_labeled = Y_labeled_dataset[label_column].values.astype(int)
Yhat_labeled = Y_labeled_dataset[prediction_column].values.astype(int)


#Yhat_unlabeled_filepath = "../../datasets/synthetic_queries_v8.1.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../../datasets/synthetic_queries_v10_with_Model_Predictions.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/qa_logs_with_logging_details_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_with_Model_Predictions_test.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/wow_reformatted_validation_with_negatives_True_True_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/wow_reformatted_validation_with_negatives_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/fever_reformatted_validation_with_negatives_True_True_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/fever_reformatted_validation_with_negatives_False_False_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_True_True_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negatives_False_True_False_False_False_True_False_False_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_True_True_False_False_True_True_False_False_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/nq_reformatted_validation_with_negatives_v2_False_True_False_False_False_True_False_False_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/qa_logs_with_logging_details_False_True_True_True_False_True_True_True_non_sampled.tsv"
#Yhat_unlabeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negativesAnswer_Faithfulness_Label_with_Model_Predictions_NonSampled.tsv"
Yhat_unlabeled_filepath = "../datasets_v2/hotpotqa_reformatted_validation_with_negativesContext_Relevance_Label_with_Model_Predictions_NonSampled.tsv"

Yhat_unlabeled_dataset = pd.read_csv(Yhat_unlabeled_filepath, sep="\t")
if 'Context_Relevance_Label_Model_Predictions' in Yhat_unlabeled_dataset.columns:
    Yhat_unlabeled_dataset["Context_Relevance_Prediction"] = Yhat_unlabeled_dataset['Context_Relevance_Label_Model_Predictions']
if 'Answer_Faithfulness_Label_Model_Predictions' in Yhat_unlabeled_dataset.columns:
    Yhat_unlabeled_dataset["Answer_Faithfulness_Prediction"] = Yhat_unlabeled_dataset['Answer_Faithfulness_Label_Model_Predictions']

Yhat_unlabeled_dataset = Yhat_unlabeled_dataset[Yhat_unlabeled_dataset[label_column].notna()]
Yhat_unlabeled_dataset = Yhat_unlabeled_dataset[Yhat_unlabeled_dataset[prediction_column].notna()]

if 'synthetic_query' in Yhat_unlabeled_dataset.columns:
    Yhat_unlabeled_dataset = Yhat_unlabeled_dataset.dropna(subset=['synthetic_query',"document", label_column])
else:
    Yhat_unlabeled_dataset = Yhat_unlabeled_dataset.dropna(subset=['Query',"Document", label_column])

######################################################################

print("Yhat_unlabeled_dataset length")
print(len(Yhat_unlabeled_dataset))

#conversion_dict = {"Yes": 1, "No": 0}
#Yhat_unlabeled_dataset[prediction_column] = Yhat_unlabeled_dataset[prediction_column].apply(lambda x: conversion_dict[x])
Yhat_unlabeled = Yhat_unlabeled_dataset[prediction_column].values.astype(int)

print(type(Y_labeled))
print(Y_labeled.shape)
print(set(Y_labeled))
print(type(Yhat_labeled))
print(Yhat_labeled.shape)
print(set(Yhat_labeled))
print(type(Yhat_unlabeled))
print(Yhat_unlabeled.shape)
print(set(Yhat_unlabeled))

print("Positive/Negative Ratio for " + label_column)
print(Yhat_unlabeled_dataset[label_column].tolist().count(1))
print(Yhat_unlabeled_dataset[label_column].tolist().count(0))
print(Yhat_unlabeled_dataset[label_column].tolist().count(1) / len(Yhat_unlabeled_dataset))
print(Yhat_unlabeled_dataset[label_column].tolist().count(0) / len(Yhat_unlabeled_dataset))

######################################################################

avg_ci, avg_ci_classical, ci_imputed = calculate_ppi(Y_labeled,  Yhat_labeled, Yhat_unlabeled, alpha, num_trials)
print("PPI Confidence Interval")
print(avg_ci)
print("Classical Confidence Interval")
print(avg_ci_classical)
print("Imputed Confidence Interval")
print(ci_imputed)