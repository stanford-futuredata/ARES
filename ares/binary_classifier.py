# binary_classifier.py

from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import (
    load_model,
    prepare_and_clean_data,
    analyze_and_report_data,
    transform_data,
    split_dataset,
    prepare_dataset,
    initalize_dataset_for_tokenization,
    train_and_evaluate_model,
    evaluate_model,
    print_and_save_model,
    set_random_seed
)
from tqdm import tqdm
import torch
import pandas as pd
import sys
import warnings

def binary_classifer_config(
    training_dataset: list,
    validation_set: list,
    label_column: list,
    num_epochs: int,
    patience_value: int,
    learning_rate: float,
    training_dataset_path: str = "None",
    validation_dataset_path: str = "None",
    model_choice: str = "microsoft/deberta-v3-large",
    validation_set_scoring: bool = True,
    assigned_batch_size: int = 1,
    gradient_accumulation_multiplier: int = 32,
    number_of_runs: int = 1,
    num_warmup_steps: int = 100,
    training_row_limit: int = -1,
    validation_row_limit: int = -1
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seed(42)

    for label in label_column:
        set_random_seed(42)

        tokenizer, max_token_length = load_model(model_choice)

        prepare_data_settings = {
            "training_dataset": training_dataset,
            "validation_set": validation_set,
            "label": label,
            "training_dataset_path": training_dataset_path,
            "validation_dataset_path": validation_dataset_path,
            "training_row_limit": training_row_limit,
            "validation_row_limit": validation_row_limit,
            "model_choice": model_choice,
            "num_epochs": num_epochs,
            "patience_value": patience_value,
            "chosen_learning_rate": learning_rate,
            "gradient_accumulation_multiplier": gradient_accumulation_multiplier,
            "assigned_batch_size": assigned_batch_size,
            "tokenizer": tokenizer
        }

        checkpoint_path, patience_value = prepare_and_clean_data(prepare_data_settings)

        synth_queries = analyze_and_report_data(training_dataset_path, label, tokenizer, max_token_length)

        train_df, test_set = transform_data(synth_queries, validation_dataset_path, label)

        train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, text_set_label_, labels_list = split_dataset(train_df, training_dataset_path, test_set, label)

        training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas = prepare_dataset(validation_set_scoring, train_set_label, train_set_text, dev_set_label, dev_set_text)

        tokenized_datasets = initalize_dataset_for_tokenization(tokenizer, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow)

        train_and_eval_settings = {
            "number_of_runs": number_of_runs,
            "tokenized_datasets": tokenized_datasets,
            "assigned_batch_size": assigned_batch_size,
            "train_set_label": train_set_label,
            "model_choice": model_choice,
            "chosen_learning_rate": learning_rate,
            "device": device,
            "checkpoint_path": checkpoint_path,
            "patience_value": patience_value,
            "num_epochs": num_epochs,
            "num_warmup_steps": num_warmup_steps,
            "gradient_accumulation_multiplier": gradient_accumulation_multiplier
        }

        model, avg_train_losses, avg_valid_losses, eval_dataloader, inference_times = train_and_evaluate_model(train_and_eval_settings)

        total_predictions, total_references, metric = evaluate_model(model, model_choice, checkpoint_path, device, eval_dataloader, inference_times)

        print_and_save_model(total_predictions, total_references, checkpoint_path, metric)
