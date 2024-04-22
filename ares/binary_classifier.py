from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import load_model
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import prepare_and_clean_data
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import analyze_and_report_data
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import transform_data
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import split_dataset
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import prepare_dataset
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import initalize_dataset_for_tokenization
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import train_and_evaluate_model
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import evaluate_model 
from .LLM_as_a_Judge_Adaptation.General_Binary_Classifier import print_and_save_model
from tqdm import tqdm 
import torch
import pandas as pd
import sys
import warnings


def binary_classifer_config(training_dataset: list, validation_set: list, label_column: list, 
                    num_epochs: int, patience_value: int, learning_rate: float, training_dataset_path: str = "None",
                    validation_dataset_path: str = "None", model_choice: str = "microsoft/deberta-v3-large", validation_set_scoring: bool = True, 
                    assigned_batch_size: int = 1, gradient_accumulation_multiplier: int = 32, 
                    number_of_runs: int = 1, num_warmup_steps: int = 100, training_row_limit: int = -1, validation_row_limit: int = -1):

    device = "cuda:0"
    device = torch.device(device)
    learning_rate_choices = [learning_rate]
    # training_datasets = [training_dataset]

    if training_row_limit == -1: # Edge case to assign row_limit to None if not passed 
        training_row_limit = None

    if validation_row_limit == -1: 
        validation_row_limit = None

    if len(training_dataset) == 1:
        # If there's only one dataset, use it directly, regardless of training_dataset_path being None
        training_dataset_path = training_dataset[0]
    elif training_dataset_path == "None":
        # Exit if there's more than one dataset and no path provided for the combined dataset
        sys.exit("Error: A path for saving the combined dataset is required when multiple training datasets are specified. Please specify the 'training_dataset_path' parameter in your configuration. This path will be used to store the dataset resulting from merging all provided training datasets.")
    else:
        combined_df = pd.DataFrame()
        print("\n")
        for dataset in tqdm(training_dataset, desc="Combining datasets"):
            temp_df = pd.read_csv(dataset, sep='\t', low_memory=False, nrows=training_row_limit)
            print(f"Loaded dataset {dataset} with {temp_df.shape[0]} rows" + 
              (f" (limited to first {training_row_limit} rows)" if training_row_limit is not None else ""))
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        combined_df.to_csv(training_dataset_path, sep='\t', index=False)
        # combined_df = pd.concat((pd.read_csv(dataset, sep='\t', low_memory=False) for dataset in tqdm(training_dataset, desc="Combining datasets")), ignore_index=True)
        # combined_df.to_csv(training_dataset_path, sep='\t', index=False)
    
    if len(validation_set) == 1: 
        validation_dataset_path = validation_set[0]
    elif len(validation_set) > 1: 
        if validation_dataset_path == "None": 
            sys.exit("Error: A path for saving the combined validation dataset is required when multiple validation sets are specified. Please specify the 'validation_dataset_path'.")
        elif len(validation_set) != len(training_dataset): 
           warnings.warn(f"Error: Validations sets provided are not equivalent to number of training datasets. {len(validation_set)} validation set(s) are provided, and {len(training_dataset)} are provided")
        else: 
            # for dataset in tqdm(validation_set, desc="Counting rows in validation datasets"):
            #     df = pd.read_csv(dataset, sep='\t', low_memory=False)
            #     num_rows = df.shape[0]
            #     print(f"Number of rows in {dataset}: {num_rows}")
            
            # Combine all validation datasets into one and save it
            combined_val_df = pd.DataFrame()
            print("--------------------------------------------------------------------------")
            for dataset in tqdm(validation_set, desc="Combining validation datasets"):
                temp_val_df = pd.read_csv(dataset, sep='\t', low_memory=False, nrows=validation_row_limit)
                print(f"Loaded dataset {dataset} with {temp_val_df.shape[0]} rows" + 
                      (f" (limited to first {validation_row_limit} rows)" if validation_row_limit is not None else ""))
                combined_val_df = pd.concat([combined_val_df, temp_val_df], ignore_index=True)
            combined_val_df.to_csv(validation_dataset_path, sep='\t', index=False)
            print("--------------------------------------------------------------------------")

            # Filter out rows where 'Answer' column has NaN values
            initial_count = combined_val_df.shape[0]
            combined_val_df = combined_val_df.dropna(subset=['Answer'])
            final_count = combined_val_df.shape[0]

            # Calculate and print the number of removed data points, if any were removed
            removed_count = initial_count - final_count
            if removed_count > 0:
                print(f"Removed {removed_count} data points from the validation set due to NaN in the 'Answer' column.")

            # Save the cleaned validation dataset
            combined_val_df.to_csv(validation_dataset_path, sep='\t', index=False)

    for label in label_column:
        for chosen_learning_rate in learning_rate_choices:
            # for dataset in training_datasets:

            tokenizer, max_token_length  = load_model(model_choice) # Possibly pass in max_token_length as parameter into config

            checkpoint_path, patience_value = prepare_and_clean_data(training_dataset_path, learning_rate_choices, chosen_learning_rate, model_choice, 
                                                                    number_of_runs, validation_set_scoring, 
                                                                    label, validation_dataset_path, patience_value, 
                                                                    num_epochs, num_warmup_steps, gradient_accumulation_multiplier, 
                                                                    assigned_batch_size, tokenizer)

            synth_queries = analyze_and_report_data(training_dataset_path, label, tokenizer, max_token_length)

            train_df, test_set = transform_data(synth_queries, validation_dataset_path, label)

            train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, text_set_label_, labels_list = split_dataset(train_df, training_dataset_path, test_set, label)

            training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas = prepare_dataset(validation_set_scoring, train_set_label, train_set_text, dev_set_label, dev_set_text)

            tokenized_datasets = initalize_dataset_for_tokenization(tokenizer, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow)

            model, avg_train_losses, avg_valid_losses, eval_dataloader, inference_times = train_and_evaluate_model(number_of_runs, tokenized_datasets, assigned_batch_size, train_set_label, model_choice, chosen_learning_rate, 
                                    device, checkpoint_path, patience_value, num_epochs, num_warmup_steps, gradient_accumulation_multiplier)

            total_predictions, total_references, metric = evaluate_model(model, model_choice, checkpoint_path, device, eval_dataloader, inference_times)

            print_and_save_model(total_predictions, total_references, checkpoint_path, metric)

