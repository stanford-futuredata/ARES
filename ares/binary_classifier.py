from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import load_model
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import prepare_and_clean_data
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import analyze_and_report_data
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import transform_data
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import split_dataset
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import prepare_dataset
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import initalize_dataset_for_tokenization
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import train_and_evaluate_model
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import evaluate_model 
from ares.LLM_as_a_Judge_Adaptation.General_Binary_Classifier import print_and_save_model
import pandas as pd
import torch

def binary_classifer_config(classification_datasets: list, test_set_selection: str, label_column: str, 
                    num_epochs: int, patience_value: int, learning_rate: float, 
                    model_choice: str = "microsoft/deberta-v3-large", validation_set_scoring: bool = True, 
                    assigned_batch_size: int = 1, gradient_accumulation_multiplier: int = 32, 
                    number_of_runs: int = 1, num_warmup_steps: int = 100):

    device = "cuda:0"
    device = torch.device(device)
    learning_rate_choices = [learning_rate]
    combined_dataset = None

    for dataset_path in classification_datasets:
        dataset = pd.read_csv(dataset_path)  
        if combined_dataset is None:
            combined_dataset = dataset
        else:
            combined_dataset = pd.concat([combined_dataset, dataset], ignore_index=True)

    combined_dataset_path = "combined_classification_dataset.csv"
    combined_dataset.to_csv(combined_dataset_path, index=False)

    for chosen_learning_rate in learning_rate_choices:
        tokenizer, max_token_length  = load_model(model_choice) # Possibly pass in max_token_length as parameter into config
        
        checkpoint_path, patience_value = prepare_and_clean_data(combined_dataset_path, learning_rate_choices, chosen_learning_rate, 
                                                                combined_dataset_path, model_choice, 
                                                                number_of_runs, validation_set_scoring, 
                                                                label_column, test_set_selection, patience_value, 
                                                                num_epochs, num_warmup_steps, gradient_accumulation_multiplier, 
                                                                assigned_batch_size, tokenizer)
        
        synth_queries = analyze_and_report_data(combined_dataset_path, label_column, tokenizer, max_token_length)
        
        train_df, test_set = transform_data(synth_queries, test_set_selection, label_column)
        
        train_set_text, train_set_label, dev_set_text, dev_set_label, test_set_text, text_set_label_, labels_list = split_dataset(train_df, combined_dataset_path, test_set, label_column)
        
        training_dataset_pandas, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow, test_dataset_pandas = prepare_dataset(validation_set_scoring, train_set_label, train_set_text, dev_set_label, dev_set_text)
        
        tokenized_datasets = initalize_dataset_for_tokenization(tokenizer, training_dataset_arrow, validation_dataset_arrow, test_dataset_arrow)
        
        model, avg_train_losses, avg_valid_losses, eval_dataloader, inference_times = train_and_evaluate_model(number_of_runs, tokenized_datasets, assigned_batch_size, train_set_label, model_choice, chosen_learning_rate, 
                                device, checkpoint_path, patience_value, num_epochs, num_warmup_steps, gradient_accumulation_multiplier)

        total_predictions, total_references, metric = evaluate_model(model, model_choice, checkpoint_path, device, eval_dataloader, inference_times)

        print_and_save_model(total_predictions, total_references, checkpoint_path, metric)






