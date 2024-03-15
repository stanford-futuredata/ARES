from .synthetic_generator import synthetic_generator_config
from .binary_classifier import binary_classifer_config
from .rag_scoring import rag_scoring_config
from .ues_idp import ues_idp_config
from typing import List
import pandas as pd

class ARES: 
    config_spec = {
        "ues_idp": {
            "in_domain_prompts_dataset": (str, None), # Required parameter (No default)
            "unlabeled_evaluation_set" : (str, None), # Required parameter (No default)
            "model_choice": (str, "gpt-3.5-turbo-1106"),
        },

        "synthetic_query_generator": {
            "document_filepaths": (list, None),  # Required parameter (No default)
            "few_shot_prompt_filename": (str, None),  # Required parameter (No default)
            "synthetic_queries_filenames": (list, None),  # Required parameter (No default)
            "documents_sampled": (int, None),  # Required parameter (No default)
            "model_choice": (str, "google/flan-t5-xxl"),
            "flan_approach": (bool, True), 
            "clean_documents": (bool, False), 
            "regenerate_synth_questions": (bool, True), 
            "percentiles": (list,[0.05, 0.25, 0.5, 0.95]), 
            "question_temperatures": (list, [2.0, 1.5, 1.0, 0.5, 0.0]), 
            "regenerate_answers": (bool, True), 
            "generate_contradictory_answers_with_flan": (bool, True), 
            "number_of_negatives_added_ratio": (float, 0.5), 
            "lower_bound_for_negatives": (int, 5), 
            "number_of_contradictory_answers_added_ratio": (float, 0.67), 
            "number_of_positives_added_ratio": (float, 0.0),
            "regenerate_embeddings": (float, True), 
            "synthetic_query_prompt": (str, "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n")
        },
        
        "classifier_model": {
            "classification_dataset": (list, None), # Required parameter (No default)
            "test_set_selection": (str, None), # Required parameter (No default)
            "label_column": (str, None), # Required parameter (No default)
            "num_epochs": (int, None), # Required parameter (No default)
            "patience_value": (int, None), # Required parameter (No default)
            "learning_rate": (float, None), # Required parameter (No default)
            "model_choice": (str, "microsoft/deberta-v3-large"), 
            "validation_set_scoring": (bool, True), 
            "assigned_batch_size": (int, 1), 
            "gradient_accumulation_multiplier": (int, 32), 
            "number_of_runs": (int, 1), 
            "num_warmup_steps": (int, 100)
        },

        "ppi": {
            "evaluation_datasets": (list, None), # Required parameter (No default)
            "few_shot_examples_filepath": (str, None), # Required parameter (No default)
            "checkpoints": (list, None), # Required parameter (No default)
            "labels": (list, None), # Required parameter (No default)
            "GPT_scoring":(bool, None), # Required parameter (No default)
            "gold_label_path": (str, None), # Required parameter (No default)
            "swap_human_labels_for_gpt4_labels": (bool, None), # Required parameter (No default)
            "use_pseudo_human_labels": (bool, False),
            "model_choice": (str, "microsoft/deberta-v3-large"), 
            "assigned_batch_size": (int, 1),
            "number_of_labels": (int, 2),
            "alpha": (int, 0.05),   
            "num_trials": (int, 1000)
        }
    }

    def __init__(self, synthetic_query_generator={}, ues_idp={}, classifier_model={}, ppi={}):
        self.synthetic_query_generator_config = self.prepare_config("synthetic_query_generator", synthetic_query_generator)
        self.classifier_model_config = self.prepare_config("classifier_model", classifier_model)
        self.ppi_config = self.prepare_config("ppi", ppi)
        self.ues_idp_config = self.prepare_config("ues_idp", ues_idp)

    def generate_synthetic_data(self): 
        if self.synthetic_query_generator_config == {}: 
            print("Skipping synthetic generator configuration due to missing parameters.")
        else: 
            synthetic_generator_config(**self.synthetic_query_generator_config) 

    def train_classifier(self): 
        if self.classifier_model_config == {}:
            print("Skipping binary classifier configuration due to missing parameters.")
        else:
            binary_classifer_config(**self.classifier_model_config) 

    def evaluate_RAG(self):
        if self.ppi_config == {}: 
            print("Skipping RAG evaluation configuration due to no parameters")
        else:
            rag_scoring_config(**self.ppi_config)
        
    def ues_idp(self):
        if 'in_domain_prompts_dataset' in self.ues_idp_config:
            in_domain_prompts_dataset = pd.read_csv(self.ues_idp_config['in_domain_prompts_dataset'], sep='\t')
        else:
            in_domain_prompts_dataset = None 

        if 'unlabeled_evaluation_set' in self.ues_idp_config:
            unlabeled_evaluation_set = pd.read_csv(self.ues_idp_config['unlabeled_evaluation_set'], sep='\t')
        else:
            unlabeled_evaluation_set = None
        if self.ues_idp_config == {}: 
            print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
        else: 
            model_choice = self.ues_idp_config["model_choice"]
            return ues_idp_config(in_domain_prompts_dataset, unlabeled_evaluation_set, model_choice)

    def run(self): 
        if self.synthetic_query_generator_config == {}: 
            print("Skipping synthetic generator configuration due to no parameters.")
        else: 
            synthetic_generator_config(**self.synthetic_query_generator_config) 

        if self.classifier_model_config == {}:
            print("Skipping binary classifier configuration due to no parameters.")
        else:
            binary_classifer_config(**self.classifier_model_config) 
        
        if self.ppi_config == {}: 
            print("Skipping RAG evaluation configuration due to no parameters")
        else:
            rag_scoring_config(**self.ppi_config)
        
        if 'in_domain_prompts_dataset' in self.ues_idp_config:
            in_domain_prompts_dataset = pd.read_csv(self.ues_idp_config['in_domain_prompts_dataset'], sep='\t')
        else:
            in_domain_prompts_dataset = None 

        if 'unlabeled_evaluation_set' in self.ues_idp_config:
            unlabeled_evaluation_set = pd.read_csv(self.ues_idp_config['unlabeled_evaluation_set'], sep='\t')
        else:
            unlabeled_evaluation_set = None
        if self.ues_idp_config == {}: 
            print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
            exit()
        else: 
            model_choice = self.ues_idp_config["model_choice"]
            ues_idp_config(in_domain_prompts_dataset, unlabeled_evaluation_set, model_choice)

    def prepare_config(self, component_name, user_config):
        if not user_config:
            return {}
        component = self.config_spec[component_name]
        prepared_config = {}
        for param, (expected_type, default) in component.items():
            if param in user_config:
                value = user_config[param]
                if not isinstance(value, expected_type):
                    raise TypeError(f"Parameter '{param}' for {component_name} is expected to be of type {expected_type.__name__}, received {type(value).__name__} instead.")
                prepared_config[param] = value
            elif default is not None:
                prepared_config[param] = default
            else:
                raise ValueError(f"Missing required parameter '{param}' for {component_name}.")
        return prepared_config