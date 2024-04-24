from .synthetic_generator import synthetic_generator_config
from .binary_classifier import binary_classifer_config
from .rag_scoring import rag_scoring_config
from .ues_idp import ues_idp_config
from .kilt_filter import KILT_dataset_process
from .superglue_filter import superGlue
from typing import List
import pandas as pd

context_relevance_system_prompt = "You are an expert dialogue agent."
context_relevance_system_prompt += "Your task is to analyze the provided document and determine whether it is relevant for responding to the dialogue. "
context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
context_relevance_system_prompt = context_relevance_system_prompt

answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
answer_relevance_system_prompt = answer_relevance_system_prompt

answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
answer_faithfulness_system_prompt = answer_faithfulness_system_prompt

class ARES: 
    REQUIRED_BUT_HAS_DEFAULT = object()

    config_spec = {
        "ues_idp": {
            "in_domain_prompts_dataset": (str, None), # Required parameter (No default),
            "unlabeled_evaluation_set" : (str, None), # Required parameter (No default),
            "context_relevance_system_prompt": (str, context_relevance_system_prompt),
            "answer_relevance_system_prompt": (str, answer_relevance_system_prompt),
            "answer_faithfulness_system_prompt": (str, answer_faithfulness_system_prompt),
            "debug_mode": (bool, False),
            "documents": (int, 0),
            "model_choice": (str, "gpt-3.5-turbo-1106")
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
            "training_dataset": (list, None), # Required parameter (No default)
            "validation_set": (list, None), # Required parameter (No default)
            "label_column": (list, None), # Required parameter (No default)
            "num_epochs": (int, None), # Required parameter (No default)
            "patience_value": (int, None), # Required parameter (No default)
            "learning_rate": (float, None), # Required parameter (No default)
            "training_dataset_path": (str, "None"),
            "validation_dataset_path": (str, "None"),
            "model_choice": (str, "microsoft/deberta-v3-large"), 
            "validation_set_scoring": (bool, True), 
            "assigned_batch_size": (int, REQUIRED_BUT_HAS_DEFAULT, 1),  # Default is 1
            "gradient_accumulation_multiplier": (int, REQUIRED_BUT_HAS_DEFAULT, 32),  # Default is 32
            "number_of_runs": (int, 1), 
            "num_warmup_steps": (int, 100),
            "training_row_limit": (int, -1),
            "validation_row_limit": (int, -1)
        },

        "ppi": {
            "evaluation_datasets": (list, None), # Required parameter (No default)
            "few_shot_examples_filepath": (str, None), # Required parameter (No default)
            "labels": (list, None), # Required parameter (No default)
            "gold_label_path": (str, None), # Required parameter (No default)
            "checkpoints": (list, []), # Required parameter (No default)
            "rag_type": (str, "question_answering"), 
            "annotated_datapoints_filepath": (bool, False), 
            "GPT_scoring":(bool, False), 
            "model_choice":(str, "microsoft/deberta-v3-large"),
            "llm_judge": (str, "None"), 
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
        return ues_idp_config(**self.ues_idp_config)
        # if 'in_domain_prompts_dataset' in self.ues_idp_config:
        #     in_domain_prompts_dataset = pd.read_csv(self.ues_idp_config['in_domain_prompts_dataset'], sep='\t')
        # else:
        #     in_domain_prompts_dataset = None 

        # if 'unlabeled_evaluation_set' in self.ues_idp_config:
        #     unlabeled_evaluation_set = pd.read_csv(self.ues_idp_config['unlabeled_evaluation_set'], sep='\t')
        # else:
        #     unlabeled_evaluation_set = None
        # if self.ues_idp_config == {}: 
        #     print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
        # else: 
        #     model_choice = self.ues_idp_config["model_choice"]
    
    def KILT_dataset(self, dataset_name): 
        KILT_dataset_process(dataset_name)

    def superGlue_dataset(self, dataset_name): 
        superGlue(dataset_name)

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
        
        # if 'in_domain_prompts_dataset' in self.ues_idp_config:
        #     in_domain_prompts_dataset = pd.read_csv(self.ues_idp_config['in_domain_prompts_dataset'], sep='\t')
        # else:
        #     in_domain_prompts_dataset = None 

        # if 'unlabeled_evaluation_set' in self.ues_idp_config:
        #     unlabeled_evaluation_set = pd.read_csv(self.ues_idp_config['unlabeled_evaluation_set'], sep='\t')
        # else:
        #     unlabeled_evaluation_set = None
        if self.ues_idp_config == {}: 
            print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
            exit()
        else: 
            model_choice = self.ues_idp_config["model_choice"]
            ues_idp_config(**self.ues_idp_config)

    def prepare_config(self, component_name, user_config):
        if not user_config:
            return {}
        component = self.config_spec[component_name]
        prepared_config = {}
        for param, config in component.items():
            expected_type, default, default_value = config if len(config) == 3 else (*config, None)
            if param in user_config:
                value = user_config[param]
                if not isinstance(value, expected_type):
                    raise TypeError(f"Parameter '{param}' for {component_name} is expected to be of type {expected_type.__name__}, received {type(value).__name__} instead.")
                prepared_config[param] = value
            elif default is self.REQUIRED_BUT_HAS_DEFAULT:
                # If the parameter is required but not provided, use the specified default value.
                if param not in user_config:
                    print(f"\nWarning: '{param}' not provided for {component_name}, using default value {default_value}.")
                prepared_config[param] = default_value
            elif default is not None:
                prepared_config[param] = default
            else:
                raise ValueError(f"Missing required parameter '{param}' for {component_name}.")
        return prepared_config