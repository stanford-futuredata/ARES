from .synthetic_generator import synthetic_generator_config
from .binary_classifier import binary_classifer_config
from .rag_scoring import rag_scoring_config
from .ues_idp import ues_idp_config
from .kilt_filter import KILT_dataset_process
from .superglue_filter import superGlue
from .prompts import context_relevance_system_prompt, answer_relevance_system_prompt, answer_faithfulness_system_prompt
from typing import List
import pandas as pd

class ARES:
    """
    ARES class encapsulates configurations for various components of the ARES system.
    It allows for the setup and management of configurations for synthetic query generation,
    binary classification, RAG scoring, and UES/IDP processes.
    """
    
    # A placeholder object used to denote default values in the configuration specification
    REQUIRED_BUT_HAS_DEFAULT = object()

    # Configuration specification dictionary defining the parameters for each component
    config_spec = {
        "ues_idp": {
            "in_domain_prompts_dataset": (str, None),  # Required parameter with no default value
            "unlabeled_evaluation_set": (str, None),  # Required parameter with no default value
            "context_relevance_system_prompt": (str,context_relevance_system_prompt),  # Optional with default
            "answer_relevance_system_prompt": (str, answer_relevance_system_prompt),  # Optional with default
            "answer_faithfulness_system_prompt": (str, answer_faithfulness_system_prompt),  # Optional with default
            "debug_mode": (bool, False),  # Optional with default
            "documents": (int, 0),  # Optional with default
            "model_choice": (str, "gpt-3.5-turbo-1106"),  # Optional with default
            "request_delay": (int, 0),  # Optional with default
            "vllm": (bool, False),  # Optional with default
            "host_url": (str, "None")  # Optional with default
        },

        "synthetic_query_generator": {
            "document_filepaths": (list, None),  # Required parameter with no default value
            "few_shot_prompt_filename": (str, None),  # Required parameter with no default value
            "synthetic_queries_filenames": (list, None),  # Required parameter with no default value
            "documents_sampled": (int, None),  # Required parameter with no default value
            "model_choice": (str, "google/flan-t5-xxl"),  # Optional with default
            "clean_documents": (bool, False),  # Optional with default
            "regenerate_synth_questions": (bool, True),  # Optional with default
            "percentiles": (list, [0.05, 0.25, 0.5, 0.95]),  # Optional with default
            "question_temperatures": (list, [2.0, 1.5, 1.0, 0.5, 0.0]),  # Optional with default
            "regenerate_answers": (bool, True),  # Optional with default
            "number_of_negatives_added_ratio": (float, 0.5),  # Optional with default
            "lower_bound_for_negatives": (int, 5),  # Optional with default
            "number_of_contradictory_answers_added_ratio": (float, 0.67),  # Optional with default
            "number_of_positives_added_ratio": (float, 0.0),  # Optional with default
            "regenerate_embeddings": (float, True),  # Optional with default
            "synthetic_query_prompt": (str, "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n")  # Optional with default
        },
        
        "classifier_model": {
            "training_dataset": (list, None),  # Required parameter with no default value
            "validation_set": (list, None),  # Required parameter with no default value
            "label_column": (list, None),  # Required parameter with no default value
            "num_epochs": (int, None),  # Required parameter with no default value
            "patience_value": (int, None),  # Required parameter with no default value
            "learning_rate": (float, None),  # Required parameter with no default value
            "training_dataset_path": (str, "None"),  # Optional with default
            "validation_dataset_path": (str, "None"),  # Optional with default
            "model_choice": (str, "microsoft/deberta-v3-large"),  # Optional with default
            "validation_set_scoring": (bool, True),  # Optional with default
            "assigned_batch_size": (int, REQUIRED_BUT_HAS_DEFAULT, 1),  # Default is 1
            "gradient_accumulation_multiplier": (int, REQUIRED_BUT_HAS_DEFAULT, 32),  # Default is 32
            "number_of_runs": (int, 1),  # Optional with default
            "num_warmup_steps": (int, 100),  # Optional with default
            "training_row_limit": (int, -1),  # Optional with default
            "validation_row_limit": (int, -1)  # Optional with default
        },

        "ppi": {
            "evaluation_datasets": (list, None),  # Required parameter with no default value
            "few_shot_examples_filepath": (str, None),  # Required parameter with no default value
            "labels": (list, None),  # Required parameter with no default value
            "checkpoints": (list, []),  # Required parameter with no default value
            "gold_label_path": (str, "None"),  # Optional with default
            "rag_type": (str, "question_answering"),  # Optional with default
            "model_choice": (str, "microsoft/deberta-v3-large"),  # Optional with default
            "llm_judge": (str, "None"),  # Optional with default
            "assigned_batch_size": (int, 1),  # Optional with default
            "number_of_labels": (int, 2),  # Optional with default
            "alpha": (int, 0.05),  # Optional with default
            "num_trials": (int, 1000),  # Optional with default
            "vllm": (bool, False),  # Optional with default
            "host_url": (str, "http://0.0.0.0:8000/v1"),  # Optional with default
            "request_delay": (int, 0),  # Optional with default
            "debug_mode": (bool, False),  # Optional with default
            "machine_label_llm_model": (str, "None"),  # Optional with default
            "gold_machine_label_path": (str, "None")  # Optional with default
        }
    }

    def __init__(self, synthetic_query_generator={}, ues_idp={}, classifier_model={}, ppi={}):
        """
        Initializes the ARES class with configurations for different components.

        Args:
            synthetic_query_generator (dict): Configuration for the synthetic query generator.
            ues_idp (dict): Configuration for UES IDP.
            classifier_model (dict): Configuration for the classifier model.
            ppi (dict): Configuration for PPI (Protein-Protein Interaction).

        Each configuration dictionary is passed to the `prepare_config` method to validate
        and prepare the final configuration based on the default settings defined in `config_spec`.
        """
        self.synthetic_query_generator_config = self.prepare_config("synthetic_query_generator", synthetic_query_generator)
        self.classifier_model_config = self.prepare_config("classifier_model", classifier_model)
        self.ppi_config = self.prepare_config("ppi", ppi)
        self.ues_idp_config = self.prepare_config("ues_idp", ues_idp)

    def generate_synthetic_data(self):
        """
        Generates synthetic data using the synthetic query generator configuration.
        If the configuration is empty, the generation process is skipped.
        """
        if not self.synthetic_query_generator_config:
            print("Skipping synthetic generator configuration due to missing parameters.")
        else:
            synthetic_generator_config(**self.synthetic_query_generator_config)

    def train_classifier(self):
        """
        Trains the classifier using the classifier model configuration.
        If the configuration is empty, the training process is skipped.
        """
        if not self.classifier_model_config:
            print("Skipping binary classifier configuration due to missing parameters.")
        else:
            binary_classifer_config(**self.classifier_model_config)

    def evaluate_RAG(self):
        """
        Evaluates the RAG using the PPI configuration.
        If the configuration is empty, the evaluation process is skipped.
        """
        if not self.ppi_config:
            print("Skipping RAG evaluation configuration due to no parameters")
        else:
            rag_scoring_config(**self.ppi_config)

    def ues_idp(self):
        """
        Executes the UES IDP configuration.
        """
        return ues_idp_config(**self.ues_idp_config)

    def KILT_dataset(self, dataset_name):
        """
        Processes the KILT dataset with the specified dataset name.
        """
        KILT_dataset_process(dataset_name)

    def superGlue_dataset(self, dataset_name):
        """
        Processes the SuperGlue dataset with the specified dataset name.
        """
        superGlue(dataset_name)

    def run(self):
        """
        Executes the configurations for synthetic data generation, classifier training, and RAG evaluation.
        If any configuration is empty, the corresponding process is skipped.
        """
        self.generate_synthetic_data()
        self.train_classifier()
        self.evaluate_RAG()

        if not self.ues_idp_config:
            print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
            exit()
        else:
            model_choice = self.ues_idp_config["model_choice"]
            ues_idp_config(**self.ues_idp_config)

    def prepare_config(self, component_name, user_config):
        """
        Prepares and validates the configuration for a given component based on the default settings.

        Args:
            component_name (str): The name of the component for which the configuration is being prepared.
            user_config (dict): The user-provided configuration dictionary.

        Returns:
            dict: The prepared and validated configuration dictionary.

        Raises:
            TypeError: If the provided value type does not match the expected type.
            ValueError: If a required parameter is missing and no default is specified.
        """
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