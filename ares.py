import subprocess 
from RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring 
from RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring
from RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring
from RAG_Automatic_Evaluation.Evaluation_Functions import calculate_accuracy
from synthetic_generator import synthetic_generator_config
from binary_classifier import binary_classifer_config
from rag_scoring import rag_scoring_config
from typing import List
import pandas as pd

class ARES: 
    config_spec = {
        "ues_idp": {
            "in_domain_prompts_dataset": (str, None), # Required parameter (No default)
            "unlabeled_evaluation_set" : (str, None), # Required parameter (No default)
        },

        "synthetic_query_generator": {
            "document_filepath": (str, None),  # Required parameter (No default)
            "few_shot_prompt_filename": (str, None),  # Required parameter (No default)
            "synthetic_queries_filename": (str, None),  # Required parameter (No default)
            "documents_sampled": (int, None),  # Required parameter (No default)
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
            "classification_dataset": (str, None), # Required parameter (No default)
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
     
     ####################################
        self.ues_idp_config = self.prepare_config("ues_idp", ues_idp)

        if 'in_domain_prompts_dataset' in self.ues_idp_config:
            self.in_domain_prompts_dataset = pd.read_csv(self.ues_idp_config['in_domain_prompts_dataset'], sep='\t')
        else:
            self.in_domain_prompts_dataset = None 
        
        if 'unlabeled_evaluation_set' in self.ues_idp_config:
            self.unlabeled_evaluation_set = pd.read_csv(self.ues_idp_config['unlabeled_evaluation_set'], sep='\t')
        else:
            self.unlabeled_evaluation_set = None

        self.gpt_model = "gpt-3.5-turbo-1106"
        #####
        context_relevance_system_prompt = "You are an expert dialogue agent."
        context_relevance_system_prompt += "Your task is to analyze the provided document and determine whether it is relevant for responding to the dialogue. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.context_relevance_system_prompt = context_relevance_system_prompt
        #####
        answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.answer_relevance_system_prompt = answer_relevance_system_prompt
        #####
        answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
        answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
        answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
        answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.answer_faithfulness_system_prompt = answer_faithfulness_system_prompt
        ####

##############################################################################################################################
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
        
        if self.unlabeled_evaluation_set.empty or self.in_domain_prompts_dataset.empty: 
            print(f"Skipping UES and IDP since either unlabeled evaluation set or in domain prompts not provided")
        else: 
            context_relevance_answers = self.unlabeled_evaluation_set["Context_Relevance_Label"].tolist()
            answer_relevance_answers = self.unlabeled_evaluation_set["Answer_Relevance_Label"].tolist()
            answer_faithfulness_answers = self.unlabeled_evaluation_set["Answer_Faithfulness_Label"].tolist()
            context_relevance_scores = []
            answer_relevance_scores = []
            answer_faithfulness_scores = []

            for index, row in self.unlabeled_evaluation_set.iterrows():
                # Extract query, document, and answer from the row
                query = row["Query"]
                document = row["Document"]
                answer = row["Answer"]

                # Scoring
                context_score = few_shot_context_relevance_scoring(
                    self.context_relevance_system_prompt, query, document, self.gpt_model, self.in_domain_prompts_dataset)

                answer_relevance_score = few_shot_answer_relevance_scoring(
                    self.answer_relevance_system_prompt, query, document, answer, self.gpt_model, self.in_domain_prompts_dataset)
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring(self.answer_faithfulness_system_prompt, query, document, answer, self.gpt_model, self.in_domain_prompts_dataset)

                    # Append scores to respective lists
                context_relevance_scores.append(context_score)
                answer_relevance_scores.append(answer_relevance_score)
                answer_faithfulness_scores.append(answer_faithfulness_score)

            # Compile results into a dictionary
            return {
                "Context Relevance Scores": round(sum(context_relevance_scores)/len(context_relevance_scores), 3),
                "Answer Faithfulness Scores": round(sum(answer_faithfulness_scores)/len(answer_faithfulness_answers), 3),
                "Answer Relevance Scores": round(sum(answer_relevance_scores)/len(answer_relevance_answers), 3)
            }

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