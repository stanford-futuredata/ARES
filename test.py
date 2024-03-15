from ares.ares import ARES
import pandas as pd

ues_idp_config = {
    # Dataset for in-domain prompts
    "in_domain_prompts_dataset": "/future/u/manihani/ARES/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    
    # Dataset for unlabeled evaluation
    "unlabeled_evaluation_set": "/future/u/manihani/ARES/datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv", 
    
    # Model: Mistral 7B
    "model_choice" : "mistralai/Mistral-7B-Instruct-v0.2"
}

# synth_config = { 
#     "document_filepaths": "datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv",
#     "few_shot_prompt_filename": "datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
#     "synthetic_queries_filename": "data/output/synthetic_queries_1.tsv",
#     "documents_sampled": 6381
# }

synth_config = { 
    "document_filepaths": ["/future/u/manihani/ARES/data/datasets_v2/hotpotqa/ratio_0.6_reformatted_full_articles_False_validation_with_negatives.tsv", "/future/u/manihani/ARES/data/datasets_v2/wow/ratio_0.6_reformatted_full_articles_False_validation_with_negatives.tsv"],
    "few_shot_prompt_filename": "/future/u/manihani/ARES/data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "synthetic_queries_filenames": ["data/output/hotpotqa_synthetc_queries_1.tsv", "data/output/WoW_synthetic_queries_1.tsv"],
    "documents_sampled": 10000
}

classifier_config = {
    "classification_dataset": "output/synthetic_queries_1.tsv", 
    "test_set_selection": "/future/u/manihani/ARES/datasets_v2/nq/ratio_0.6_reformatted_full_articles_False_validation_with_negatives.tsv", 
    "label_column": "Answer_Relevance_Label", 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

# ppi_config = { 
#     "evaluation_datasets": ['datasets_v2/nq/ratio_0.525_reformatted_full_articles_False_validation_with_negatives.tsv'], 
#     "few_shot_examples_filepath": "datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
#     "checkpoints": ["/future/u/manihani/ARES/example_checkpoints/Context-Relevance-Label.pt", "/future/u/manihani/ARES/example_checkpoints/Answer-Relevance-Validation-1867825.pt"],
#     "labels": ["Context_Relevance_Label", "Answer_Relevance_Label"], 
#     "GPT_scoring": False, 
#     "gold_label_path": "datasets_v2/nq/ratio_0.525_reformatted_full_articles_False_validation_with_negatives.tsv", 
#     "swap_human_labels_for_gpt4_labels": False
# }

ppi_config = { 
    "evaluation_datasets": ['/future/u/manihani/ARES/datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv'], 
    "few_shot_examples_filepath": "/future/u/manihani/ARES/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Context_Relevance_Label_ratio_0.6_reformatted_full_articles_False_validation_with_negatives_428380.pt"],
    "labels": ["Context_Relevance_Label"], 
    "GPT_scoring": False, 
    "gold_label_path": "/future/u/manihani/ARES/datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv", 
    "swap_human_labels_for_gpt4_labels": False
}

# ares = ARES(ues_idp=ues_idp_config)
# results = ares.ues_idp()
# print(results)

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)

# ares = ARES(classifier_model=classifier_config)
# results = ares.train_classifier()
# print(results)

# ares = ARES(ppi=ppi_config)
# results = ares.evaluate_RAG()
# print(results)