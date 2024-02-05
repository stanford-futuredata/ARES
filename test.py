from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": "/future/u/manihani/ARES/datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv",
    "unlabeled_evaluation_set": "/future/u/manihani/ARES/datasets_v2/nq/nq_reformatted_full_articles_False_validation.tsv"
}

synth_config = { 
    "document_filepath": "/future/u/manihani/ARES/datasets_v2/nq/nq_reformatted_full_articles_False_validation.tsv",
    "few_shot_prompt_filename": "/future/u/manihani/ARES/datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv",
    "synthetic_queries_filename": "output/synthetic_queries_1.tsv",
    "documents_sampled": 1000
}

classifier_config = {
    "classification_dataset": "output/synthetic_queries_1.tsv", 
    "test_set_selection": "example_files/evaluation_datasets.tsv", 
    "label_column": "Context_Relevance_Label", 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

ppi_config = { 
    "evaluation_datasets": ['/future/u/manihani/ARES/example_files/evaluation_datasets.tsv'], 
    "few_shot_examples_filepath": "example_files/few_shot_prompt_filename.tsv",
    "checkpoints": ["checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Context_Relevance_Label_evaluation_datasets_670487.pt"],
    "labels": ["Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label"], 
    "GPT_scoring": False, 
    "gold_label_path": "example_files/gold_label_path.tsv", 
    "swap_human_labels_for_gpt4_labels": False
}


ares = ARES(ues_idp=ues_idp_config)
results = ares.run()
print(results)


