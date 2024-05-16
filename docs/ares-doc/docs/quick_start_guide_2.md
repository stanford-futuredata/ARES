<hr> 

## Overview

!!! note "Required Datasets"
    Ensure you have installed all necessary datasets from the setup page [here](setup.md)


This quick start guide demonstrates how to use a large language model (LLM), specifically GPT-3.5, to evaluate an unlabeled dataset with in-domain prompting. Additionally, it showcases ARES's robust process, which includes synthetic data generation, training a classifier, and using Prediction Powered Inference (PPI) to significantly enhance evaluation accuracy.

<hr> 

#### Step 1) Run the following to see GPT 3.5's accuracy on the NQ unlabeled dataset!

```python
from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "unlabeled_evaluation_set": "nq_unlabeled_output.tsv", 
    "model_choice" : "gpt-3.5-turbo-0125"
} 

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)
# {'Context Relevance Scores': [Score], 'Answer Faithfulness Scores': [Score], 'Answer Relevance Scores': [Score]}
```

<hr> 

#### Step 2) Run the following to see ARES's synthetic generation in action! 
```python

from ares import ARES

synth_config = { 
    "document_filepaths": ["nq_labeled_output.tsv"] ,
    "few_shot_prompt_filename": "nq_few_shot_prompt_for_synthetic_query_generation.tsv",
    "synthetic_queries_filenames": ["synthetic_queries_1.tsv"], 
    "documents_sampled": 6189
}

ares_module = ARES(synthetic_query_generator=synth_config)
results = ares_module.generate_synthetic_data()
print(results)
```

<hr>

#### Step 3) Run the following to see ARES's training classifier in action!
```python

from ares import ARES

classifier_config = {
    "training_dataset": ["synthetic_queries_1.tsv"], 
    "validation_set": ["nq_labeled_output.tsv"], 
    "label_column": ["Context_Relevance_Label"], 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,  
    "gradient_accumulation_multiplier": 32,  
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)
```

Note: This code creates a checkpoint for the trained classifier.
Training may take some time. You can download our jointly trained checkpoint on context relevance here!
[Download Checkpoint](https://drive.google.com/file/d/15poFyeoqdnaNZVjl41HllL2213DKyZjH/view?usp=sharing)

Alternatively, you can download our jointly trained checkpoint on answer relevance as well! Be sure to change the parameters in the config to match the label "Answer_Relevance_Label" [Download Checkpoint](https://drive.google.com/file/d/1wGcgELBfnCGqXlPEbpPmf7LJ53DPWVXI/view?usp=sharing) 

<hr>

#### Step 4) Run the following to see ARES's PPI in action!
```python

from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['nq_unlabeled_output.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["Context_Relevance_Label_nq_labeled_output_date_time.pt"], 
    "rag_type": "question_answering", 
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv", 
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)
```

<br>