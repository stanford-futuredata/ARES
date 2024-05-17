
This tutorial will showcase a <b>full walkthrough of how to use ARES on an NQ dataset with a ground truth accuracy of 60%</b>, showcasing ARES's robust evaluation accuracies. If you haven't, download the necessary datasets to the follow tutorials [here](/datasets.html).

<hr>


## [1] Synthetic Generation 

The first step is to configure the synthetic generation. Below contains the code for configuring the synthetic generation.

```python

from ares import ARES

synth_config = { 
    "document_filepaths": ["nq_labeled_output.tsv"] ,
    "few_shot_prompt_filename": "nq_few_shot_prompt_for_synthetic_query_generation.tsv",
    "synthetic_queries_filenames": ["nq_0.6_synthetic_queries.tsv"], 
    "documents_sampled": 6189
}

```

<hr>

## [2] Training Classifier

The second step is to train the classifier. Below contains the code for training the classifier.

```python
from ares import ARES

classifier_config = {
    "training_dataset": ["nq_0.6_synthetic_queries.tsv"], 
    "validation_set": ["nq_labeled_output.tsv"], 
    "label_column": ["Context_Relevance_Label"], 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,  # Change according to GPU memory
    "gradient_accumulation_multiplier": 32,  # Change according to GPU memory
}
```

<hr>

## [3] RAG Evaluation w/ ARES's PPI

The third step is to evaluate the unlabeled evaluation set using ARES's PPI in conjunction with the trained classifier we have from step 2. Below contains the code for evaluating the unlabeled evaluation set.

```python
from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['nq_unlabeled_output.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["Context_Relevance_Label_nq_labeled_output_date_time.pt"], 
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv", # Samples 300 labeled examples 
}
```

<hr>

## [4] Run all configurations together

THe final step is to run this entire pipeline. Below contains the code from previous steps and how to run this entire pipeline.

```python
from ares import ARES

synth_config = { 
    "document_filepaths": ["nq_labeled_output.tsv"] ,
    "few_shot_prompt_filename": "nq_few_shot_prompt_for_synthetic_query_generation.tsv",
    "synthetic_queries_filenames": ["nq_0.6_synthetic_queries.tsv"], 
    "documents_sampled": 6189
}

classifier_config = {
    "training_dataset": ["nq_0.6_synthetic_queries.tsv"], 
    "validation_set": ["nq_labeled_output.tsv"], 
    "label_column": ["Context_Relevance_Label"], 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6,
    "assigned_batch_size": 1,  # Change according to GPU memory
    "gradient_accumulation_multiplier": 32,  # Change according to GPU memory
}

ppi_config = { 
        "evaluation_datasets": ['nq_unlabeled_output.tsv'], 
        "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
        "checkpoints": ["Context_Relevance_Label_nq_labeled_output_date_time.pt"], 
        "labels": ["Context_Relevance_Label"], 
        "gold_label_path": "nq_labeled_output.tsv", # Samples 300 labeled examples 
}

ares = ARES(synthetic_query_generator=synth_config, classifier_model=classifier_config, ppi=ppi_config)
results = ares.run() 
print(results)
```








