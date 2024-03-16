<h3>This page teaches you how to configure the RAG model evaluation with ARES to accurately evaluate your model's performance.</h3>

<hr>

## Configure OpenAI API Key. 

```
export OPENAI_API_KEY=<your key here>
```

<hr>

## RAG Evaluation Configuration

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. Below is how the synthetic generation configuration style.

```python 
from ares import ARES

ppi_config = { 
    "evaluation_datasets": [<eval_dataset_filepath>],
    "few_shot_examples_filepath": <few_shot_filepath>,
    "checkpoints": [<checkpoint_filepath>],
    "labels": [<labels>], 
    "model_choice": <model_choice>, # Default model is "microsoft/deberta-v3-large"
    "GPT_scoring": <True or False>, 
    "gold_label_path": <gold_label_filepath>, 
    "swap_human_labels_for_gpt4_labels": False
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
```

<hr>

## Evaluation Dataset(s)

Input file paths to datasets for PPI evaluation, which should contain labeled data for validating classifier performance.

```python
    "evaluation_datasets": ["/data/datasets_v2/nq/nq_ratio_0.5_.tsv"],
```

Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets_v2/nq) for evaluation dataset example file used. 

<hr>

## Few-Shot Prompt File Path

Specify the file path for a file with few-shot examples, which PPI uses to understand the labeling schema and guide the evaluation.

```python
    "few_shot_prompt_filename": "data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
```

Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets) for few-shot file example used. 

<hr>

## Checkpoint(s) File Path

Generated from ARES [Training Classifier](training_classifier.md), provide file path(s) to model checkpoint file(s), representing the saved states of the trained classifiers used for evaluation.

```python
"checkpoints": ["output/checkpoint_generated_from_training_classifier"],
```

<hr>


## Labels

List the names of label columns or individua label column in your dataset(s) that PPI will use for evaluation metrics.

```python
    "labels": ["Context_Relevance_Label"], 
```

<hr>

## GPT_scoring

Set this flag to True if you want to use a GPT model for scoring; False uses the trained classifiers' scores.

```python
    "GPT_scoring": False,
```

<hr>


## Gold Label Path

```python
    "gold_label_path": "/data/datasets_v2/nq/nq_ratio_0.6_.tsv"
```

<hr>


Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets_v2/nq) for gold label path file example used. 

## Swapping Human Labels for GPT4 Labels

If True, PPI replaces human-provided labels with GPT-4's labels during evaluation; if False, it uses the original human labels.

```python
    "swap_human_labels_for_gpt4_labels": False
```
<hr>


## RAG Evaluation Configuration: Full Example

```python
from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['../data/datasets_v2/nq/nq_ratio_0.6_.tsv'], 
    "few_shot_examples_filepath": "../data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "checkpoints": ["../data/checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Context_Relevance_Label_ratio_0.6_reformatted_full_articles_False_validation_with_negatives_428380.pt", "../data/checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Answer_Relevance_Label_ratio_0.6_reformatted_full_articles_False_validation_with_negatives_428380.pt"],
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label"], 
    "GPT_scoring": False, 
    "gold_label_path": "../data/datasets_v2/nq/nq_ratio_0.6_.tsv", 
    "swap_human_labels_for_gpt4_labels": False
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)

```