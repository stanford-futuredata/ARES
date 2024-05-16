<h3>This page teaches you how to configure the RAG model evaluation with ARES to accurately evaluate your model's performance.</h3>

<hr>

## Configure API Key. 

```
export OPENAI_API_KEY=<your key here>
export TOGETHER_API_KEY=<your key here>
export ANTHROPIC_API_KEY=<your key here>
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
    "gold_label_path": <gold_label_filepath>
}

ares_module = ARES(ppi=ppi_config)
results = ares_module.evaluate_RAG()
print(results)
```

<hr>

## Evaluation Dataset(s)

Input file paths to datasets for PPI evaluation, which should contain labeled data for validating classifier performance.

```python
"evaluation_datasets": ["nq_unlabeled_output.tsv"],
```

Link to [ARES Setup](setup.md) for evaluation dataset example file used. 

<hr>

## Few-Shot Prompt File Path

Specify the file path for a file with few-shot examples, which PPI uses to understand the labeling schema and guide the evaluation.

```python
"few_shot_prompt_filename": "data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
```

Link to [ARES Setup](setup.md) for few-shot file example used. 

<hr>

## Checkpoint(s) File Path

Generated from ARES [Training Classifier](training_classifier.md), provide file path(s) to model checkpoint file(s), representing the saved states of the trained classifiers used for evaluation.

```python
"checkpoints": ["output/checkpoint_generated_from_training_classifier"],
```

<hr>


## Labels

List the names of label columns or individual label column in your dataset(s) that PPI will use for evaluation metrics.

```python
"labels": ["Context_Relevance_Label"], 
```

<hr>


## Gold Label Path

```python
"gold_label_path": "nq_labeled_output.tsv"
```

<hr>


Link to [ARES Setup](setup.md) for gold label path file example used. 


## RAG Evaluation Configuration: Full Example

```python
from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['nq_ratio_0.6.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Context_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt"], 
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv"
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)

```

Download the necessary files for this example [here](setup.md)!

