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
    "model_choice": <model_choice>, 
    "gold_label_path": <gold_label_filepath>, 
    "model_choice": "microsoft/deberta-v3-large",
    "llm_judge": "None",
    "assigned_batch_size": 1,
    "number_of_labels": 2,
    "alpha": 0.05,
    "num_trials": 1000,
    "vllm": False,
    "host_url": "http://0.0.0.0:8000/v1",
    "request_delay": 0,
    "debug_mode": False,
    "machine_label_llm_model": "None",
    "gold_machine_label_path": "None"
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

Specify the file path for the gold label dataset, which contains the true labels for the evaluation dataset. This file is used as a reference to measure the performance of the classifier by comparing its predictions against these true labels. 

```python
"gold_label_path": "nq_labeled_output.tsv"
```

<hr>

Link to [ARES Setup](setup.md) for gold label path file example used. 

## Model Choice

The name of the pre-trained model to be used for evaluation. In this example, microsoft/deberta-v3-large is chosen for its strong performance in text understanding and classification tasks. You can replace this with other model names based on your requirements and the specific tasks you are working on.

```python
"model_choice": "microsoft/deberta-v3-large"
```

## LLM Judge 

Specify the name of the LLM model to be used for evaluation. This is an optional parameter and can be set to "None" if no LLM model is needed for evaluation.

```python
"llm_judge": "None"
```

!!! note "Notice"
    Only specify LLM judge if you do not provide a checkpoint, if both are specified then the checkpoint will be used.


## Assigned Batch Size 

Determines the number of samples processed in each batch during evaluation. Smaller batch sizes can lead to more frequent updates but might be slower due to less parallel processing. Larger batch sizes can speed up the process but require more memory.

```python
"assigned_batch_size": 1
```

## Number of Labels 

Specifies the number of distinct labels used for classification tasks. This is crucial for setting up the model's output layer correctly and for interpreting the evaluation results.

```python
"number_of_labels": 2
```

## Alpha

Represents the significance level used in statistical hypothesis testing. It defines the threshold for rejecting the null hypothesis, with common values being 0.05, 0.01, etc.

```python
"alpha": 0.05
```

## Num Trials 

Specifies the number of trials or iterations used to estimate confidence intervals and other statistics utilized in PPI. Higher values can improve the accuracy of the estimates but require more computational resources.

```python
"num_trials": 1000
```

## vLLM

A flag to determine whether to use the vLLM API for evaluation. Setting this to True enables the use of vLLM.

```python
"vllm": False
```

## Host URL 

Specifies the host URL for the LLM API. This is an optional parameter and can be set to "http://0.0.0.0:8000/v1" if the LLM API is running locally.

```python
"host_url": "http://0.0.0.0:8000/v1"
```

!!! note "Notice"
    If you are using vLLM, ensure that the host URL is correct and the LLM API is running.

## Request Delay 

Specifies the delay in seconds between each request to the LLM API. This is an optional parameter and can be set to 0 if no delay is needed.

```python
"request_delay": 0
```

## Debug Mode 

A flag to determine whether to run the evaluation in debug mode. This is an optional parameter and can be set to False if debug mode is not needed.

```python
"debug_mode": False
```

## Machine Label LLM Model 

The machine_label_llm_model parameter specifies the LLM model to be used for generating machine labels. This can be useful for automated labeling processes in the absence of a gold label path.

```python
"machine_label_llm_model": "None"
```

## Gold Machine Label Path 

The file path to the machine-generated gold labels. By default is set to "None" if not using machine-generated gold labels.

```python
"gold_machine_label_path": "None"
```

!!! note "Notice"
    Specify gold_machine_label_path if you are using a machine_label_llm_model.

## RAG Evaluation Configuration: Full Example

```python
from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['nq_ratio_0.6.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["/future/u/manihani/ARES/checkpoints/microsoft-deberta-v3-large/Context_Relevance_Label_joint_datasets_2024-04-30_01:01:01.pt"], 
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv",
    "model_choice": "microsoft/deberta-v3-large",
    "llm_judge": "None",
    "assigned_batch_size": 1,
    "number_of_labels": 2,
    "alpha": 0.05,
    "num_trials": 1000,
    "vllm": False,
    "host_url": "http://0.0.0.0:8000/v1",
    "request_delay": 0,
    "debug_mode": False,
    "machine_label_llm_model": "None",
    "gold_machine_label_path": "None"
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)

```

Download the necessary files for this example [here](setup.md)!