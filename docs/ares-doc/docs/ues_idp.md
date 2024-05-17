
<hr>

This guide will walk you through the parameters for running a simple evaluation on
an unlabeled evaluation set in conjunction with in-domain prompts (UES/IDP).

<hr> 

## UES/IDP Configuration

Below is an example of a UES/IDP configuration. Blank parameters are placeholders.

```python 
from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": <few_shot_filepath, 
    "unlabeled_evaluation_set": <eval_dataset_filepath>,
    "context_relevance_system_prompt": context_relevance_system_prompt,
    "answer_relevance_system_prompt": answer_relevance_system_prompt,
    "answer_faithfulness_system_prompt": answer_faithfulness_system_prompt,
    "debug_mode": False,
    "documents": 0,
    "model_choice": "gpt-3.5-turbo-1106",
    "request_delay": 0,
    "vllm": False,
    "host_url": "None"
}

```

<hr> 

### In-Domain Prompts 

In-domain prompts are a set of few-shot examples that are relevant to the evaluation set.

```python
in_domain_prompts_dataset = <few_shot_filepath>
```

[Here](https://github.com/stanford-futuredata/ARES/blob/main/datasets/example_files/nq_few_shot_prompt_for_synthetic_query_generation.tsv) is an example of an in-domain prompt dataset for the Natural Questions (NQ) dataset.

<hr> 

### Unlabeled Evaluation Set 

The unlabeled evaluation set is a set of unlabeled examples containing questions, documents, and answers which will be evaluated 
for either context relevance, answer relevance, or answer faithfulness.

```python
unlabeled_evaluation_set = <eval_dataset_filepath>
```

<hr> 

### Context Relevance System Prompt 

The context relevance system prompt is a prompt that will be used to evaluate the context relevance of the answers to the questions.

```python
context_relevance_system_prompt = (
    "You are an expert dialogue agent. "
    "Your task is to analyze the provided document and determine whether 
    "it is relevant for responding to the dialogue. "
    "In your evaluation, you should consider the content of the document "
    "and how it relates to the provided dialogue. "
    "Output your final verdict by strictly following this format: "[[Yes]]" 
    "if the document is relevant and "[[No]]" 
    "if the document provided is not relevant."
    "Do not provide any additional explanation for your decision.\n\n"
)
```

### Answer Relevance System Prompt 

The answer relevance system prompt is a prompt that will be used to evaluate the answer relevance of the answers to the questions.

```python 
answer_relevance_system_prompt = (
    "Given the following question, document, and answer, "
    "you must analyze the provided answer and document before determining "
    "whether the answer is relevant for the provided question. "
    "In your evaluation, you should consider whether the answer "
    "addresses all aspects of the question and provides only correct "
    "information from the document for answering the question. "
    "Output your final verdict by strictly following this format: "
    "[[Yes]]" if the answer is relevant for the given question and " 
    "[[No]]" if the answer is not relevant for the given question. "
    "Do not provide any additional explanation for your decision.\n\n"
)

```

### Answer Faithfulness System Prompt 

The answer faithfulness system prompt is a prompt that will be used to evaluate the answer faithfulness of the answers to the questions.

```python 
answer_faithfulness_system_prompt = (
    "You are an expert dialogue agent. "
    "Your task is to analyze the provided document and determine 
    "whether it is relevant for responding to the dialogue. "
    "In your evaluation, you should consider the content of the "
    "document and how it relates to the provided dialogue. "
    'Output your final verdict by strictly following this format: "[[Yes]]" 
    "if the document is relevant and "[[No]]" if the document 
    "provided is not relevant. 'Do not provide any additional 
    "explanation for your decision.\n\n"
)
```

### Debug Mode 

The debug mode is a flag that will be used to determine whether to run the evaluation in debug mode.

```python 
debug_mode = False
```

### Documents 

The documents parameter is the number of documents to be evaluated. Default is 0 which means all documents in the evaluation set will be evaluated.

```python 
documents = 0
```

### Model Choice 

The model_choice parameter is the model to be used for the evaluation. Default is GPT3.5

```python 
model_choice = "gpt-3.5-turbo-1106"
```

### Request Delay 

The request_delay parameter is the delay between requests to the API. Default is 0.

```python 
request_delay = 0
```

### VLLM 

The vllm parameter is the flag to use VLLM. Default is False.

```python 
vllm = False
```

### Host URL 

The host_url parameter is the host url to use for the evaluation. Default is None.

```python 
host_url = "None"
```

## UES/IDP Full Example

```python 

from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": nq_few_shot_prompt_for_judge_scoring.tsv, 
    "unlabeled_evaluation_set": nq_unlabeled_output.tsv,
    "model_choice": "gpt-3.5-turbo-1106",
}

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)

```