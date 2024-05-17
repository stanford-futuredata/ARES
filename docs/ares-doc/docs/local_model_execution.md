<hr>

ARES provides full support to run models locally and leverage them in ARES's RAG evaluation framework. This guide will walk you through the steps to set up your local environment for model execution.

<hr>

### Setup

ARES supports [vLLM](https://docs.vllm.ai/en/latest/), enabling you to run models locally and leverage them in ARES's RAG evaluation framework. To get started, please check your model is running locally and is accessible via HTTP.

<hr>

### Integration 

#### Running vLLM on UES/IDP

Toggle vLLM to True and specify the host url you are running vLLM on. 

!!! note "Host URL followed by /v1" 
    
    Make sure to include "/v1" in the host url.
    

```python
from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "unlabeled_evaluation_set": "nq_unlabeled_output.tsv", 
    "model_choice": "meta-llama/Llama-2-13b-hf", # Specify vLLM model
    "vllm": True, # Toggle vLLM to True 
    "host_url": "http://0.0.0.0:8000/v1" # Replace with server hosting model followed by "/v1"
} 

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)
```

<hr>

#### PPI w/ vLLM

Toggle vLLM to True and specify the host url you are running vLLM on. 

!!! note "Host URL followed by /v1" 
    
    Make sure to include "/v1" in the host url.
    

```python
from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['nq_unabeled_output.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "llm_judge": "meta-llama/Llama-2-13b-hf", # Specify vLLM model
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv",
    "vllm": True, # Toggle vLLM to True 
    "host_url": "http://0.0.0.0:8000/v1" # Replace with server hosting model followed by "/v1"
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)
```

<br>
