## Local Model Execution with vLLM

ARES supports [vLLM](https://github.com/vllm-project/vllm), allowing for local execution of LLM models, offering enhanced privacy and the ability to operate ARES offline. Below are steps to use vLLM for with ARES's UES/IDP and PPI!

#### UES/IDP w/ vLLM

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

For more details, refer to our [documentation](https://ares-ai.vercel.app/).

<br>