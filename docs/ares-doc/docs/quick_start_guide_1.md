<hr>

## Overview

!!! note "Required Datasets"
    Ensure you have installed all necessary datasets from the setup page [here](setup.md)


This quick start guide demonstrates how to run an evaluation using a large language model (LLM) (in this case, GPT-3.5) on an unlabeled evaluation set with in-domain prompting. We will compare the results of running the evaluation with the LLM model alone and in conjunction with ARES's Prediction Powered Inference (PPI).

By following this guide, you will see how ARES's PPI significantly enhances the performance and accuracy of the evaluation. Just copy-paste as you go to see ARES in action! Below is an example of a configuration for ARES:

<hr>

#### Step 1) Run the following to retrive the UES/IDP scores with GPT3.5!

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

#### Step 2) Run the following to retrieve ARES's PPI scores with GPT3.5!


```python
ppi_config = { 
    "evaluation_datasets": ['nq_unlabeled_output.tsv'], 
    "few_shot_examples_filepath": "nq_few_shot_prompt_for_judge_scoring.tsv",
    "llm_judge": "gpt-3.5-turbo-1106",
    "labels": ["Context_Relevance_Label"], 
    "gold_label_path": "nq_labeled_output.tsv", 
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)
```

<hr>