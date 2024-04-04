<h3>This page shows you how to automatically create synthetic datasets that closely mimic real-world scenarios for robust RAG testing.</h3>

<hr>

## Configure OpenAI API Key. 

```
export OPENAI_API_KEY=<your key here>
```

<hr>

## Synth Gen Configuration 

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. Below is how the synthetic generation configuration style.

```python 
from ares import ARES

synth_config = { 
    "document_filepaths": [<document_filepaths>],
    "few_shot_prompt_filename": few_shot_filepath,
    "synthetic_queries_filenames": [<synthetic_queries_filepaths>],
    "model_choice": <model_choice>,
    "documents_sampled": 10000
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)
```
<hr>

### Document File Path(s)

A single or list of file paths to the document(s) you want to use for generating synthetic queries. If
given a list of file paths, each file path should point to a file containing raw text from which ARES can derive context for the synthetic queries. 

```python 
"document_filepaths": ["/data/datasets_v2/nq/nq_ratio_0.5_.tsv"], 
```
Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets_v2/nq) for document example file used. 

<hr>

### Few-Shot Prompt File Path

This refers to the file paths for a few-shot prompt file that provide examples of queries and answers for ARES to learn from. Few-shot learning uses a small amount of labeled training data to guide the generation of synthetic queries.

```python 
"few_shot_prompt_filename": "data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
```

Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets) for few-shot file example used. 

<hr>

### Synthetic Queries Filepath

A list of file paths where the generated synthetic queries will be saved. These files will store the queries created by ARES for use in training or evaluation. 

!!! note "NOTE - List Size Verification"
    Ensure the synthetic queries file paths list matches the document file paths list in size for consistency.

```python
"synthetic_queries_filenames": ["/output/synthetic_queries_1.tsv"],
```

<hr>

### Model Choice

Specifies the pre-trained language model to create the synthetic data. By default, ARES uses "google/flan-t5-xxl". You can replace this with any Hugging Face model suitable for your task.

```python
 "model_choice": "google/flan-t5-xxl",
```

<hr>

### Documents Sampled

An integer indicating how many documents to sample from your dataset when generating synthetic queries. Sampling can help speed up processing and manage computational resources. Choose a value that represents a large enough sample to generate meaningful synthetic queries, but not so large as to make processing infeasible. ARES will automatically filter documents

!!! note "NOTE - Document Filter"
    ARES will automatically filter documents less than 50 words

```python
"documents_sampled": 10000,
```

<hr>

## Synthetic Generation Configuration: Full Example

```python
from ares import ARES

synth_config = { 
    "document_filepaths": ["/data/datasets_v2/nq/nq_ratio_0.5_.tsv"],
    "few_shot_prompt_filename": "data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "synthetic_queries_filenames": ["/output/synthetic_queries_1.tsv"],
    "model_choice": "google/flan-t5-xxl",
    "documents_sampled": 10000
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)
```




