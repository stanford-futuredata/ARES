<hr>

<h3>This page provides an in-depth overview of the parameters and capabilities available for synthetic data generation in ARES, allowing users to fully customize their datasets for robust testing and evaluation.</h3>

<hr>

## Synthetic Generation In-Depth Configuration 

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. Below is how the synthetic generation configuration style.

!!! note "Synthetic Generation Parameters"
    Inherently, in ARES the values past ```documents_sampled``` are not required and will use the default values if not provided.

```python 
from ares import ARES

synth_config = { 
    "document_filepaths": [<document_filepaths>],
    "few_shot_prompt_filename": few_shot_filepath,
    "synthetic_queries_filenames": [<synthetic_queries_filepaths>],
    "model_choice": <model_choice>,
    "documents_sampled": 10000,
    "model_choice": "google/flan-t5-xxl", 
    "clean_documents": False, 
    "regenerate_synth_questions": True, 
    "percentiles": 0.05, 0.25, 0.5, 0.95,
    "question_temperatures": 2.0, 1.5, 1.0, 0.5, 0.0,
    "regenerate_answers": True, 
    "number_of_negatives_added_ratio": 0.5,
    "lower_bound_for_negatives": 5,
    "number_of_contradictory_answers_added_ratio": 0.67,
    "number_of_positives_added_ratio": 0.0,
    "regenerate_embeddings": True,
    "synthetic_query_prompt": "You are an expert question-answering system. 
    You must create a question for the provided document. 
    The question must be answerable within the context of the document.\n\n"
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

### Clean Documents

A boolean indicating whether to clean the documents before generating synthetic queries. This is useful if the documents contain special characters or are in a different language.

```python
"clean_documents": True,
```

<hr>

### Regenerate Synth Questions

A boolean indicating whether to regenerate synthetic questions for each document. This is useful if the synthetic questions are not satisfactory.

```python
"regenerate_synth_questions": True,
```

<hr>

### Percentiles

A list of floats indicating the percentiles of the synthetic questions to generate. The percentiles should be between 0 and 1.

```python
"percentiles": [0.05, 0.25, 0.5, 0.95],
```

<hr>

### Question Temperatures

A list of floats indicating the temperatures of the synthetic questions to generate. The temperatures should be between 0 and 2.

```python
"question_temperatures": [2.0, 1.5, 1.0, 0.5, 0.0],
```

<hr>

### Regenerate Answers

A boolean indicating whether to regenerate synthetic answers for each document. This is useful if the synthetic answers are not satisfactory.

```python
"regenerate_answers": True,
```

<hr>

### Number of Negatives Added Ratio

A float indicating the ratio of synthetic queries to generate that are negatives. The ratio should be between 0 and 1.

```python
"number_of_negatives_added_ratio": 0.5,
```

<hr>

### Number of Contradictory Answers Added Ratio

A float indicating the ratio of synthetic queries to generate that are contradictory answers. The ratio should be between 0 and 1.

```python
"number_of_contradictory_answers_added_ratio": 0.67,
```

<hr>

### Number of Positives Added Ratio

A float indicating the ratio of synthetic queries to generate that are positives. The ratio should be between 0 and 1.

```python
"number_of_positives_added_ratio": 0.0,
```

<hr>

### Regenerate Embeddings

A boolean indicating whether to regenerate embeddings for each document. This is useful if the embeddings are not satisfactory.

```python
"regenerate_embeddings": True,
```

<hr>

### Synthetic Query Prompt

A string indicating the prompt for generating synthetic queries. The prompt should be a clear and concise explanation of the task and the format of the synthetic queries.

```python
"synthetic_query_prompt": "You are an expert question-answering system. 
    You must create a question for the provided document. 
    The question must be answerable within the context of the document.\n\n",
```

!!! note "Prompt Engineering"
    Proceed with caution when modifying the prompt, the data generation process is crucial to ARES's performance.
