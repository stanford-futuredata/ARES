<h3>This page shows you how to automatically create synthetic datasets that closely mimic real-world scenarios for robust RAG testing.</h3>

<hr>

## Configure OpenAI API Key. 

```
export OPENAI_API_KEY=<your key here>
```

<hr>

## Synth Gen Configuration 

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. To utilize this configuration, first, import the ARES class from the ares module.

```python 
from ares import ARES

synth_config = { 
    "document_filepaths": [<document_filepaths>],
    "few_shot_prompt_filename": [few_shot_filepaths],
    "synthetic_queries_filenames": [<synthetic_queries_filepaths>],
    "documents_sampled": 10000
}
```

## Document File Path(s)

A single or list of file paths to the document(s) you want to use for generating synthetic queries. If
given a list of file paths, each file path should point to a file containing raw text from which ARES can derive context for the synthetic queries. 

```python 
"document_filepaths": ["/data/datasets_v2/nq/nq_ratio_0.5_.tsv"], 
```
Link to example file used!

## Document File Path

```python 
"few_shot_prompt_filename": [few_shot_filepaths]
```







