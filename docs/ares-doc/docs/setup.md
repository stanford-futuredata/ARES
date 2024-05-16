### Requirements
<a id="section2"></a>
<hr>

To implement ARES for scoring your RAG system and comparing to other RAG configurations, you need three components:​

* A human preference validation set of annotated query, document, and answer triples for the evaluation criteria (e.g. context relevance, answer faithfulness, and/or answer relevance). There should be at least 50 examples but several hundred examples is ideal.
* A set of few-shot examples for scoring context relevance, answer faithfulness, and/or answer relevance in your system
* A much larger set of unlabeled query-document-answer triples outputted by your RAG system for scoring

<a id="section3"></a>
<hr>

To get started with ARES, you'll need to set up your configuration. Below is an example of a configuration for ARES!

Copy-paste each step to see ARES in action!

<hr>

### Download datasets

<hr>

Use the following command to quickly obtain the necessary files for getting started! This includes the 'few_shot_prompt' file for judge scoring and synthetic query generation, as well as both labeled and unlabeled datasets.
```python 
wget https://raw.githubusercontent.com/stanford-futuredata/ARES/main/datasets/example_files/nq_few_shot_prompt_for_judge_scoring.tsv
wget https://raw.githubusercontent.com/stanford-futuredata/ARES/main/datasets/example_files/nq_few_shot_prompt_for_synthetic_query_generation.tsv
wget https://raw.githubusercontent.com/stanford-futuredata/ARES/main/datasets/example_files/nq_labeled_output.tsv
wget https://raw.githubusercontent.com/stanford-futuredata/ARES/main/datasets/example_files/nq_unlabeled_output.tsv
```

OPTIONAL: You can run the following command to get the full NQ dataset! (347 MB)
```python
from ares import ARES
ares = ARES() 
ares.KILT_dataset("nq")

# Fetches NQ datasets with ratios including 0.5, 0.6, 0.7, etc.
# For purposes of our quick start guide, we rename nq_ratio_0.5 to nq_unlabeled_output and nq_labeled_output.
```