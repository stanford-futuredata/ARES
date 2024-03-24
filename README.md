<h2 align="center">ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems</h2>

<p align="center">
  <a>Table of Contents:</a>
  <a href="#section1">Installation</a> |
  <a href="#section2">Requirements</a> |
  <a href="#section3">Quick Start</a> |
  <a href="#section4">Citation</a>
</p>


<p align="center">

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/release-v0.1.0-blue?style=flat&link=https%3A%2F%2Fpython.org%2F">
  </a>

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/Read-ARES%20Paper-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.09476">
  </a>

  <a href="https://ares-ai.vercel.app/">
    <img alt="Static Badge" src="https://img.shields.io/badge/Read-documentation-purple?style=flat">
  </a>

  <a href="https://colab.research.google.com/drive/1lc8Tkcair7wWZVbsdNKmfSM5rbAqOeeO#scrollTo=03609iqyArxM" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/Made%20with-Python-red?style=flat&link=https%3A%2F%2Fpython.org%2F">
  </a>

</p>


ARES is a groundbreaking framework for evaluating Retrieval-Augmented Generation (RAG) models. The automated process combines synthetic data generation with fine-tuned classifiers to efficiently assess context relevance, answer faithfulness, and answer relevance, minimizing the need for extensive human annotations. ARES employs synthetic query generation and Precision-Performance Iteration (PPI), providing accurate evaluations with statistical confidence.


### 💬 Mini Q&A
<hr>

**What does ARES assess in RAG models?**

ARES conducts a comprehensive evaluation of Retrieval-Augmented Generation (RAG) models, assessing the systems for context relevance, answer faithfulness, and answer relevance. This thorough assessment ensures a complete understanding of the performance of the RAG system.

**How does ARES automate the evaluation process?**

ARES minimizes the need for human labeling by leveraging fine-tuned classifiers and synthetic data. Its PPI component, Prediction-Powered inference, refines evaluations considering model response variability and provides statistical confidence in the results. By using fine-tuned classifiers and synthetically generated data, ARES cuts down on human labeling needs while providing accurate assessments. 

**Can ARES handle my custom RAG model?**

Yes, ARES is a model-agnostic tool that enables you to generate synthetic queries and answers from your documents. With ARES, you can evaluate these generated queries and answers from your RAG model.
​
### ⚙️ Installation
<a id="section1"></a>
<hr>
​
To install ARES, run the following commands:
​

```python

pip install ares-ai

```
​
*Optional: Initalize OpenAI or TogetherAI API key with the following command:*


```python

export OPENAI_API_KEY=<your key here>
export TOGETHER_API_KEY=<your key here>

```

### 📝 Requirements
<a id="section2"></a>
<hr>

To implement ARES for scoring your RAG system and comparing to other RAG configurations, you need three components:​

* A human preference validation set of annotated query, document, and answer triples for the evaluation criteria (e.g. context relevance, answer faithfulness, and/or answer relevance). There should be at least 50 examples but several hundred examples is ideal.
* A set of few-shot examples for scoring context relevance, answer faithfulness, and/or answer relevance in your system
* A much larger set of unlabeled query-document-answer triples outputted by your RAG system for scoring

### 🚀 Quick Start
<a id="section3"></a>
<hr>

To get started with ARES, you'll need to set up your configuration. Below is an example of a configuration for ARES!

Copy-paste each step to see ARES in action!

<hr>

Run the following to get the few-shot tsv file! 
```python 
wget https://github.com/stanford-futuredata/ARES/blob/new-dev/data/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv
```

<hr>

Run the following command to get the NQ dataset! (We use this for configuration)
```python
from ares import ARES
ares = ARES() 
ares.KILT_dataset("nq")
```

<hr>

Step 1) Run the following to see GPT 3.5's accuracy on the nq 0.5 ratio dataset!

```python
from ares import ARES

ues_idp_config = {
    # Dataset for in-domain prompts
    "in_domain_prompts_dataset": "multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    
    # Dataset for unlabeled evaluation
    "unlabeled_evaluation_set": "/datasets_v2/nq/nq_ratio_0.5.tsv", 

    # Default context relevance prompt
    "context_relevance_system_prompt": """You are an expert dialogue agent. Your task is to analyze the provided document and determine whether it 
    is relevant for responding to the dialogue. In your evaluation, you should consider the content of the document and how 
    it relates to the provided dialogue. Output your final verdict by strictly following this format: \"[[Yes]]\" 
    if the document is relevant and \"[[No]]\" if the document provided is not relevant. Do not provide any additional explanation for your decision.\n\n """,

    # Default answer relevance prompt
    "answer_relevance_system_prompt": """
    Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant 
    for the provided question. In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information 
    from the document for answering the question. Output your final verdict by strictly following this format: \"[[Yes]]\" if the answer is relevant for the given question 
    and \"[[No]]\" if the answer is not relevant for the given question. Do not provide any additional explanation for your decision.\n\n\ """,
    
    # Default answer faithfulness prompt
    "answer_faithfulness_system_prompt": """ 
    Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. 
    The answer must not offer new information beyond the context provided in the document. The answer also must not contradict information provided in the document. 
    Output your final verdict by strictly following this format: \"[[Yes]]\" if the answer is faithful to the document and \"[[No]]\" if the answer is not faithful to the document. 
    Do not provide any additional explanation for your decision.\n\n\
    """,

    "model_choice" : "gpt-3.5-turbo-0125"

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)
}
```

<hr>

Step 2) Run the following to see ARES's synthetic generation in action! 
```python

from ares import ARES

synth_config = { 
    "document_filepaths": "datasets_v2/nq/nq_ratio_0.6.tsv",
    "few_shot_prompt_filename": "datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "synthetic_queries_filename": "data/output/synthetic_queries_1.tsv",
    "documents_sampled": 10000
}

ares_module = ARES(synthetic_query_generator=synth_config)
results = ares_module.generate_synthetic_data()
print(results)
```

<hr>

Step 3) Run the following to see ARES's training classifier in action!
```python

from ares import ARES

classifier_config = {
    "classification_dataset": "output/synthetic_queries_1.tsv", 
    "validation_set": "datasets_v2/nq/nq_ratio_0.6.tsv", 
    "label_column": "Answer_Relevance_Label", 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)
```

<hr>

Step 4) Run the following to see ARES's PPI in action!
```python

from ares import ARES

ppi_config = { 
    "evaluation_datasets": ['/datasets_v2/nq/nq_ratio_0.6.tsv'], 
    "few_shot_examples_filepath": "/future/u/manihani/ARES/datasets/multirc_few_shot_prompt_for_synthetic_query_generation_v1.tsv",
    "checkpoints": ["/checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Context_Relevance_Label_ratio_0.6_reformatted_full_articles_False_validation_with_negatives_428380.pt"],
    "labels": ["Context_Relevance_Label"], 
    "GPT_scoring": False, 
    "gold_label_path": "/datasets_v2/nq/nq_ratio_0.5.tsv", 
    "swap_human_labels_for_gpt4_labels": False
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)
```
​
For more details, refer to our [documentation](https://ares-ai.vercel.app/).

## Results Replication

We include synthetic datasets for key experimental results in `synthetic_datasets`. The few-shot prompts used for generation and evaluation are included in `datasets`. We also include instructions for fine-tuning LLM judges in the paper itself. Please reach out to jonsaadfalcon@stanford.edu or manihani@stanford.edu if you have any further questions.

## Citation
<a id="section4"></a>

To cite our work, please use the following Bibtex:

````
@misc{saadfalcon2023ares,
      title={ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems}, 
      author={Jon Saad-Falcon and Omar Khattab and Christopher Potts and Matei Zaharia},
      year={2023},
      eprint={2311.09476},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
````

# Appendix
### Machine requirements and setup when not using OpenAI API
**Machine requirements**

- Over ~100 GB of available disk space
- GPU
    - Should work: A100 (e.g. `Standard_NC24ads_A100_v4` on Azure)
    - Does not work:
        - Tested on 2023-12-17 with both `Standard_NC6s_v3` and `Standard_NC12s_v3`, and ran into this error: `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 160.00 MiB (GPU 0; 15.77 GiB total capacity; 15.12 GiB already allocated; 95.44 MiB free; 15.12 GiB reserved in total by PyTorch)`


**Machine setup**

For example, on an Azure VM running Linux (ubuntu 20.04), you will need to do the following:
- Install conda
    - First set of commands (can copy-paste multiple lines)
        - `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
        - `chmod +x Miniconda3-latest-Linux-x86_64.sh`
        - `./Miniconda3-latest-Linux-x86_64.sh -b`
    - Second set of commands (can copy-paste multiple lines)
        - `export PATH="~/miniconda3/bin:$PATH"`
        - `conda init`
- Install gcc
    - `sudo apt-get -y update`
    - `sudo apt-get -y upgrade`
    - `sudo apt-get -y install build-essential`
    - `sudo apt-get -y install libpcre3-dev`
- Install NVIDIA drivers
    - `sudo apt install ubuntu-drivers-common -y`
    - `sudo ubuntu-drivers autoinstall`
    - `sudo reboot`
    - SSH in again and confirm the installation was successful by running `nvidia-smi`
- `cd` to ARES folder and follow the rest of the README
