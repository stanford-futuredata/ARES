<h2 align="center">ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems</h2>


<div style="text-align:center;">

<div style="text-align:center;">
Table of Contents
</div>

**[Installation](#section1)** | **[Requirements](#section2)** | **[Quick Start](#section3)** | **[Citation](#section4)**

</div>

<p align="center">

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/release-v0.1.0-blue?style=flat&link=https%3A%2F%2Fpython.org%2F">
  </a>

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/Read-ARES%20Paper-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.09476">
  </a>

  <a href="https://ares-ai.vercel.app/">
    <img alt="Static Badge" src="https://img.shields.io/badge/read-documentation-purple?style=plastic">
  </a>

  <a href="https://colab.research.google.com/drive/1lc8Tkcair7wWZVbsdNKmfSM5rbAqOeeO#scrollTo=03609iqyArxM" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

  <a>
  <img alt="Static Badge" src="https://img.shields.io/badge/Made%20with-Python-red?style=flat&link=https%3A%2F%2Fpython.org%2F">
  </a>

</p>


ARES is a groundbreaking framework for evaluating Retrieval-Augmented Generation (RAG) models. The automated process combines synthetic data generation with fine-tuned classifiers to efficiently assess context relevance, answer faithfulness, and answer relevance, minimizing the need for extensive human annotations. ARES employs synthetic query generation and Precision-Performance Iteration (PPI), providing accurate evaluations with statistical confidence.

---
​
### ⚙️ Installation
<a id="section1"></a>
<hr>
​
To install the necessary dependencies, run the following commands:
​
````
pip install ares-ai
````
​
Optional: Initalize OpenAI or TogetherAI API key with the following command:
````
export OPENAI_API_KEY=<your key here>
export TOGETHER_API_KEY=<your key here>
````

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

To get started with ARES, you'll need to set up your configuration. Below is is an example of how to structure your configuration for ARES.

```python

from ares import ARES

synth_config = { 
    "document_filepaths": [<document_filepath>], 
    "few_shot_prompt_filename": <few_shot_filepath>, 
    "synthetic_queries_filenames": [<synthetic_queries_filepath>],
    "model_choice": <model_choice>, # Default model is "microsoft/deberta-v3-large"
    "documents_sampled": 10000 
}

classifier_config = {
    "classification_dataset": [<classification_dataset_filepath>],
    "test_set_selection": <test_set_selection_filepath>, 
    "label_column": [<labels>], 
    "model_choice": <model_choice>, # Default model is "microsoft/deberta-v3-large"
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

ppi_config = { 
    "evaluation_datasets": [<eval_dataset_filepath>],
    "few_shot_examples_filepath": <few_shot_filepath>,
    "checkpoints": [<checkpoint_filepath>],
    "labels": [<labels>], 
    "model_choice": <model_choice>, # Default model is "microsoft/deberta-v3-large"
    "GPT_scoring": <True or False>, 
    "gold_label_path": <gold_label_filepath>, 
    "swap_human_labels_for_gpt4_labels": False
}

ares_module = ARES(synthetic_query_generator=synth_config, 
classifier_model=classifier_config, ppi=ppi_config)
results = ares_module.run()
print(results)

```
​
Refer to [documentation](https://ares-ai.vercel.app/) to learn more!

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
