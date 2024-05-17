<hr>

## Introduction 

ARES is an automatic RAG evaluation framework, and as such, it requires a few datasets to run. 

The following are the datasets that ARES requires:

1. In-domain prompts dataset
2. Unlabeled evaluation set
3. Labeled evaluation set (Optional, as ARES can create a lebeled evaluation set using machine labels in PPI! )

<hr>

## KILT Huggingface Dataset

To run ARES, you can evaluate it on any unlabeled evaluation set. However, if you would like to further test ARES, we have provided a filter to retrieve the KILT Huggingface dataset. 

To load the dataset, use the following code:

!!! note "Specify dataset name"

    Specify the name of the dataset you would like to use. Feel free to choose 
    any including "nq", "hotpotqa", "wow", and "fever". 

```python
from ares import ARES

dataset = ares.KILT_Dataset(<specify dataset name>)

# Specify "nq", "hotpotqa", "wow", or "fever"
```

In the dataset, you will retrieve different ratios of testing data, ensuring a diverse set of evaluation metrics. 

<hr>

## SuperGLUE Huggingface Dataset

Futhermore, we have provided a filter to retrieve the SuperGLUE Huggingface dataset. 

To load the dataset, use the following code:

!!! note "Specify dataset name"

    Specify the name of the dataset you would like to use. Feel free to choose 
    any including "record", "rte", "boolq", or "multirc".

```python
from ares import ARES

dataset = ares.superGlue_dataset(<specify dataset name>)

# Specify "record", "rte", "boolq", or "multirc"
```

<hr>

## Remarks

The provided datasets can be used for conducting any tests you would like on ARES. The ratios represent the ground truth accuracies of the datasets. We have provided and will continue to curate tutorials here utilizing these datasets, showcasing ARES's robust RAG evaluations. 



