<h3>This pages teach you how to train high-precision classifiers to determine the relevance and faithfulness of RAG outputs</h3>

<hr>

## Configure OpenAI API Key. 

```
export OPENAI_API_KEY=<your key here>
```

<hr>

## Training Classifier Configuration

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. Below is how the training classifier configuration style.


```python 
from ares import ARES

classifier_config = {
    "classification_dataset": [<classification_dataset_filepath>],
    "test_set_selection": <test_set_selection_filepath>, 
    "label_column": [<labels>], 
    "model_choice": "microsoft/deberta-v3-large", # Default model is "microsoft/deberta-v3-large"
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)

```

### Classification Dataset
Generated from the ARES [synthetic generator](synth_gen.md), here you should provide a list of file paths or an individual filepath to your labeled dataset used for training the classifier. The dataset should include text data and corresponding labels for supervised learning.

```python
"classification_dataset": ["output/synthetic_queries_1.tsv"],
```

### Test Set Selection

Provide the file path to your test set for evaluating the classifier's performance. This should be separate from the training data to ensure an unbiased assessment.

```python
"test_set_selection": "/data/datasets_v2/nq/nq_ratio_0.6_.tsv"
```

Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets_v2/nq) for test set selection file example used. 

### Label Column(s)

List the column name(s) in your dataset that contain the label(s). These are the targets your classifier will predict.

```python
"label_column": ["Conmtext_Relevance_Label"], 
```

### Model Choice

Specifies the pre-trained language model to fine-tune for classification. By default, ARES uses "microsoft/deberta-v3-large". You can replace this with any Hugging Face model suitable for your task.

```python
 "model_choice": "google/flan-t5-xxl",
```

### Num Epochs

Determines the number of training epochs, which is the number of times the learning algorithm will work through the entire training dataset.

```python
"num_epochs": 10, 
```

### Patience Value

This is used in early stopping to prevent overfitting. It's the number of epochs with no improvement on the validation set after which training will be stopped.

```python
"patience_value": 3, 
```

### Learning Rate
Sets the initial learning rate for the optimizer. This is a crucial hyperparameter that controls the adjustment of model weights during training. 

```python
 "learning_rate": 5e-6
```

## Training Classifier Configuration: Full Example

```python
from ares import ARES

classifier_config = {
    "classification_dataset": ["output/synthetic_queries_1.tsv"], 
    "test_set_selection": "./datasets_v2/nq/ratio_0.5_reformatted_full_articles_False_validation_with_negatives.tsv",
    "label_column": ["Context_Relevance_Label"], 
    "model_choice": "microsoft/deberta-v3-large",
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6
}

ares = ARES(classifier_model=classifier_config)
results = ares.train_classifier()
print(results)
```

