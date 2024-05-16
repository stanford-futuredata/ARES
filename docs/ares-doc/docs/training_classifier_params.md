<hr> 

<h3>This page provides an in-depth overview of the parameters and capabilities available for the training classifier in ARES, allowing users to fully customize the training pipeline in ARES.</h3>

<hr>

## Training Classifier Configuration

The synth_config dictionary is a configuration object that sets up ARES for generating synthetic queries based on a given dataset. Below is how the training classifier configuration style.

!!! note "Training Classifier Parameters"
    Inherently, in ARES the values past ```learning_rate``` are not required and will use the default values if not provided.
    <b>Review values '''assigned_batch_size''' and '''gradient_accumulation_multiplier''', they are dependent on your system.</b>

```python 

    classifier_config = {
    "classification_dataset": [<classification_dataset_filepath>],
    "validation_set": <test_set_selection_filepath>, 
    "label_column": [<labels>], 
    "model_choice": "microsoft/deberta-v3-large", 
    "num_epochs": 10, 
    "patience_value": 3, 
    "learning_rate": 5e-6,
    "training_dataset_path": "path/to/training/dataset.tsv",
    "validation_dataset_path":  "path/to/validation/dataset.tsv",
    "validation_set_scoring": True,
    "assigned_batch_size": 1,
    "gradient_accumulation_multiplier": 32,
    "number_of_runs": 1,
    "num_warmup_steps": 100,
    "training_row_limit": -1,
    "validation_row_limit": -1
}

```

<hr>

### Classification Dataset
Generated from the ARES [synthetic generator](synth_gen.md), here you should provide a list of file paths or an individual filepath to your labeled dataset used for training the classifier. The dataset should include text data and corresponding labels for supervised learning.

```python
"classification_dataset": ["output/synthetic_queries_1.tsv"],
```

!!! note "# of Training Datasets"
    Ensure the number of training datasets provided aligns with number of validation datasets.

<hr>

### Validation Set

Provide the file path to your validation set for evaluating the classifier's performance. This should be separate from the training data to ensure an unbiased assessment.

```python
"validation_set": "/data/datasets_v2/nq/nq_ratio_0.6_.tsv"
```

!!! note "# of Training Datasets"
    Ensure the number of validation datasets provided aligns with number of training datasets.

<hr>


Link to [ARES Github Repo](https://github.com/stanford-futuredata/ARES/tree/new-dev/data/datasets_v2/nq) for test set selection file example used. 

### Label Column(s)

List the column name(s) in your dataset that contain the label(s). These are the targets your classifier will predict.

```python
"label_column": ["Conmtext_Relevance_Label"], 
```

<hr>

### Model Choice

Specifies the pre-trained language model to fine-tune for classification. By default, ARES uses "microsoft/deberta-v3-large". You can replace this with any Hugging Face model suitable for your task.

```python
 "model_choice": "google/flan-t5-xxl",
```

<hr>

### Num Epochs

Determines the number of training epochs, which is the number of times the learning algorithm will work through the entire training dataset.

```python
"num_epochs": 10, 
```

<hr>

### Patience Value

This is used in early stopping to prevent overfitting. It's the number of epochs with no improvement on the validation set after which training will be stopped.

```python
"patience_value": 3, 
```

<hr>

### Learning Rate
Sets the initial learning rate for the optimizer. This is a crucial hyperparameter that controls the adjustment of model weights during training. 

```python
 "learning_rate": 5e-6
```

<hr>

### Training Dataset Path

If more than 1 training dataset is provided, the classifier will combine all the datasets into one dataset path and train on all of them. In this case, please provide path to save the combined training dataset.

```python
"training_dataset_path": "path/to/training/dataset.tsv"
```

<hr> 

### Validation Dataset Path

If more than 1 validation dataset is provided, the classifier will combine all the datasets into one dataset path and utilize all of them for validation. In this case, please provide path to save the combined validation dataset.

```python 
"validation_dataset_path": "path/to/validation/dataset.tsv"
```

<hr>

### Validation Set Scoring

If True, the classifier will evaluate the model on the validation set after each epoch. If False, the classifier will only evaluate the model on the test set after the final epoch.

```python 
"validation_set_scoring": True,
```

<hr>

### Assigned Batch Size

The batch size for training. This is a crucial hyperparameter that controls the number of samples processed in each iteration.

```python 
"assigned_batch_size": 1,
```

<hr>

### Gradient Accumulation Multiplier

The number of steps to accumulate the gradients before performing a backward pass. This is a crucial hyperparameter that controls the number of steps to accumulate the gradients before performing a backward pass.

```python 
"gradient_accumulation_multiplier": 32,
```

<hr>

### Number of Runs

The number of times to run the training process. This is a crucial hyperparameter that controls the number of times to run the training process.

```python 
"number_of_runs": 1,
```

<hr>

### Num Warmup Steps

The number of steps to warm up the learning rate. This is a crucial hyperparameter that controls the number of steps to warm up the learning rate.

```python 
"num_warmup_steps": 100,
```

<hr>

### Training Row Limit

The number of rows to limit the training dataset to. This is a crucial hyperparameter that controls the number of rows to limit the training dataset to.

```python 
"training_row_limit": -1,
```

<hr>

### Validation Row Limit

The number of rows to limit the validation dataset to. This is a crucial hyperparameter that controls the number of rows to limit the validation dataset to.

```python 
"validation_row_limit": -1,
```

<hr>

