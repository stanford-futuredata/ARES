
from ragas import evaluate
from datasets import Dataset
import os
import pandas as pd

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

#os.environ["OPENAI_API_KEY"] = "your-openai-key"

evaluation_datasets = ['../datasets_v2/nq/ratio_0.7_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.725_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.75_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.775_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.8_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.825_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.85_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.875_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.9_reformatted_full_articles_False_validation_with_negatives.tsv']
    
for evaluation_dataset in evaluation_datasets[:1]:
    dataset = pd.read_csv(evaluation_dataset.replace("../", "../../ColBERT-FM/"), sep="\t")
    dataset = dataset[:100]

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.rename_column("Query", "question")
    dataset = dataset.rename_column("Answer", "answer")
    dataset = dataset.rename_column("Document", "contexts")

    results = evaluate(dataset, metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
    ])
    print("Results for " + evaluation_dataset)
    print(results)


# prepare your huggingface dataset in the format
# Dataset({
#     features: ['question', 'contexts', 'answer', 'ground_truths'],
#     num_rows: 25
# })

#dataset: Dataset

#results = evaluate(dataset)
# {'ragas_score': 0.860, 'context_precision': 0.817,
# 'faithfulness': 0.892, 'answer_relevancy': 0.874}

