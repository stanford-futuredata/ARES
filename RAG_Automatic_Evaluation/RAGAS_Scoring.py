
from ragas import evaluate
from datasets import Dataset
import os
import pandas as pd
import scipy.stats as stats

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

#os.environ["OPENAI_API_KEY"] = "your-openai-key"

evaluation_datasets = ['../datasets_v2/nq/ratio_0.7_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.725_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.75_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.775_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.8_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.825_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.85_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.875_reformatted_full_articles_False_validation_with_negatives.tsv', '../datasets_v2/nq/ratio_0.9_reformatted_full_articles_False_validation_with_negatives.tsv']
evaluation_datasets = evaluation_datasets[:2]
correct_ranking = [i for i in range(0, len(evaluation_datasets))]

labels = ["Context Relevance", "Answer Faithfulness", "Answer Relevance"]
context_scores = []
answer_relevance_scores = []
answer_faithfulness_scores = []

for evaluation_dataset in evaluation_datasets:
    dataset = pd.read_csv(evaluation_dataset.replace("../", "../../ColBERT-FM/"), sep="\t")
    dataset = dataset[:2000]

    def string_to_list(text):
        return [text]

    # Apply the function to the 'text_column' to convert it to a column of lists
    dataset['contexts'] = dataset['Document'].apply(string_to_list)

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.rename_column("Query", "question")
    dataset = dataset.rename_column("Answer", "answer")
    #dataset = dataset.rename_column("Document", "contexts")

    results = evaluate(dataset, metrics=[
        context_precision,
        #faithfulness,
        answer_relevancy,
    ])

    print("Results for " + evaluation_dataset)
    print(results)

    context_scores.append(results['context_precision'])
    answer_relevance_scores.append(results['answer_relevancy'])
    answer_faithfulness_scores.append(results['faithfulness'])

####################################

for label, scores in zip(labels, [context_scores, answer_relevance_scores, answer_faithfulness_scores]):

    indexed_list = list(enumerate(scores))
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    sorted_indices = [index for index, _ in sorted_list]
    tau, p_value = stats.kendalltau(correct_ranking, sorted_indices)

    print("--------------------------------------------------")
    print(label + " Scoring")
    print("Correct Ranking v. LLM-as-a-Judge Ranking")
    print(correct_ranking)
    print(sorted_indices)
    print("Kendall's Tau: " + str(tau))
    print("P-Value: " + str(p_value))
    print("--------------------------------------------------")


# prepare your huggingface dataset in the format
# Dataset({
#     features: ['question', 'contexts', 'answer', 'ground_truths'],
#     num_rows: 25
# })

#dataset: Dataset

#results = evaluate(dataset)
# {'ragas_score': 0.860, 'context_precision': 0.817,
# 'faithfulness': 0.892, 'answer_relevancy': 0.874}

