# %% Before running the code, make sure to install the required libraries in a new virtualenv using Python 3.10 (3.11 and newer won't work)
from ares import ARES
import requests
import os

# %% Before running the code, download the checkpoints from the following URLs
# context_relevance_model_url = (
#     "https://drive.google.com/file/d/15poFyeoqdnaNZVjl41HllL2213DKyZjH/view?usp=sharing"
# )
# answer_relevance_model_url = (
#     "https://drive.google.com/file/d/1wGcgELBfnCGqXlPEbpPmf7LJ53DPWVXI/view?usp=sharing"
# )

context_relevance_model_path = os.path.join(
    "checkpoints", "ares_context_relevance_general_checkpoint_V1.1.pt"
)
answer_relevance_model_path = os.path.join(
    "checkpoints", "ares_answer_relevance_general_checkpoint_V1.1.pt"
)


# %%
ppi_config = {
    "evaluation_datasets": [
        os.path.join("datasets", "eval_datasets", "nq", "nq_ratio_0.55.tsv"),
        os.path.join("datasets", "eval_datasets", "nq", "nq_ratio_0.65.tsv"),
        os.path.join("datasets", "eval_datasets", "nq", "nq_ratio_0.7.tsv"),
    ],
    "few_shot_examples_filepath": os.path.join("datasets", "few_shot_datasets", "judge_scoring", "nq_few_shot_prompt_for_judge_scoring.tsv"),
    "checkpoints": [context_relevance_model_path, answer_relevance_model_path],
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Relevance_Label"],
    # Based on the references in the repository, 
    # we will use nq_ratio_0.5.tsv as nq_labeled_output, and as a gold label path
    "gold_label_path": os.path.join(
        "datasets", "eval_datasets", "nq", "nq_ratio_0.5.tsv"
    ),
}

ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)

# %%