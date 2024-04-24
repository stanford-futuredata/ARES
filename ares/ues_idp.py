import pandas as pd
import torch
import openai
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring 
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_togetherai
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring_togetherai
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring_togetherai
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_claude 
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring_claude
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring_claude
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_vllm
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring_vllm
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring_vllm
if 'ipykernel' in sys.modules:
    # We are in a Jupyter notebook or similar (uses IPython kernel)
    from tqdm.notebook import tqdm
else:
    # We are in a regular Python environment (e.g., terminal or script)
    from tqdm import tqdm

def ues_idp_config(in_domain_prompts_dataset: str, unlabeled_evaluation_set: str, context_relevance_system_prompt: str, answer_relevance_system_prompt: str, answer_faithfulness_system_prompt: str, debug_mode: bool, documents: int, model_choice: str) -> dict:

    if in_domain_prompts_dataset is not None:
            in_domain_prompts_dataset = pd.read_csv(in_domain_prompts_dataset, sep='\t')
    else:
        in_domain_prompts_dataset = None 

    if unlabeled_evaluation_set is not None:
        unlabeled_evaluation_set = pd.read_csv(unlabeled_evaluation_set, sep='\t')
    else:
        unlabeled_evaluation_set = None
    if in_domain_prompts_dataset is None and unlabeled_evaluation_set is None: 
        print(f"Error: UES and IDP are not provided")
        exit()
    if in_domain_prompts_dataset is None: 
        print(f"Error: IDP is not provided")
        exit()
    if unlabeled_evaluation_set is None: 
        print(f"Error: UES is not provided")
        exit()

    if documents > len(unlabeled_evaluation_set): 
        sys.exit("Error: documents size passed in is larger than documents present in unlabeled evaluation set")
    
    if documents == 0: 
        documents = len(unlabeled_evaluation_set)

    context_relevance_answers = unlabeled_evaluation_set["Context_Relevance_Label"].tolist()
    answer_relevance_answers = unlabeled_evaluation_set["Answer_Relevance_Label"].tolist()
    answer_faithfulness_answers = unlabeled_evaluation_set["Answer_Faithfulness_Label"].tolist()
    context_relevance_scores = []
    answer_relevance_scores = []
    answer_faithfulness_scores = []

    gpt_models = [
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613"
    ]

    claude_models = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229", 
    "claude-3-haiku-20240307", 
    ]

    required_columns = ['Query', 'Document', 'Answer']
    missing_columns = [col for col in required_columns if col not in unlabeled_evaluation_set.columns]
    if missing_columns:
        print(f"Missing columns in the DataFrame: {missing_columns}")


    if model_choice in gpt_models:
        with tqdm(total=documents, desc=f"Evaluating large subset with {model_choice}") as pbar:
            for index, row in unlabeled_evaluation_set[:documents].iterrows():
                # Extract query, document, and answer from the row
                try:
                    query = row['Query']
                except KeyError:
                    query = row['Question']
                
                try:
                    _ = in_domain_prompts_dataset.iloc[0]['Query']
                    query_id = "Query"
                except KeyError:
                    try:
                        _ = in_domain_prompts_dataset.iloc[0]['Question']
                        query_id = "Question"
                    except KeyError:
                        sys.exit("Both 'Query' and 'Question' keys are missing for the given row.")

                document = row['Document']
                answer = row['Answer']

                # Scoring
                context_score = few_shot_context_relevance_scoring(
                    context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                answer_relevance_score = few_shot_answer_relevance_scoring(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)
                
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                    # Append scores to respective lists
                context_relevance_scores.append(context_score)
                answer_relevance_scores.append(answer_relevance_score)
                answer_faithfulness_scores.append(answer_faithfulness_score)

                pbar.update(1)
    elif model_choice in claude_models:
        with tqdm(total=documents, desc=f"Evaluating large subset with {model_choice}") as pbar:
            for index, row in unlabeled_evaluation_set[:documents].iterrows():
                # Extract query, document, and answer from the row
                try:
                    query = row['Query']
                except KeyError:
                    query = row['Question']
                
                try:
                    _ = in_domain_prompts_dataset.iloc[0]['Query']
                    query_id = "Query"
                except KeyError:
                    try:
                        _ = in_domain_prompts_dataset.iloc[0]['Question']
                        query_id = "Question"
                    except KeyError:
                        sys.exit("Both 'Query' and 'Question' keys are missing for the given row.")

                document = row['Document']
                answer = row['Answer']

                # Scoring
                context_score = few_shot_context_relevance_scoring_claude(
                    context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                answer_relevance_score = few_shot_answer_relevance_scoring_claude(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)
                
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring_claude(answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                    # Append scores to respective lists
                context_relevance_scores.append(context_score)
                answer_relevance_scores.append(answer_relevance_score)
                answer_faithfulness_scores.append(answer_faithfulness_score)

                pbar.update(1)
    else: 
        with tqdm(total=documents, desc=f"Evaluating large subset with {model_choice}") as pbar:
            for index, row in unlabeled_evaluation_set[:documents].iterrows():
                # Extract query, document, and answer from the row
                try:
                    query = row['Query']
                except KeyError:
                    query = row['Question']
                
                try:
                    _ = in_domain_prompts_dataset.iloc[0]['Query']
                    query_id = "Query"
                except KeyError:
                    try:
                        _ = in_domain_prompts_dataset.iloc[0]['Question']
                        query_id = "Question"
                    except KeyError:
                        sys.exit("Both 'Query' and 'Question' keys are missing for the given row.")

                document = row['Document']
                answer = row['Answer']

                # Scoring
                context_score = few_shot_context_relevance_scoring_togetherai(
                    context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                answer_relevance_score = few_shot_answer_relevance_scoring_togetherai(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)
                
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring_togetherai(answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, in_domain_prompts_dataset)

                # Append scores to respective lists
                context_relevance_scores.append(context_score)
                answer_relevance_scores.append(answer_relevance_score)
                answer_faithfulness_scores.append(answer_faithfulness_score)

                pbar.update(1)

    # Compile results into a dictionary
    return {
        "Context Relevance Scores": round(sum(context_relevance_scores)/len(context_relevance_scores), 3),
        "Answer Faithfulness Scores": round(sum(answer_faithfulness_scores)/len(answer_faithfulness_answers), 3),
        "Answer Relevance Scores": round(sum(answer_relevance_scores)/len(answer_relevance_answers), 3)
    }