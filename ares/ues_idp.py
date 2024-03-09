import pandas as pd
import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring 
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_context_relevance_scoring_local
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_faithfulness_scoring_local
from ares.RAG_Automatic_Evaluation.Evaluation_Functions import few_shot_answer_relevance_scoring_local

def ues_idp_config(in_domain_prompts_dataset: pd.DataFrame, unlabeled_evaluation_set: pd.DataFrame, model_choice: str) -> dict:
    #####
    context_relevance_system_prompt = "You are an expert dialogue agent."
    context_relevance_system_prompt += "Your task is to analyze the provided document and determine whether it is relevant for responding to the dialogue. "
    context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
    context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
    context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    context_relevance_system_prompt = context_relevance_system_prompt
    #####
    answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
    answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
    answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
    answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    answer_relevance_system_prompt = answer_relevance_system_prompt
    #####
    answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
    answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
    answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
    answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
    answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
    answer_faithfulness_system_prompt = answer_faithfulness_system_prompt

    context_relevance_answers = unlabeled_evaluation_set["Context_Relevance_Label"].tolist()
    answer_relevance_answers = unlabeled_evaluation_set["Answer_Relevance_Label"].tolist()
    answer_faithfulness_answers = unlabeled_evaluation_set["Answer_Faithfulness_Label"].tolist()
    context_relevance_scores = []
    answer_relevance_scores = []
    answer_faithfulness_scores = []

    required_columns = ['Query', 'Document', 'Answer', 'Context_Relevance_Label', 'Answer_Relevance_Label', 'Answer_Faithfulness_Label']
    missing_columns = [col for col in required_columns if col not in unlabeled_evaluation_set.columns]
    if missing_columns:
        print(f"Missing columns in the DataFrame: {missing_columns}")

    if model_choice == "gpt-3.5-turbo-1106":
        for index, row in unlabeled_evaluation_set.iterrows():
            # Extract query, document, and answer from the row
            query = row['Query']
            document = row['Document']
            answer = row['Answer']

            # Scoring
            context_score = few_shot_context_relevance_scoring(
                context_relevance_system_prompt, query, document, model_choice, in_domain_prompts_dataset)

            answer_relevance_score = few_shot_answer_relevance_scoring(
                answer_relevance_system_prompt, query, document, answer, model_choice, in_domain_prompts_dataset)
            
            answer_faithfulness_score = few_shot_answer_faithfulness_scoring(answer_faithfulness_system_prompt, query, document, answer, model_choice, in_domain_prompts_dataset)

                # Append scores to respective lists
            context_relevance_scores.append(context_score)
            answer_relevance_scores.append(answer_relevance_score)
            answer_faithfulness_scores.append(answer_faithfulness_score)
    else: 
        for index, row in unlabeled_evaluation_set[:100].iterrows():
            # Extract query, document, and answer from the row
            query = row['Query']
            document = row['Document']
            answer = row['Answer']

            # Scoring
            context_score = few_shot_context_relevance_scoring_local(
                context_relevance_system_prompt, query, document, model_choice, in_domain_prompts_dataset)

            answer_relevance_score = few_shot_answer_relevance_scoring_local(
                answer_relevance_system_prompt, query, document, answer, model_choice, in_domain_prompts_dataset)
            
            answer_faithfulness_score = few_shot_answer_faithfulness_scoring_local(answer_faithfulness_system_prompt, query, document, answer, model_choice, in_domain_prompts_dataset)

            # Append scores to respective lists
            context_relevance_scores.append(context_score)
            answer_relevance_scores.append(answer_relevance_score)
            answer_faithfulness_scores.append(answer_faithfulness_score)

    # Compile results into a dictionary
    return {
        "Context Relevance Scores": round(sum(context_relevance_scores)/len(context_relevance_scores), 3),
        "Answer Faithfulness Scores": round(sum(answer_faithfulness_scores)/len(answer_faithfulness_answers), 3),
        "Answer Relevance Scores": round(sum(answer_relevance_scores)/len(answer_relevance_answers), 3)
    }