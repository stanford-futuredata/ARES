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
failed_extraction_count = {'failed': 0}

if 'ipykernel' in sys.modules:
    # We are in a Jupyter notebook or similar (uses IPython kernel)
    from tqdm.notebook import tqdm
else:
    # We are in a regular Python environment (e.g., terminal or script)
    from tqdm import tqdm

def validate_inputs(vllm: bool, host_url: str, in_domain_prompts_dataset: str, unlabeled_evaluation_set: str, documents: int) -> tuple:
    """
    Validates the input parameters for the evaluation process.

    Parameters:
    vllm (bool): Indicates if vLLM is being used.
    host_url (str): The URL of the host if vLLM is used.
    in_domain_prompts_dataset (str): Path to the in-domain prompts dataset file.
    unlabeled_evaluation_set (str): Path to the unlabeled evaluation set file.
    documents (int): Number of documents to be processed.

    Returns:
    tuple: A tuple containing the in-domain prompts dataset and the unlabeled evaluation set as pandas DataFrames.

    Raises:
    ValueError: If the input parameters are not valid.
    SystemExit: If the documents size is larger than the documents present in the unlabeled evaluation set.
    """

    # Validate vLLM and host_url relationship
    if vllm and host_url == "None":
        raise ValueError("host_url must be provided if vllm is True.")
    elif not vllm and host_url != "None":
        raise ValueError("vLLM must be set to True if host_url is provided")

    # Load in-domain prompts dataset if provided
    if in_domain_prompts_dataset is not None:
        in_domain_prompts_dataset = pd.read_csv(in_domain_prompts_dataset, sep='\t', engine="python", on_bad_lines='skip')
    else:
        in_domain_prompts_dataset = None

    # Load unlabeled evaluation set if provided
    if unlabeled_evaluation_set is not None:
        unlabeled_evaluation_set = pd.read_csv(unlabeled_evaluation_set, sep='\t', engine="python", on_bad_lines='skip')
    else:
        unlabeled_evaluation_set = None

    # Check if both datasets are provided
    if in_domain_prompts_dataset is None and unlabeled_evaluation_set is None:
        print("Error: UES and IDP are not provided")
        exit()
    if in_domain_prompts_dataset is None:
        print("Error: IDP is not provided")
        exit()
    if unlabeled_evaluation_set is None:
        print("Error: UES is not provided")
        exit()

    # Set documents to the length of the unlabeled evaluation set if documents is 0
    if documents == 0 or documents == "None":
        documents = len(unlabeled_evaluation_set)

    # Validate documents size
    if documents > len(unlabeled_evaluation_set):
        sys.exit("Error: documents size passed in is larger than documents present in unlabeled evaluation set")

    return in_domain_prompts_dataset, unlabeled_evaluation_set, documents

def extract_query(row: pd.Series, in_domain_prompts_dataset: pd.DataFrame) -> tuple:
    """
    Extracts the query from a given row and determines the query identifier.

    Parameters:
    row (pd.Series): A row from the dataset containing the query or question.
    in_domain_prompts_dataset (pd.DataFrame): The in-domain prompts dataset to check for the query identifier.

    Returns:
    tuple: A tuple containing the extracted query and the query identifier.

    Raises:
    SystemExit: If both 'Query' and 'Question' keys are missing in the given row.
    """
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
    
    return query, query_id

def score_row(row: pd.Series, 
              in_domain_prompts_dataset: pd.DataFrame, 
              context_relevance_system_prompt: str, 
              answer_relevance_system_prompt: str, 
              answer_faithfulness_system_prompt: str, 
              model_choice: str, 
              query_id: str, 
              debug_mode: bool, 
              request_delay: int, 
              vllm: bool, 
              host_url: str) -> tuple:
    """
    Scores a row based on context relevance, answer relevance, and answer faithfulness.

    Parameters:
    row (pd.Series): A row from the dataset containing the query, document, and answer.
    in_domain_prompts_dataset (pd.DataFrame): The in-domain prompts dataset.
    context_relevance_system_prompt (str): The system prompt for context relevance scoring.
    answer_relevance_system_prompt (str): The system prompt for answer relevance scoring.
    answer_faithfulness_system_prompt (str): The system prompt for answer faithfulness scoring.
    model_choice (str): The model choice for scoring (e.g., "gpt", "claude").
    query_id (str): The query identifier.
    debug_mode (bool): Flag to enable or disable debug mode.
    request_delay (int): The delay between requests.
    vllm (bool): Flag to indicate if vllm is used.
    host_url (str): The host URL for the scoring service.

    Returns:
    tuple: A tuple containing context score, answer relevance score, and answer faithfulness score.
    """
    query, query_id = extract_query(row, in_domain_prompts_dataset)
    document = row['Document']
    answer = row['Answer']

    if vllm:
        context_score = few_shot_context_relevance_scoring_vllm(
            context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, in_domain_prompts_dataset)
        
        if context_score == 0:
            return 0, 0, 0
        else:
            answer_relevance_score = few_shot_answer_relevance_scoring_vllm(
                answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, in_domain_prompts_dataset)
            answer_faithfulness_score = few_shot_answer_faithfulness_scoring_vllm(
                answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, host_url, request_delay, failed_extraction_count, in_domain_prompts_dataset)
    else:
        if "gpt" in model_choice:
            context_score = few_shot_context_relevance_scoring(
                context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
        elif "claude" in model_choice:
            context_score = few_shot_context_relevance_scoring_claude(
                context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
        else:
            context_score = few_shot_context_relevance_scoring_togetherai(
                context_relevance_system_prompt, query, document, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
        
        if context_score == 0:
            return 0, 0, 0
        else:
            if "gpt" in model_choice:
                answer_relevance_score = few_shot_answer_relevance_scoring(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring(
                    answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
            elif "claude" in model_choice:
                answer_relevance_score = few_shot_answer_relevance_scoring_claude(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring_claude(
                    answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
            else:
                answer_relevance_score = few_shot_answer_relevance_scoring_togetherai(
                    answer_relevance_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
                answer_faithfulness_score = few_shot_answer_faithfulness_scoring_togetherai(
                    answer_faithfulness_system_prompt, query, document, answer, model_choice, query_id, debug_mode, request_delay, failed_extraction_count, in_domain_prompts_dataset)
    
    return context_score, answer_relevance_score, answer_faithfulness_score

def evaluate_documents(
    unlabeled_evaluation_set: pd.DataFrame, 
    in_domain_prompts_dataset: str, 
    context_relevance_system_prompt: str, 
    answer_relevance_system_prompt: str, 
    answer_faithfulness_system_prompt: str, 
    model_choice: str, 
    debug_mode: bool, 
    request_delay: int, 
    vllm: bool, 
    host_url: str, 
    documents: int
) -> tuple[list[float], list[float], list[float]]:
    """
    Evaluates a subset of documents for context relevance, answer relevance, and answer faithfulness.

    Parameters:
    unlabeled_evaluation_set (pd.DataFrame): The dataset containing the documents to be evaluated.
    in_domain_prompts_dataset (str): Path to the in-domain prompts dataset.
    context_relevance_system_prompt (str): System prompt for context relevance scoring.
    answer_relevance_system_prompt (str): System prompt for answer relevance scoring.
    answer_faithfulness_system_prompt (str): System prompt for answer faithfulness scoring.
    model_choice (str): The model choice for evaluation.
    debug_mode (bool): Flag to enable or disable debug mode.
    request_delay (int): Delay between requests in seconds.
    vllm (bool): Flag to indicate if vllm is used.
    host_url (str): The host URL for the model.
    documents (int): Number of documents to evaluate.

    Returns:
    tuple: A tuple containing three lists - context relevance scores, answer relevance scores, and answer faithfulness scores.
    """
    context_relevance_scores = []
    answer_relevance_scores = []
    answer_faithfulness_scores = []

    with tqdm(total=documents, desc=f"Evaluating large subset with {model_choice}") as pbar:
        for index, row in unlabeled_evaluation_set[:documents].iterrows():
            context_score, answer_relevance_score, answer_faithfulness_score = score_row(
                row, in_domain_prompts_dataset, context_relevance_system_prompt, answer_relevance_system_prompt, answer_faithfulness_system_prompt, model_choice, "Query", debug_mode, request_delay, vllm, host_url)
            
            context_relevance_scores.append(context_score)
            answer_relevance_scores.append(answer_relevance_score)
            answer_faithfulness_scores.append(answer_faithfulness_score)

            pbar.update(1)
    
    return context_relevance_scores, answer_relevance_scores, answer_faithfulness_scores

def ues_idp_config(
    in_domain_prompts_dataset: str, 
    unlabeled_evaluation_set: str, 
    context_relevance_system_prompt: str, 
    answer_relevance_system_prompt: str, 
    answer_faithfulness_system_prompt: str, 
    debug_mode: bool, 
    documents: int, 
    model_choice: str, 
    vllm: bool, 
    host_url: str, 
    request_delay: int
) -> dict:
    """
    Configures UES and IDP for evaluation.

    Parameters:
    in_domain_prompts_dataset (str): Path to the in-domain prompts dataset.
    unlabeled_evaluation_set (str): Path to the unlabeled evaluation set.
    context_relevance_system_prompt (str): System prompt for context relevance scoring.
    answer_relevance_system_prompt (str): System prompt for answer relevance scoring.
    answer_faithfulness_system_prompt (str): System prompt for answer faithfulness scoring.
    debug_mode (bool): Flag to enable or disable debug mode.
    documents (int): Number of documents to evaluate.
    model_choice (str): The model choice for evaluation.
    vllm (bool): Flag to indicate if vllm is used.
    host_url (str): The host URL for the model.
    request_delay (int): Delay between requests in seconds.

    Returns:
    dict: A dictionary containing the mean scores for context relevance, answer faithfulness, and answer relevance.
    """
    
    # Validate inputs and get the processed datasets
    in_domain_prompts_dataset, unlabeled_evaluation_set, documents = validate_inputs(
        vllm, host_url, in_domain_prompts_dataset, unlabeled_evaluation_set, documents
    )

    # Evaluate the documents and get the scores
    context_relevance_scores, answer_relevance_scores, answer_faithfulness_scores = evaluate_documents(
        unlabeled_evaluation_set, in_domain_prompts_dataset, context_relevance_system_prompt, 
        answer_relevance_system_prompt, answer_faithfulness_system_prompt, model_choice, 
        debug_mode, request_delay, vllm, host_url, documents
    )

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Context_Relevance_Score': context_relevance_scores,
        'Answer_Relevance_Score': answer_relevance_scores,
        'Answer_Faithfulness_Score': answer_faithfulness_scores
    })

    # Filter out invalid results
    valid_results_df = results_df[
        ~((results_df['Context_Relevance_Score'] == 0) & 
        ((results_df['Answer_Relevance_Score'] == 1) | 
        (results_df['Answer_Faithfulness_Score'] == 1)))
    ]
    
    # Print the number of failed extractions
    print("Number of times did not extract Yes or No:", failed_extraction_count['failed'])

    # Return the mean scores rounded to three decimal places
    return {
        "Context Relevance Scores": round(valid_results_df['Context_Relevance_Score'].mean(), 3),
        "Answer Faithfulness Scores": round(valid_results_df['Answer_Faithfulness_Score'].mean(), 3),
        "Answer Relevance Scores": round(valid_results_df['Answer_Relevance_Score'].mean(), 3)
    }