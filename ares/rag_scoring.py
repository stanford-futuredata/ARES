from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import begin
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import filter_dataset
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import preprocess_data
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import load_api_model 
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import load_tokenizer_and_model
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_model
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import post_process_predictions
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_and_scoring_data
import torch

machine_label_system_prompt = (
    "Given the following question and document, you must analyze the provided document "
    "and determine whether it is sufficient for answering the question. In your evaluation, "
    "you should consider the content of the document and whether it contains the answer to "
    "the provided question. Output your final verdict by strictly following this format: "
    "'[[Yes]]' if the document is sufficient and '[[No]]' if the document provided is not sufficient."
)

def rag_scoring_config(alpha, num_trials, evaluation_datasets, few_shot_examples_filepath, checkpoints, labels,
    model_choice, llm_judge, assigned_batch_size, number_of_labels, gold_label_paths, rag_type, vllm, host_url, request_delay, debug_mode, 
    machine_label_llm_model, gold_machine_label_path, prediction_filepaths, azure_openai_config):
    """
    Configures and runs the RAG scoring process.

    Parameters:
    - alpha: The alpha value for the scoring process.
    - num_trials: Number of trials to run.
    - evaluation_datasets: List of datasets to evaluate.
    - few_shot_examples_filepath: Filepath for few-shot examples.
    - checkpoints: List of model checkpoints.
    - labels: List of labels.
    - model_choice: Choice of model.
    - llm_judge: LLM judge to use.
    - assigned_batch_size: Batch size to use.
    - number_of_labels: Number of labels.
    - gold_label_paths: List of paths to the gold labels.
    - rag_type: Type of RAG.
    - vllm: VLLM to use.
    - host_url: Host URL.
    - request_delay: Delay between requests.
    - debug_mode: Whether to run in debug mode.
    - machine_label_llm_model: Machine label LLM model.
    - gold_machine_label_path: Path to the gold machine labels.
    - prediction_filepaths: List of file paths to save predictions.
    - azure_openai_config: Dictionary of information to setup Azure model
    """
    
    if few_shot_examples_filepath == "None" and (llm_judge != "None" or machine_label_llm_model != "None"):
        raise ValueError("'few_shot_examples_filepath' cannot be None if generating machine labels.")
    
    # Validate if either gold_label_paths or gold_machine_label_path is provided
    if gold_label_paths == ["None"] and gold_machine_label_path == "None":
        raise ValueError("Either 'gold_label_paths' or 'gold_machine_label_path' must be provided.")

    # Validate inputs and determine model loading strategy
    if checkpoints:
        if (llm_judge and llm_judge != "None") or azure_openai_config:
            print("Warning: Both checkpoint and llm_judge/azure openai model were provided. Using checkpoints.")
        model_loader = lambda chk: load_tokenizer_and_model(model_choice, number_of_labels, chk)
    # elif azure_openai_config:
    #     if llm_judge and llm_judge != "None":
    #         print("Warning: Both azure openai model and llm_judge were provided. Using azure openai model.")
    #     model_loader = lambda _: load_azure_model(azure_openai_config)
    elif llm_judge and llm_judge != "None":
        model_loader = lambda _: load_api_model(llm_judge, vllm)
    else:
        raise ValueError("No valid model or checkpoint provided.")

    # Use zip only if checkpoints are not empty, otherwise assume only llm_judge is used
    if checkpoints:
        # Here we assume that the length of checkpoints and labels is the same
        pairings = zip(checkpoints, labels)
    else:
        # If no checkpoints, create dummy pairs for labels with None for checkpoint
        pairings = ((None, label) for label in labels)

    all_evaluation_results = []
    
    for idx, (checkpoint, label_column) in enumerate(pairings):

        chekpoint_results = []

        LLM_judge_ratio_predictions = []
        validation_set_lengths = []
        validation_set_ratios = []
        ppi_confidence_intervals = []
        accuracy_scores = []
        for test_set_idx, test_set_selection in enumerate(evaluation_datasets):

            few_shot_examples = begin(evaluation_datasets, checkpoints, labels, few_shot_examples_filepath)

            context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt = filter_dataset(rag_type)
            
            test_set, text_column = preprocess_data(test_set_selection, label_column, labels)

            loaded_model = model_loader(checkpoint)
            if isinstance(loaded_model, tuple):
                model, tokenizer, device = loaded_model
            else:
                model = loaded_model
                tokenizer = None
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            eval_model_settings = {
                "test_set": test_set,
                "label_column": label_column,
                "text_column": text_column,
                "device": device,
                "checkpoint": checkpoint,
                "tokenizer": tokenizer,
                "model": model,
                "assigned_batch_size": assigned_batch_size,
                "model_choice": model_choice,
                "context_relevance_system_prompt": context_relevance_system_prompt,
                "answer_faithfulness_system_prompt": answer_faithfulness_system_prompt,
                "answer_relevance_system_prompt": answer_relevance_system_prompt,
                "few_shot_examples_filepath": few_shot_examples_filepath,
                "llm_judge": llm_judge,
                "vllm": vllm,
                "host_url": host_url,
                "request_delay": request_delay,
                "debug_mode": debug_mode,
                "azure_openai_config": azure_openai_config
            }

            total_predictions, total_references, results, metric = evaluate_model(eval_model_settings)
            
            post_process_settings = {
                "checkpoint": checkpoint,
                "test_set": test_set,
                "label_column": label_column,
                "total_predictions": total_predictions,
                "labels": labels,
                "gold_label_path": gold_label_paths[test_set_idx] if test_set_idx < len(gold_label_paths) else gold_label_paths[-1],
                "tokenizer": tokenizer,
                "assigned_batch_size": assigned_batch_size,
                "device": device,
                "gold_machine_label_path": gold_machine_label_path,
                "machine_label_system_prompt": machine_label_system_prompt,
                "machine_label_llm_model": machine_label_llm_model,
                "vllm": vllm,
                "host_url": host_url,
                "debug_mode": debug_mode,
                "request_delay": request_delay,
                "few_shot_examples": few_shot_examples,
                "azure_openai_config": azure_openai_config
            }

            test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column = post_process_predictions(post_process_settings) 
            
            evaluate_scoring_settings = {
                "test_set": test_set,
                "Y_labeled_predictions": Y_labeled_predictions,
                "Y_labeled_dataset": Y_labeled_dataset,
                "Y_labeled_dataloader": Y_labeled_dataloader,
                "Yhat_unlabeled_dataset": Yhat_unlabeled_dataset,
                "alpha": alpha,
                "num_trials": num_trials,
                "model": model,
                "device": device,
                "model_choice": model_choice,
                "context_relevance_system_prompt": context_relevance_system_prompt,
                "answer_faithfulness_system_prompt": answer_faithfulness_system_prompt,
                "answer_relevance_system_prompt": answer_relevance_system_prompt,
                "few_shot_examples": few_shot_examples,
                "metric": metric,
                "prediction_column": prediction_column,
                "label_column": label_column,
                "test_set_selection": test_set_selection,
                "LLM_judge_ratio_predictions": LLM_judge_ratio_predictions,
                "validation_set_lengths": validation_set_lengths,
                "validation_set_ratios": validation_set_ratios,
                "ppi_confidence_intervals": ppi_confidence_intervals,
                "accuracy_scores": accuracy_scores,
                "results": results,
                "checkpoint": checkpoint,
                "llm_judge": llm_judge,
                "vllm": vllm,
                "host_url": host_url,
                "request_delay": request_delay,
                "debug_mode": debug_mode,
                "prediction_filepath": prediction_filepaths[test_set_idx] if test_set_idx < len(prediction_filepaths) else prediction_filepaths[-1],
                "azure_openai_config": azure_openai_config
            }
            dataset_results = evaluate_and_scoring_data(evaluate_scoring_settings)
        
        all_evaluation_results.append(dataset_results)
            
    return all_evaluation_results