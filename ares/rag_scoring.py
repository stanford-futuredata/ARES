from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import begin
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import filter_dataset
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import preprocess_data
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import load_model
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_model
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import post_process_predictions
from ares.RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_and_scoring_data

def rag_scoring_config(alpha, num_trials, evaluation_datasets, few_shot_examples_filepath, checkpoints, labels,
GPT_scoring, annotated_datapoints_filepath, model_choice, llm_judge, assigned_batch_size, number_of_labels, gold_label_path, rag_type):
    # Validate inputs and determine model loading strategy
    if checkpoints:
        if llm_judge and llm_judge != "None":
            print("Warning: Both checkpoint and llm_judge were provided. Using checkpoints.")
        model_loader = lambda chk: load_model(model_choice, number_of_labels, chk)
    elif llm_judge and llm_judge != "None":
        model_loader = lambda _: load_model(llm_judge, number_of_labels, None)
    else:
        raise ValueError("No valid model or checkpoint provided.")

    # Use zip only if checkpoints are not empty, otherwise assume only llm_judge is used
    if checkpoints:
        # Here we assume that the length of checkpoints and labels is the same
        pairings = zip(checkpoints, labels)
    else:
        # If no checkpoints, create dummy pairs for labels with None for checkpoint
        pairings = ((None, label) for label in labels)

    for checkpoint, label_column in pairings:
        LLM_judge_ratio_predictions = []
        validation_set_lengths = []
        validation_set_ratios = []
        ppi_confidence_intervals = []
        accuracy_scores = []
        for test_set_selection in evaluation_datasets:

            few_shot_examples = begin(evaluation_datasets, checkpoints, labels, GPT_scoring, few_shot_examples_filepath)

            context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt = filter_dataset(rag_type)
            
            test_set, text_column = preprocess_data(test_set_selection, label_column, labels)

            model, tokenizer, device = model_loader(checkpoint)

            total_predictions, total_references, results, metric = evaluate_model(test_set, label_column, text_column, device, checkpoint, tokenizer, model, assigned_batch_size, model_choice, context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt, few_shot_examples_filepath, llm_judge)

            test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column = post_process_predictions(checkpoint, test_set, label_column, total_predictions, labels, gold_label_path, tokenizer, assigned_batch_size, device) 

            evaluate_and_scoring_data(test_set, Y_labeled_predictions, Y_labeled_dataset, Y_labeled_dataloader, Yhat_unlabeled_dataset, alpha, num_trials, model, device, model_choice, 
            annotated_datapoints_filepath, context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt,
            few_shot_examples, metric, prediction_column, label_column, test_set_selection, 
            LLM_judge_ratio_predictions, validation_set_lengths, validation_set_ratios, 
            ppi_confidence_intervals, accuracy_scores, results, checkpoint, llm_judge)

# if not checkpoints and not llm_judge:
#         raise ValueError("Either checkpoints or an llm_model must be provided.")

#     if checkpoints:
#         if len(checkpoints) != len(evaluation_datasets):
#             raise ValueError("The number of checkpoints must match the number of evaluation datasets.")
#         models_to_use = checkpoints
#     else:
#         models_to_use = [llm_judge] * len(evaluation_datasets)

#     for model_to_use, test_set_selection in zip(models_to_use, evaluation_datasets):
#         for label_column in labels:
#             LLM_judge_ratio_predictions = []
#             validation_set_lengths = []
#             validation_set_ratios = []
#             ppi_confidence_intervals = []
#             accuracy_scores = []

#             few_shot_examples = begin(evaluation_datasets, model_to_use, labels, GPT_scoring, few_shot_examples_filepath)

#             context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt = filter_dataset(rag_type)
            
#             test_set, text_column = preprocess_data(test_set_selection, label_column, labels)

#             model, tokenizer, device = load_model(llm_judge, number_of_labels, GPT_scoring, model_to_use) 

#             total_predictions, total_references, results, metric = evaluate_model(test_set, label_column, text_column, device, GPT_scoring, tokenizer, model, assigned_batch_size, llm_judge, context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt)

#             test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column = post_process_predictions(test_set, label_column, total_predictions, labels, use_pseudo_human_labels, gold_label_path, tokenizer, assigned_batch_size, device) 

#             evaluate_and_scoring_data(test_set, Y_labeled_predictions, Y_labeled_dataset, Y_labeled_dataloader, Yhat_unlabeled_dataset, alpha, num_trials, model, device, llm_judge, 
#             annotated_datapoints_filepath, context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt,
#             few_shot_examples, metric, prediction_column, label_column, test_set_selection, 
#             LLM_judge_ratio_predictions, validation_set_lengths, validation_set_ratios, 
#             ppi_confidence_intervals, accuracy_scores, results)