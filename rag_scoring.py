from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import begin
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import filter_dataset
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import preprocess_data
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import load_model
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_model
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import post_process_predictions
from RAG_Automatic_Evaluation.LLMJudge_RAG_Compared_Scoring import evaluate_and_scoring_data

# alpha = 0.05 
# num_trials = 1000
# evaluation_datasets = ['/future/u/manihani/ARES/example_files/evaluation_datasets.tsv']
# few_shot_examples_filepath = "example_files/few_shot_prompt_filename.tsv"
# checkpoints = ["checkpoints/microsoft-deberta-v3-large/output-synthetic_queries_1.tsv/5e-06_1_True_Context_Relevance_Label_evaluation_datasets_670487.pt"]
# labels = ["Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label"]
# GPT_scoring = False 
# swap_human_labels_for_gpt4_labels = False
# model_choice = "microsoft/deberta-v3-large"
# assigned_batch_size = 1
# number_of_labels = 2
# use_pseudo_human_labels = False
# gold_label_path = "example_files/gold_label_path.tsv"

def rag_scoring_config(alpha, num_trials, evaluation_datasets, few_shot_examples_filepath, checkpoints, labels,
GPT_scoring, swap_human_labels_for_gpt4_labels, model_choice, assigned_batch_size, number_of_labels, 
use_pseudo_human_labels, gold_label_path):

    for checkpoint, label_column in zip(checkpoints, labels):
        LLM_judge_ratio_predictions = []
        validation_set_lengths = []
        validation_set_ratios = []
        ppi_confidence_intervals = []
        accuracy_scores = []
        for test_set_selection in evaluation_datasets:

            few_shot_examples = begin(evaluation_datasets, checkpoints, labels, GPT_scoring, few_shot_examples_filepath)

            context_relevance_system_prompt, answer_faithfulness_system_prompt, answer_relevance_system_prompt = filter_dataset(evaluation_datasets)
            
            test_set, text_column = preprocess_data(test_set_selection, label_column, labels)

            model, tokenizer, device = load_model(model_choice, number_of_labels, GPT_scoring, checkpoint) 

            total_predictions, total_references, results, metric = evaluate_model(test_set, label_column, text_column, device, GPT_scoring, tokenizer, model, assigned_batch_size, model_choice)

            test_set, Y_labeled_dataset, Y_labeled_dataloader, Y_labeled_predictions, Yhat_unlabeled_dataset, prediction_column = post_process_predictions(test_set, label_column, total_predictions, labels, use_pseudo_human_labels, gold_label_path, tokenizer, assigned_batch_size, device) 

            evaluate_and_scoring_data(test_set, Y_labeled_predictions, Y_labeled_dataset, Y_labeled_dataloader, Yhat_unlabeled_dataset, alpha, num_trials, model, device, model_choice, 
            swap_human_labels_for_gpt4_labels, context_relevance_system_prompt, 
            few_shot_examples, metric, prediction_column, label_column, test_set_selection, 
            LLM_judge_ratio_predictions, validation_set_lengths, validation_set_ratios, 
            ppi_confidence_intervals, accuracy_scores, results)