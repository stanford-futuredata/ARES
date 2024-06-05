from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import (
    load_model,
    load_documents,
    load_few_shot_prompt,
    generate_contradictory_answers,
    generate_few_shot_prompts,
    generate_synthetic_queries,
    Generate_Synthetic_Answers
)

def synthetic_generator_config(
    document_filepaths: list, 
    few_shot_prompt_filename: str,
    synthetic_queries_filenames: list, 
    documents_sampled: int,
    model_choice: str = "google/flan-t5-xxl", 
    clean_documents: bool = False,
    regenerate_synth_questions: bool = True, 
    percentiles: list = [0.05, 0.25, 0.5, 0.95], 
    question_temperatures: list = [2.0, 1.5, 1.0, 0.5, 0.0],
    regenerate_answers: bool = True,
    number_of_negatives_added_ratio: float = 0.5, 
    lower_bound_for_negatives: int = 5, 
    number_of_contradictory_answers_added_ratio: float = 0.67, 
    number_of_positives_added_ratio: float = 0.0, 
    regenerate_embeddings: bool = True, 
    synthetic_query_prompt: str = (
        "You are an expert question-answering system. You must create a question for the provided document. "
        "The question must be answerable within the context of the document.\n\n"
    )
) -> None:
    """
    Configures and generates synthetic queries and answers based on the provided parameters.

    Args:
        document_filepaths (list): List of file paths to the documents.
        few_shot_prompt_filename (str): Filename for the few-shot prompt.
        synthetic_queries_filenames (list): List of filenames for the synthetic queries.
        documents_sampled (int): Number of documents to sample.
        model_choice (str, optional): Model choice for the generation. Defaults to "google/flan-t5-xxl".
        clean_documents (bool, optional): Whether to clean the documents. Defaults to False.
        regenerate_synth_questions (bool, optional): Whether to regenerate synthetic questions. Defaults to True.
        percentiles (list, optional): List of percentiles for the generation. Defaults to [0.05, 0.25, 0.5, 0.95].
        question_temperatures (list, optional): List of temperatures for question generation. Defaults to [2.0, 1.5, 1.0, 0.5, 0.0].
        regenerate_answers (bool, optional): Whether to regenerate answers. Defaults to True.
        number_of_negatives_added_ratio (float, optional): Ratio of negatives to add. Defaults to 0.5.
        lower_bound_for_negatives (int, optional): Lower bound for negatives. Defaults to 5.
        number_of_contradictory_answers_added_ratio (float, optional): Ratio of contradictory answers to add. Defaults to 0.67.
        number_of_positives_added_ratio (float, optional): Ratio of positives to add. Defaults to 0.0.
        regenerate_embeddings (float, optional): Whether to regenerate embeddings. Defaults to True.
        synthetic_query_prompt (str, optional): Prompt for synthetic query generation. Defaults to a predefined string.

    Raises:
        ValueError: If the lengths of document_filepaths and synthetic_queries_filenames do not match.
    """
    
    model, tokenizer, device = load_model(model_choice)

    if len(document_filepaths) != len(synthetic_queries_filenames):
        raise ValueError("document_filepaths and synthetic_queries_filenames lists must be of the same length.")

    for document_filepath, synthetic_queries_filename in zip(document_filepaths, synthetic_queries_filenames):
        for_fever_dataset = "fever" in document_filepath.lower()
        for_wow_dataset = "wow" in document_filepath.lower()

        documents = load_documents(document_filepath, clean_documents, documents_sampled)

        few_shot_examples, length_of_fewshot_prompt = load_few_shot_prompt(
            few_shot_prompt_filename, for_fever_dataset, for_wow_dataset
        )

        few_shot_examples_for_contradictory_answers = generate_contradictory_answers(
            few_shot_prompt_filename, for_fever_dataset, for_wow_dataset
        )

        answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen = generate_few_shot_prompts(
            few_shot_prompt_filename, for_fever_dataset, for_wow_dataset
        )

        synthetic_queries_config = {
            'few_shot_examples': few_shot_examples,
            'length_of_fewshot_prompt': length_of_fewshot_prompt,
            'device': device,
            'tokenizer': tokenizer,
            'model': model,
            'percentiles': percentiles,
            'for_fever_dataset': for_fever_dataset,
            'for_wow_dataset': for_wow_dataset,
            'synthetic_query_prompt': synthetic_query_prompt,
            'synthetic_queries_filename': synthetic_queries_filename,
            'question_temperatures': question_temperatures
        }

        generate_synthetic_queries(documents, synthetic_queries_config)

        synthetic_answers_config = {
            'regenerate_answers': regenerate_answers,
            'answer_gen_few_shot_examples': answer_gen_few_shot_examples,
            'length_of_fewshot_prompt_answer_gen': length_of_fewshot_prompt_answer_gen,
            'device': device,
            'tokenizer': tokenizer,
            'model': model,
            'for_fever_dataset': for_fever_dataset,
            'for_wow_dataset': for_wow_dataset,
            'few_shot_examples_for_contradictory_answers': few_shot_examples_for_contradictory_answers,
            'number_of_negatives_added_ratio': number_of_negatives_added_ratio,
            'lower_bound_for_negatives': lower_bound_for_negatives,
            'number_of_contradictory_answers_added_ratio': number_of_contradictory_answers_added_ratio,
            'number_of_positives_added_ratio': number_of_positives_added_ratio,
            'regenerate_embeddings': regenerate_embeddings
        }

        Generate_Synthetic_Answers(synthetic_queries_filename, synthetic_answers_config)
