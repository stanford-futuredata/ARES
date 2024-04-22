from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import load_model
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import load_documents
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import load_few_shot_prompt
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import generate_contradictory_answers
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import generate_few_shot_prompts
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import save_synthetic_queries
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import Generate_Synthetic_Answers
from .LLM_as_a_Judge_Adaptation.Generate_Synthetic_Queries_and_Answers import print_synthetic_queries

def synthetic_generator_config(document_filepaths: list, few_shot_prompt_filename: str,
                               synthetic_queries_filenames: list, documents_sampled: int,
                               model_choice: str = "google/flan-t5-xxl", flan_approach: bool = True, clean_documents: bool = False,
                               regenerate_synth_questions: bool = True, 
                               percentiles: list = [0.05, 0.25, 0.5, 0.95], 
                               question_temperatures: list = [2.0, 1.5, 1.0, 0.5, 0.0],
                               regenerate_answers: bool = True,
                               generate_contradictory_answers_with_flan: bool = True, 
                               number_of_negatives_added_ratio: float = 0.5, # Check whether can also be an int
                               lower_bound_for_negatives: int = 5, # Need to be an int value
                               number_of_contradictory_answers_added_ratio: float = 0.67, # Check whether can also be an int
                               number_of_positives_added_ratio: float = 0.0, # Check whether can also be an int
                               regenerate_embeddings: float = True, 
                               synthetic_query_prompt: str = "You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.\n\n"
                               ): 
    
    model, tokenizer, device = load_model(flan_approach, model_choice)

    if len(document_filepaths) != len(synthetic_queries_filenames):
        raise ValueError("document_filepaths and synthetic_queries_filenames lists must be of the same length.")

    for document_filepath, synthetic_queries_filename in zip(document_filepaths, synthetic_queries_filenames):
        for_fever_dataset = False
        if "fever" in document_filepath.lower():
            for_fever_dataset = True
        for_wow_dataset = False
        if "wow" in document_filepath.lower():
            for_wow_dataset = True

        documents = load_documents(document_filepath, clean_documents, documents_sampled)

        few_shot_examples, length_of_fewshot_prompt = load_few_shot_prompt(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset)

        few_shot_examples_for_contradictory_answers = generate_contradictory_answers(few_shot_prompt_filename,generate_contradictory_answers_with_flan,for_fever_dataset,for_wow_dataset)

        answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen = generate_few_shot_prompts(few_shot_prompt_filename,for_fever_dataset,for_wow_dataset)

        save_synthetic_queries(documents, regenerate_synth_questions, flan_approach, few_shot_examples, 
        length_of_fewshot_prompt, device, tokenizer, model, percentiles, for_fever_dataset, for_wow_dataset, 
        synthetic_query_prompt, synthetic_queries_filename, question_temperatures)

        Generate_Synthetic_Answers(synthetic_queries_filename,
        regenerate_answers, flan_approach, answer_gen_few_shot_examples, length_of_fewshot_prompt_answer_gen, 
        device, tokenizer, model, for_fever_dataset, for_wow_dataset, generate_contradictory_answers_with_flan,
        few_shot_examples_for_contradictory_answers, number_of_negatives_added_ratio, lower_bound_for_negatives, number_of_contradictory_answers_added_ratio, number_of_positives_added_ratio, regenerate_embeddings)

        # print_synthetic_queries(synthetic_queries_filename)
    